"""
aggregate_dashboard.py
------------------------
TensorBoard does NOT average across separate runs on its own — it only
smooths within a single run's curve. If NUM_EXPERIMENTS > 1 (repeated
trials of the same mode/direction/height/... config), each repeat is
logged as its own run (e.g. "GA_straight_dir1_h0.2_exp0", "..._exp1", ...)
and TensorBoard just overlays them as separate lines.

This module combines them into one "<config>_MEAN" run per group — but
"MEAN" isn't applied blindly to every tag. Different indicators need
different combination rules:

  - tags with "best" in the name (score/best_this_gen, score/all_time_best)
    are combined with MAX — you want the best result found across all
    experiments, not the average of everyone's best.
  - tags with "worst" in the name (score/worst_this_gen) are combined
    with MIN, for the same reason in the other direction.
  - everything else (mean/std scores, dna/*, metrics/*) is combined with
    the plain arithmetic MEAN, which is what you want for population-level
    trend indicators.

Two ways to use it:

1. aggregate_runs() — a one-shot pass, good for re-aggregating after the
   fact or as a standalone script (`python aggregate_dashboard.py`).

2. LiveDashboardAggregator — a background thread that polls the same way
   every few seconds *while experiments are still running*, so the
   combined curve fills in live alongside the individual per-experiment
   curves. It only ever writes a given (tag, step) once it's seen, so the
   curve never gets rewritten or duplicated as more data arrives.
"""

import os
import re
import threading
from collections import defaultdict

from dashboard_logger import DashboardLogger

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    _READER_AVAILABLE = True
except ImportError:
    _READER_AVAILABLE = False

_WARNED = False


def _warn_once():
    global _WARNED
    if not _WARNED:
        print(">>> [dashboard] tensorboard not installed — can't read back runs to aggregate. "
              "Run: pip install tensorboard")
        _WARNED = True


def _group_key(run_name):
    """Strip the trailing '_expN' so repeats of the same config group together."""
    return re.sub(r"_exp\d+$", "", run_name)


def _read_run_scalars(log_root, run_name):
    """tag -> {step: value} for one run directory."""
    ea = EventAccumulator(os.path.join(log_root, run_name))
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = {event.step: event.value for event in ea.Scalars(tag)}
    return out


def _scan_groups(log_root):
    """group_key -> [run_name, ...], excluding already-aggregated runs."""
    if not os.path.isdir(log_root):
        return {}
    run_dirs = [d for d in os.listdir(log_root)
                if os.path.isdir(os.path.join(log_root, d)) and not d.endswith("_MEAN")]
    groups = defaultdict(list)
    for run_name in run_dirs:
        groups[_group_key(run_name)].append(run_name)
    return groups


def _aggregator_for_tag(tag):
    """Pick the combination rule for a tag based on its name."""
    lower = tag.lower()
    if "best" in lower:
        return max
    if "worst" in lower:
        return min
    return lambda values: sum(values) / len(values)


def _aggregate_group(per_run_data, min_repeats):
    """
    per_run_data: {run_name: {tag: {step: value}}}
    Returns {tag: {step: aggregated_value}}, using max/min/mean per tag
    (see _aggregator_for_tag), only for (tag, step) combos reported by at
    least `min_repeats` runs.
    """
    per_tag_step_values = defaultdict(lambda: defaultdict(list))
    for tags in per_run_data.values():
        for tag, step_values in tags.items():
            for step, value in step_values.items():
                per_tag_step_values[tag][step].append(value)

    result = defaultdict(dict)
    for tag, step_values in per_tag_step_values.items():
        agg_fn = _aggregator_for_tag(tag)
        for step, values in step_values.items():
            if len(values) < min_repeats:
                continue
            result[tag][step] = float(agg_fn(values))
    return result


def aggregate_runs(log_root="runs", min_repeats=2):
    """
    One-shot pass: scan `log_root`, group repeats of the same config, and
    write one '<config>_MEAN' run per group with the combined value of
    every scalar tag (max for "best" tags, min for "worst" tags, mean for
    everything else), aligned by generation (step).

    Groups with fewer than `min_repeats` runs are skipped (nothing to
    combine). Suitable for calling once after a batch finishes, or as a
    standalone re-aggregation pass.
    """
    if not _READER_AVAILABLE:
        _warn_once()
        return

    groups = _scan_groups(log_root)

    for group_key, run_names in groups.items():
        if len(run_names) < min_repeats:
            continue

        per_run_data = {rn: _read_run_scalars(log_root, rn) for rn in run_names}
        combined = _aggregate_group(per_run_data, min_repeats)

        mean_run_name = f"{group_key}_MEAN"
        logger = DashboardLogger(mean_run_name, log_root=log_root, enabled=True)
        if not logger.enabled:
            continue

        for tag, step_values in combined.items():
            for step in sorted(step_values):
                logger.writer.add_scalar(tag, step_values[step], step)

        logger.writer.flush()
        logger.close()

        print(f">>> [dashboard] Combined {len(run_names)} repeats of '{group_key}' -> {mean_run_name}")


class LiveDashboardAggregator:
    """
    Background poller that keeps a '<config>_MEAN' run updated WHILE
    experiments are still running, so the combined curve fills in live
    next to the individual per-experiment curves — not just after the
    whole batch finishes.

    Uses max for "best"-type tags, min for "worst"-type tags, and mean for
    everything else (see _aggregator_for_tag). Each (tag, step) is written
    at most once, the first time at least `min_repeats` experiments have
    reported it, so the curve never jumps backward or gets duplicated as
    later repeats catch up.

    Usage:
        live_agg = LiveDashboardAggregator(log_root="runs")
        live_agg.start()
        ... run your ProcessPoolExecutor batch ...
        live_agg.stop()   # does one final poll before stopping
    """

    def __init__(self, log_root="runs", poll_interval=10.0, min_repeats=2):
        self.log_root = log_root
        self.poll_interval = poll_interval
        self.min_repeats = min_repeats

        self._writers = {}                     # group_key -> DashboardLogger
        self._written_steps = defaultdict(lambda: defaultdict(set))  # group_key -> tag -> {steps}

        self._stop_event = threading.Event()
        self._thread = None

    def _get_writer(self, group_key):
        if group_key not in self._writers:
            self._writers[group_key] = DashboardLogger(
                f"{group_key}_MEAN", log_root=self.log_root, enabled=True
            )
        return self._writers[group_key]

    def _poll_once(self):
        if not _READER_AVAILABLE:
            _warn_once()
            return

        groups = _scan_groups(self.log_root)

        for group_key, run_names in groups.items():
            if len(run_names) < self.min_repeats:
                continue

            per_run_data = {rn: _read_run_scalars(self.log_root, rn) for rn in run_names}
            combined = _aggregate_group(per_run_data, self.min_repeats)

            logger = self._get_writer(group_key)
            if not logger.enabled:
                continue

            wrote_any = False
            for tag, step_values in combined.items():
                already = self._written_steps[group_key][tag]
                for step in sorted(step_values):
                    if step in already:
                        continue
                    logger.writer.add_scalar(tag, step_values[step], step)
                    already.add(step)
                    wrote_any = True

            if wrote_any:
                logger.writer.flush()

    def _run_loop(self):
        while not self._stop_event.wait(self.poll_interval):
            self._poll_once()

    def start(self):
        if self._thread is not None:
            return  # already running
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.poll_interval + 5.0)
        self._thread = None

        self._poll_once()  # final catch-up pass so nothing's missed

        for logger in self._writers.values():
            logger.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Combine repeated-experiment TensorBoard runs.")
    parser.add_argument("--logdir", default="runs", help="Directory containing the run logs (default: runs)")
    parser.add_argument("--min-repeats", type=int, default=2,
                         help="Minimum number of repeats required to produce a combined run (default: 2)")
    args = parser.parse_args()
    aggregate_runs(log_root=args.logdir, min_repeats=args.min_repeats)
