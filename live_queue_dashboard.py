"""
live_queue_dashboard.py
--------------------------
Alternative to dashboard_logger.py + aggregate_dashboard.py: instead of
each worker process writing its own TensorBoard run to disk (which is
what produces the "_exp0, _exp1, ... _MEAN" pile of runs), workers send
their per-generation results through a shared multiprocessing Queue to a
single listener living in the MAIN process. That listener combines
reports from all repeats of the same config on the fly and writes to
ONE TensorBoard run per config — nothing else touches disk.

Combination rules (same as aggregate_dashboard.py):
  - "best"-type tags (score/best_this_gen, score/all_time_best) -> MAX
  - "worst"-type tags (score/worst_this_gen)                    -> MIN
  - everything else (mean/std scores, dna/*, metrics/*)         -> MEAN

Usage:

    # main process, before ProcessPoolExecutor:
    manager = multiprocessing.Manager()
    dashboard_queue = manager.Queue()
    listener = DashboardQueueListener(dashboard_queue, log_root="runs")
    listener.start()

    # ... pass dashboard_queue into each task tuple ...
    # inside run_experiment_worker (runs in a worker process):
    dashboard = QueueDashboardLogger(dashboard_queue, group_key, exp_idx)
    ...
    dashboard.log_generation(gen, fitnesses, best_overall_score,
                              best_overall_dna, best_overall_metrics)
    ...
    dashboard.close()

    # main process, after ProcessPoolExecutor finishes:
    listener.stop()
"""

import threading
from collections import defaultdict

from dashboard_logger import DashboardLogger


def _aggregator_for_tag(tag):
    lower = tag.lower()
    if "best" in lower:
        return max
    if "worst" in lower:
        return min
    return lambda values: sum(values) / len(values)


class QueueDashboardLogger:
    """Used inside a worker process. Sends per-generation results through
    a shared Queue instead of writing to disk itself. No-op if disabled
    or if no queue is provided."""

    def __init__(self, queue, group_key, exp_idx, enabled=True):
        self.queue = queue
        self.group_key = group_key
        self.exp_idx = exp_idx
        self.enabled = bool(enabled) and (queue is not None)

    def log_generation(self, gen, fitnesses, best_overall_score,
                        best_overall_dna, best_overall_metrics):
        if not self.enabled:
            return
        try:
            self.queue.put({
                "group": self.group_key,
                "exp": self.exp_idx,
                "gen": gen,
                "fitnesses": [float(f) for f in fitnesses],
                "best_overall_score": float(best_overall_score),
                "best_overall_dna": dict(best_overall_dna) if best_overall_dna else {},
                "best_overall_metrics": dict(best_overall_metrics) if best_overall_metrics else {},
            })
        except Exception:
            pass  # never let dashboard plumbing break the optimization run

    def close(self):
        pass  # nothing to close on the worker side — the writer lives in the main process


class DashboardQueueListener:
    """
    Lives in the main process. Consumes messages put on `queue` by
    QueueDashboardLogger instances running in worker processes, combines
    repeats of the same config as they arrive, and writes ONE TensorBoard
    run per config (named after group_key, no "_exp"/"_MEAN" suffix).

    A given generation for a group is only written once at least
    `min_repeats` experiments have reported it, and is never rewritten
    after that — so the curve grows forward cleanly, live, while the
    experiments are still running.
    """

    def __init__(self, queue, log_root="runs", min_repeats=2, poll_timeout=1.0):
        self.queue = queue
        self.log_root = log_root
        self.min_repeats = min_repeats
        self.poll_timeout = poll_timeout

        self._writers = {}                                     # group -> DashboardLogger
        self._written_gens = defaultdict(set)                   # group -> {gen, ...} already emitted
        self._pending = defaultdict(lambda: defaultdict(dict))  # group -> gen -> {exp_idx: message}

        self._stop_event = threading.Event()
        self._thread = None

    def _get_writer(self, group_key):
        if group_key not in self._writers:
            self._writers[group_key] = DashboardLogger(group_key, log_root=self.log_root, enabled=True)
        return self._writers[group_key]

    def _process_message(self, msg):
        group, gen, exp = msg["group"], msg["gen"], msg["exp"]
        self._pending[group][gen][exp] = msg

        reports = self._pending[group][gen]
        if len(reports) < self.min_repeats or gen in self._written_gens[group]:
            return

        writer = self._get_writer(group)
        if not writer.enabled:
            self._written_gens[group].add(gen)
            return

        reports_list = list(reports.values())

        best_this_gen = [max(m["fitnesses"]) for m in reports_list]
        mean_this_gen = [sum(m["fitnesses"]) / len(m["fitnesses"]) for m in reports_list]
        worst_this_gen = [min(m["fitnesses"]) for m in reports_list]
        all_time_best = [m["best_overall_score"] for m in reports_list]

        writer.writer.add_scalar("score/best_this_gen", max(best_this_gen), gen)
        writer.writer.add_scalar("score/mean_this_gen", sum(mean_this_gen) / len(mean_this_gen), gen)
        writer.writer.add_scalar("score/worst_this_gen", min(worst_this_gen), gen)
        writer.writer.add_scalar("score/all_time_best", max(all_time_best), gen)

        dna_keys = set().union(*(m["best_overall_dna"].keys() for m in reports_list))
        for key in dna_keys:
            vals = [m["best_overall_dna"][key] for m in reports_list if key in m["best_overall_dna"]]
            if vals:
                writer.writer.add_scalar(f"dna/{key}", sum(vals) / len(vals), gen)

        metric_keys = set().union(*(m["best_overall_metrics"].keys() for m in reports_list))
        for key in metric_keys:
            vals = [m["best_overall_metrics"][key] for m in reports_list if key in m["best_overall_metrics"]]
            if vals:
                try:
                    writer.writer.add_scalar(f"metrics/{key}", sum(vals) / len(vals), gen)
                except (TypeError, ValueError):
                    pass

        writer.writer.flush()
        self._written_gens[group].add(gen)
        del self._pending[group][gen]  # free memory, done with this generation

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                msg = self.queue.get(timeout=self.poll_timeout)
            except Exception:
                continue
            self._process_message(msg)

    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        # Drain whatever's left in the queue before shutting down.
        while True:
            try:
                msg = self.queue.get_nowait()
            except Exception:
                break
            self._process_message(msg)

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_timeout + 5.0)
            self._thread = None

        for writer in self._writers.values():
            writer.close()
