"""
dashboard_logger.py
--------------------
Lightweight, TensorBoard-style live dashboard for the gait optimizer.

Each parallel experiment (one mode / direction / body-height / ... combo)
gets its own run folder under `runs/`. Every Optuna generation logs:
  - this generation's best / mean / worst / std score
  - the all-time-best score so far
  - a histogram of the generation's score distribution (if available)
  - the current all-time-best DNA (step_amplitude, frequency, ...)
  - the all-time-best robot's quality metrics (speed match, stability, etc.)

This is most useful in "headless" render mode, where there's no PyBullet
GUI to watch — point TensorBoard at the `runs/` folder and watch curves
update live while the optimization runs in the background.

Works with either:
  - tensorboardX (pip install tensorboardX), if present — full featured
    (scalars + histograms), or
  - plain tensorboard (pip install tensorboard) alone — no extra
    dependency needed, scalars only (used automatically as a fallback).

View:
    tensorboard --logdir runs
  or
    python launch_dashboard.py
"""

import os
import time
import numpy as np

_BACKEND = None  # "tensorboardx", "tensorboard", or None

try:
    from tensorboardX import SummaryWriter as _TBXWriter
    _BACKEND = "tensorboardx"
except ImportError:
    try:
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        _BACKEND = "tensorboard"
    except ImportError:
        _BACKEND = None

_WARNED = False


def _warn_once():
    global _WARNED
    if not _WARNED:
        print(">>> [dashboard] Neither tensorboardX nor tensorboard is installed — "
              "dashboard logging disabled. Run: pip install tensorboard")
        _WARNED = True


class _PureTBWriter:
    """Minimal scalar-only writer built directly on the `tensorboard` package's
    own event-file machinery. Used when tensorboardX isn't installed."""

    def __init__(self, log_dir):
        self._writer = EventFileWriter(log_dir)

    def add_scalar(self, tag, value, step):
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])
        event = Event(summary=summary, step=step, wall_time=time.time())
        self._writer.add_event(event)

    def add_histogram(self, *args, **kwargs):
        raise NotImplementedError  # not supported without tensorboardX; caller ignores this

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()


class DashboardLogger:
    """One instance per worker/experiment. Silently becomes a no-op if
    neither tensorboardX nor tensorboard is installed, so it never blocks a run."""

    def __init__(self, run_name, log_root="runs", enabled=True):
        self.enabled = bool(enabled) and (_BACKEND is not None)
        self.writer = None

        if enabled and _BACKEND is None:
            _warn_once()

        if self.enabled:
            log_dir = os.path.join(log_root, run_name)
            os.makedirs(log_dir, exist_ok=True)
            if _BACKEND == "tensorboardx":
                self.writer = _TBXWriter(log_dir=log_dir)
            else:
                self.writer = _PureTBWriter(log_dir)

    def log_generation(self, gen, fitnesses, best_overall_score,
                        best_overall_dna, best_overall_metrics):
        if not self.enabled:
            return

        fitnesses = np.asarray(fitnesses, dtype=float)

        self.writer.add_scalar("score/best_this_gen", float(fitnesses.max()), gen)
        self.writer.add_scalar("score/mean_this_gen", float(fitnesses.mean()), gen)
        self.writer.add_scalar("score/worst_this_gen", float(fitnesses.min()), gen)
        self.writer.add_scalar("score/std_this_gen", float(fitnesses.std()), gen)
        self.writer.add_scalar("score/all_time_best", float(best_overall_score), gen)

        try:
            self.writer.add_histogram("score/distribution", fitnesses, gen)
        except Exception:
            pass  # not supported by the pure-tensorboard fallback, or degenerate input — safe to skip

        if best_overall_dna:
            for gene, value in best_overall_dna.items():
                self.writer.add_scalar(f"dna/{gene}", float(value), gen)

        if best_overall_metrics:
            for name, value in best_overall_metrics.items():
                try:
                    self.writer.add_scalar(f"metrics/{name}", float(value), gen)
                except (TypeError, ValueError):
                    continue  # non-numeric metric, skip

        self.writer.flush()

    def close(self):
        if self.enabled and self.writer is not None:
            self.writer.close()
