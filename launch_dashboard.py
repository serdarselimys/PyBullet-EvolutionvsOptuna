"""
launch_dashboard.py
--------------------
Starts TensorBoard against the `runs/` directory produced by
Optuna_Optimization.py and opens it in your browser.

Usage:
    python launch_dashboard.py
    python launch_dashboard.py --logdir runs --port 6006

Requires:
    pip install tensorboardX tensorboard
"""

import argparse
import subprocess
import sys
import time
import webbrowser


def main():
    parser = argparse.ArgumentParser(description="Launch the gait-optimizer TensorBoard dashboard.")
    parser.add_argument("--logdir", default="runs", help="Directory containing the run logs (default: runs)")
    parser.add_argument("--port", type=int, default=6006, help="Port to serve TensorBoard on (default: 6006)")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser tab")
    args = parser.parse_args()

    cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", args.logdir, "--port", str(args.port)]

    print(f">>> Starting TensorBoard on http://localhost:{args.port}  (logdir: {args.logdir})")
    print(">>> Press Ctrl+C to stop.")

    try:
        proc = subprocess.Popen(cmd)
    except FileNotFoundError:
        print("!!! Could not find the 'tensorboard' package. Install it with:")
        print("    pip install tensorboardX tensorboard")
        sys.exit(1)

    if not args.no_browser:
        time.sleep(3.0)  # give the local server a moment to come up
        webbrowser.open(f"http://localhost:{args.port}")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n>>> Stopping TensorBoard...")
        proc.terminate()


if __name__ == "__main__":
    main()
