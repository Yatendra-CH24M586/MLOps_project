# src/retrain.py
"""
Small convenience wrapper to run dvc repro or run training directly.
Usage:
  python src/retrain.py        # runs `dvc repro`
  python src/retrain.py --no-dvc --train-cmd "python src/train.py --train ... --val ... --params params.yaml"
"""

import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--no-dvc", action="store_true", help="Do not run dvc repro, run explicit train command")
parser.add_argument("--train-cmd", type=str, help="Explicit training command to run if --no-dvc used")
args = parser.parse_args()

if args.no_dvc:
    if not args.train_cmd:
        print("Provide --train-cmd when using --no-dvc.")
        sys.exit(1)
    print("Running training command:", args.train_cmd)
    subprocess.run(args.train_cmd, shell=True, check=True)
else:
    print("R Drift detected → Triggering retraining pipeline...")
    subprocess.run(["dvc", "repro"], check=True)
    print(" Retraining finished. New model logged to MLflow.")
