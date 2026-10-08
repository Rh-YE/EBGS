"""Install notebook/runtime dependencies into the active Python environment.

Run with a Python 3.11-3.13 virtual environment. Initial installation uses the
internet; the notebook itself uses only files in this folder.
"""
from pathlib import Path
import argparse
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-torch", action="store_true",
                        help="Keep an existing matching torch/torchvision build (e.g. CUDA).")
    args = parser.parse_args()
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise SystemExit("Please use Python 3.11-3.13 for the supplied dependency versions.")
    pip = [sys.executable, "-m", "pip"]
    if not args.keep_torch:
        command = pip + ["install", "torch==2.8.0", "torchvision==0.23.0"]
        if platform.system() in {"Linux", "Windows"}:
            command += ["--index-url", "https://download.pytorch.org/whl/cpu"]
        subprocess.check_call(command)
    subprocess.check_call(pip + ["install", "-r", str(Path(__file__).with_name("requirements.txt"))])
    print("Ready. Start with: python -m jupyterlab single_galaxy_demo.ipynb")


if __name__ == "__main__":
    main()
