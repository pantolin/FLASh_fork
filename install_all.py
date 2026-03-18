#!/usr/bin/env python3
"""Install helper for the FLASh repository.

This is a thin wrapper around `install_all.sh` so users can install the project via:

    python install_all.py

It supports optional environment variables for data download.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install the FLASh library (editable) using the bundled installer script.")
    parser.add_argument("--data-url", help="Optional Google Drive URL (folder or file) to download example data.")
    parser.add_argument("--data-dir", help="Directory to place downloaded data.")
    parser.add_argument("--env-name", help="Conda environment name used by the installer.")
    args = parser.parse_args(argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    install_script = os.path.join(script_dir, "install_all.sh")

    if not os.path.exists(install_script):
        print(f"[ERROR] Could not find installer script: {install_script}")
        return 1

    env = os.environ.copy()
    if args.data_url:
        env["DATA_URL"] = args.data_url
    if args.data_dir:
        env["DATA_DIR"] = args.data_dir
    if args.env_name:
        env["QUGAR_ENV_NAME"] = args.env_name

    return subprocess.call(["bash", install_script], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
