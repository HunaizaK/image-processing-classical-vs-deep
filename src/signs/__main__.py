"""CLI: python -m signs {all,classical,cnn,robustness,report} [--full]"""

from __future__ import annotations

import argparse
import logging
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from signs.config import load_config  # noqa: E402
from signs.pipeline import STEPS, run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(prog="signs")
    parser.add_argument("step", choices=["all", *STEPS])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--full", action="store_true", help="longer CNN training")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, full=args.full)
    run(cfg, list(STEPS) if args.step == "all" else [args.step])


if __name__ == "__main__":
    main()
