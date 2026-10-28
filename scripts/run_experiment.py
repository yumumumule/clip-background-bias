"""Run from any working directory; config paths resolve from repository root."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml


def main():
    parser = argparse.ArgumentParser(description="Paired CLIP background intervention experiment")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--data-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-dir")
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--boundary", choices=["include", "exclude"])
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key] = value
    if config["samples"] < 1 or config["batch_size"] < 1 or config["bootstrap"] < 2 or config["seed"] < 0:
        parser.error("samples/batch-size must be positive, bootstrap >= 2, seed >= 0")
    # Store portable paths in metadata; switch once so all relative paths agree.
    import os
    os.chdir(ROOT)
    from src.experiment import run
    run(config)


if __name__ == "__main__":
    main()
