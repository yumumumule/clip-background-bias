import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.natural_analysis import analyze

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompute source-image cluster analysis and figures")
    parser.add_argument("--results", type=Path, default=ROOT / "results/natural_backgrounds")
    import os
    os.chdir(ROOT)
    analyze(parser.parse_args().results)
