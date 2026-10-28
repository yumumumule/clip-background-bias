import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.visualization import regenerate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate figures from measured CSV files")
    parser.add_argument("--results", type=Path, default=ROOT / "results/main")
    regenerate(parser.parse_args().results)
