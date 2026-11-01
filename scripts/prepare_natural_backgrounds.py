"""Download verified Places365 archives; extract only the fixed background pool.

    --candidates makes a pre-inference seed-42 QC proposal (twelve per category).
    Default mode reconstructs the committed, visually approved pool exactly.
    Photographs always stay under ignored data/local_outputs directories.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from src.dataset import sha256_file

URL = "https://data.csail.mit.edu/places/places365/"
ARCHIVES = {"val_256.tar": "e27b17d8d44f4af9a78502beb927f808",
            "filelist_places365-standard.tar": "35a0585fee1fa656440f3ab298f8479c"}
CATEGORIES = {"indoor": "/l/living_room", "urban": "/s/street",
              "forest": "/f/forest/broadleaf", "grassland": "/p/pasture"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", action="store_true")
    args = parser.parse_args()
    archive_dir = ROOT / "data/places365"
    output = ROOT / "data/natural_backgrounds"
    archive_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    for name, expected in ARCHIVES.items():
        path = archive_dir / name
        if not path.exists():
            temporary = path.with_suffix(".part")
            urllib.request.urlretrieve(URL + name, temporary)
            temporary.replace(path)
        with open(path, "rb") as stream:
            checksum = hashlib.file_digest(stream, "md5").hexdigest()
        if checksum != expected:
            raise ValueError(f"Corrupt {name}: expected {expected}, got {checksum}; remove and download again")
    if args.candidates:
        with tarfile.open(archive_dir / "filelist_places365-standard.tar") as archive:
            def read(name):
                member = next(m for m in archive.getmembers() if m.name.endswith(name))
                return archive.extractfile(member).read().decode()
            classes = {name: int(label) for name, label in (x.split() for x in read("categories_places365.txt").splitlines())}
            files = [(name.lstrip("/"), int(label)) for name, label in
                     (x.split() for x in read("places365_val.txt").splitlines())]
        rng = np.random.default_rng(42)
        selected = []
        for category, official in sorted(CATEGORIES.items()):
            candidates = sorted(name for name, label in files if label == classes[official])
            for rank, name in enumerate(rng.permutation(candidates)[:12]):
                selected.append({"image_id": str(name), "category": category,
                                 "places_category": official, "candidate_rank": rank})
    else:
        manifest = json.loads((ROOT / "configs/natural_background_manifest.json").read_text())
        selected = manifest["images"]
    by_name = {d["image_id"]: d for d in selected}
    with tarfile.open(archive_dir / "val_256.tar") as archive:
        for member in archive:
            name = Path(member.name).name
            if name in by_name:
                data = archive.extractfile(member).read()
                with Image.open(io.BytesIO(data)) as image:
                    image.verify()
                path = output / name
                path.write_bytes(data)
                digest = sha256_file(path)
                if not args.candidates and digest != by_name[name]["sha256"]:
                    raise ValueError(f"Donor hash mismatch: {name}")
                by_name[name]["sha256"] = digest
    if any(not (output / d["image_id"]).exists() for d in selected):
        raise FileNotFoundError("Not all selected donors were found in the verified archive")
    if args.candidates:
        local = ROOT / "local_outputs/natural_backgrounds"
        local.mkdir(parents=True, exist_ok=True)
        (local / "candidate_manifest.json").write_text(json.dumps(selected, indent=2))
        fig, axes = plt.subplots(4, 12, figsize=(28, 10))
        for ax, donor in zip(axes.flat, selected):
            with Image.open(output / donor["image_id"]) as image:
                ax.imshow(image)
            ax.set_title(f"{donor['category']} #{donor['candidate_rank']}\n{donor['image_id']}", fontsize=6)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(local / "donor_candidates.png", dpi=160)
        plt.close(fig)
    print(f"Prepared {len(selected)} verified photographic donors under ignored data/natural_backgrounds")


if __name__ == "__main__":
    main()
