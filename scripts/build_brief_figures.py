"""Export local-only brief illustrations from the completed experiment; no inference.

Run with .venv/Scripts/python.exe scripts/build_brief_figures.py from the repo.
Requires the existing experiment dependencies plus reportlab, PyMuPDF and pypdf.
No donor reselection, mask cleanup, feathering, or model calls are performed.
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fitz
import numpy as np
from PIL import Image
from pypdf import PdfReader
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from src.dataset import sha256_file
from src.natural import composite, fixed_geometry, load_backgrounds, views
from src.preprocessing import bounding_box, foreground_mask, make_conditions

OUT = ROOT / "local_outputs/brief_assets"
BASE = ROOT / "data/oxford-iiit-pet"
DPI = 600
WIDTH = 170 * mm
COLORS = {1: (59, 130, 175), 2: (237, 237, 237), 3: (237, 186, 100)}
CAPTIONS = [
    "Figure 1. Object-centric preprocessing using ground-truth segmentation. "
    "The animal is localized using its trimap, and a padded bounding box is used "
    "to reduce surrounding context before CLIP inference.",
    "Figure 2. Controlled natural-background interventions. The foreground animal "
    "is preserved while its surrounding background is replaced with three "
    "photographic scenes. Identical crop coordinates are applied across all conditions.",
]


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def snapshot():
    """Hash protected results, brief, source, configuration and local model weights."""
    paths = set()
    for folder in ["results", "docs", "output", "src", "configs", ".cache/weights"]:
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                paths.add(path)
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(paths)}


def candidates(cohort):
    """Deterministic fallback after visual inspection of all 12 existing QC cases.

    Filter on annotation/image geometry only. Rank by boundary burden, then ID.
    No prediction file is opened here or used for selection.
    """
    eligible = []
    for item in cohort:
        w, h = item["width"], item["height"]
        if (min(w, h) < 350 or not .15 <= item["foreground_fraction"] <= .55
                or item["crop_area_fraction"] > .72):
            continue
        with Image.open(BASE / "annotations/trimaps" / (item["image_id"] + ".png")) as im:
            raw = np.asarray(im)
        ys, xs = np.where(raw != 2)
        margin = min(xs.min() / w, (w - 1 - xs.max()) / w,
                     ys.min() / h, (h - 1 - ys.max()) / h)
        if margin < .025:
            continue
        ratio = float((raw == 3).sum() / (raw != 2).sum())
        eligible.append((ratio, item["image_id"], item))
    return sorted(eligible, key=lambda entry: (entry[0], entry[1]))


def text(c, x, y, value, size=7, bold=False):
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    assert c.stringWidth(value, "Helvetica-Bold" if bold else "Helvetica", size) <= 40 * mm
    c.drawString(x, y, value)


def panel(c, image, x, y, width, height, box=None):
    """Embed all source pixels losslessly; retain aspect ratio without cropping."""
    scale = min(width / image.width, height / image.height)
    w, h = image.width * scale, image.height * scale
    px, py = x + (width - w) / 2, y + (height - h) / 2
    c.drawImage(ImageReader(image), px, py, w, h, mask=None)
    c.setStrokeColorRGB(.72, .72, .72)
    c.setLineWidth(.25)
    c.rect(px, py, w, h)
    if box is not None:
        left, top, right, bottom = box
        rx, ry = px + left * scale, py + (image.height - bottom) * scale
        rw, rh = (right - left) * scale, (bottom - top) * scale
        c.saveState()
        c.rect(px, py, w, h, stroke=0, fill=0)
        for color, line_width in [((1, 1, 1), 1.8), ((0, 0, 0), .8)]:
            c.setStrokeColorRGB(*color)
            c.setLineWidth(line_width)
            c.rect(rx, ry, rw, rh)
        c.restoreState()
    return {"source_size_px": list(image.size), "display_size_pt": [w, h],
            "effective_source_ppi": 72 / scale}


def new_canvas(name, height):
    c = canvas.Canvas(str(OUT / (name + ".pdf")), pagesize=(WIDTH, height),
                      pageCompression=1, invariant=1)
    c.setTitle("Object-centric cropping" if "figure1" in name else "Natural background swap")
    c.setAuthor("AIST1000 research project; dataset credits in accompanying README")
    c.setSubject("Local research illustration; third-party image terms apply")
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, 0, WIDTH, height, stroke=0, fill=1)
    return c


def render(name):
    with fitz.open(OUT / (name + ".pdf")) as doc:
        assert len(doc) == 1
        pix = doc[0].get_pixmap(dpi=DPI, alpha=False)
        pix.save(OUT / (name + ".png"))
        # A separate PDF render at reading scale for the combined visual preview.
        p = doc[0].get_pixmap(dpi=180, alpha=False)
        preview = Image.frombytes("RGB", (p.width, p.height), p.samples)
    reader = PdfReader(OUT / (name + ".pdf"))
    assert len(reader.pages) == 1
    return preview


def make_figures(image, raw, conditions, box, variants, donors):
    c = new_canvas("figure1_object_cropping", 50 * mm)
    pad, gap = 1.5 * mm, 2.0 * mm
    cw = (WIDTH - 2 * pad - 3 * gap) / 4
    colored = np.zeros((*raw.shape, 3), dtype=np.uint8)
    for label, rgb in COLORS.items():
        colored[raw == label] = rgb
    panels = [image, Image.fromarray(colored), image, conditions["crop_original"]]
    titles = [("A  Original image", ""), ("B  Ground-truth", "segmentation trimap"),
              ("C  Bounding box", "10% padding"), ("D  Object-centric crop", "")]
    layout1 = []
    for i, (im, (title, subtitle)) in enumerate(zip(panels, titles)):
        x = pad + i * (cw + gap)
        text(c, x, 46.0 * mm, title, bold=True)
        if subtitle:
            text(c, x, 42.8 * mm, subtitle, 6.7)
        layout1.append(panel(c, im, x, 7.4 * mm, cw, 33.2 * mm,
                             box if i == 2 else None))
    labels = [(1, "1  Animal"), (2, "2  Background"), (3, "3  Uncertain boundary")]
    legend_width = sum(4 * mm + c.stringWidth(label, "Helvetica", 6.7) + 5 * mm
                       for _, label in labels) - 5 * mm
    x = (WIDTH - legend_width) / 2
    for label, value in labels:
        c.setFillColorRGB(*(v / 255 for v in COLORS[label]))
        c.setStrokeColorRGB(.45, .45, .45)
        c.rect(x, 2.4 * mm, 2.5 * mm, 2.5 * mm, stroke=1, fill=1)
        text(c, x + 4 * mm, 2.7 * mm, value, 6.7)
        x += 4 * mm + c.stringWidth(value, "Helvetica", 6.7) + 5 * mm
    c.showPage()
    c.save()

    c = new_canvas("figure2_natural_background_swap", 89 * mm)
    layout2 = []
    for row in range(2):
        for col, variant in enumerate(variants):
            index = row * 4 + col
            x = pad + col * (cw + gap)
            title_y = (85.0 if row == 0 else 41.8) * mm
            image_y = (47.1 if row == 0 else 3.9) * mm
            if col == 0:
                title = "Original full image" if row == 0 else "Original object crop"
                subtitle = ""
            else:
                title = f"Natural swap {col}" if row == 0 else f"Object crop: swap {col}"
                subtitle = {"urban": "Urban street", "indoor": "Indoor living room",
                            "grassland": "Grassland / pasture", "forest": "Broadleaf forest"}[donors[col - 1]["category"]]
            text(c, x, title_y, f"{chr(65 + index)}  {title}", bold=True)
            if subtitle:
                text(c, x, title_y - 3.2 * mm, subtitle, 6.7)
            im = variant if row == 0 else variant.crop(box)
            layout2.append(panel(c, im, x, image_y, cw, 32.7 * mm))
    c.showPage()
    c.save()
    previews = [render("figure1_object_cropping"), render("figure2_natural_background_swap")]
    preview = Image.new("RGB", (max(p.width for p in previews), sum(p.height for p in previews) + 20), "white")
    y = 0
    for im in previews:
        preview.paste(im, (0, y))
        y += im.height + 20
    preview.save(OUT / "figures_preview.png", dpi=(180, 180))
    return {"figure1": layout1, "figure2": layout2}


def main():
    before = snapshot()
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = read_json("results/natural_backgrounds/sample_manifest.json")
    baseline = read_json("results/main/sample_manifest.json")
    assert len(cohort) == len(baseline) == 3662
    ranked = candidates(cohort)
    ratio, image_id, item = ranked[0]
    # First ranked candidate visually reviewed: complete animal, no severe mask failure.
    assert image_id == "havanese_55", "Selection changed; visually review before export"
    original_item = next(r for r in baseline if r["image_id"] == image_id)
    protocol = read_json("results/natural_backgrounds/protocol.json")
    cfg = read_json("results/natural_backgrounds/config.json")
    manifest, backgrounds = load_backgrounds(ROOT / "configs/natural_background_manifest.json",
                                             ROOT / "data/natural_backgrounds")
    saved_manifest = read_json("results/natural_backgrounds/background_manifest.json")
    assert saved_manifest == manifest
    assert sha256_file(ROOT / "configs/natural_background_manifest.json") == protocol["background_manifest_sha256"]
    for name in ["src/preprocessing.py", "src/natural.py"]:
        assert sha256_file(ROOT / name) == protocol["source_sha256"][name]
    assignment = next(a for a in protocol["assignments"] if a["image_id"] == image_id)
    assert assignment["donors"] == item["donor_ids"]
    assert assignment["label"] == item["label"]
    donor_lookup = {d["image_id"]: d for d in saved_manifest["images"]}
    donors = [donor_lookup[name] for name in item["donor_ids"]]
    image_path = BASE / "images" / (image_id + ".jpg")
    trimap_path = BASE / "annotations/trimaps" / (image_id + ".png")
    for path, key in [(image_path, "image_sha256"), (trimap_path, "trimap_sha256")]:
        assert sha256_file(path) == item[key] == original_item[key]
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    with Image.open(trimap_path) as im:
        trimap = im.copy()
        raw = np.asarray(im).copy()
    assert image.size == (item["width"], item["height"])
    assert set(np.unique(raw)) == {1, 2, 3}
    mask = foreground_mask(trimap, boundary="include")
    box = bounding_box(mask, padding=cfg["padding"])
    assert cfg["padding"] == .10
    assert list(box) == item["box"] == original_item["box"]
    assert fixed_geometry(image, trimap, cfg["padding"])[0] == box
    conditions, condition_box = make_conditions(image, trimap, padding=cfg["padding"])
    assert condition_box == box
    variants = [image] + [composite(image, trimap, backgrounds[d["image_id"]], policy="hard") for d in donors]
    original_pixels = np.asarray(image)
    variant_checks = []
    l, t, r, b = box
    crop_mask = mask[t:b, l:r]
    for variant in variants:
        a = np.asarray(variant)
        cropped = np.asarray(views(variant, box, tuple(item["center_box"]))["crop"])
        assert variant.size == image.size
        assert np.array_equal(a[mask], original_pixels[mask])
        assert np.array_equal(cropped, a[t:b, l:r])
        assert np.array_equal(cropped[crop_mask], original_pixels[t:b, l:r][crop_mask])
        variant_checks.append({"full_rgb_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                               "crop_rgb_sha256": hashlib.sha256(cropped.tobytes()).hexdigest(),
                               "retained_foreground_identical": True, "box": list(box)})
    # Read assignment columns only AFTER selection; no prediction outcomes used or shown.
    rows = []
    with (ROOT / "results/natural_backgrounds/predictions.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["image_id"] == image_id and row["boundary"] == "hard":
                rows.append({k: row[k] for k in ["swap", "scene", "donor_id", "method"]})
    assert len(rows) == 9
    for index, d in enumerate(donors):
        matched = [row for row in rows if int(row["swap"]) == index]
        assert len(matched) == 3 and {row["method"] for row in matched} == {"full", "crop", "center"}
        assert all(row["donor_id"] == d["image_id"] and row["scene"] == d["category"] for row in matched)
    layout = make_figures(image, raw, conditions, box, variants, donors)
    after = snapshot()
    assert before == after, "Protected inputs changed"
    provenance = {
        "project": "Can Object-Centric Processing Reduce Background Bias in Vision Models?",
        "repository": "https://github.com/yumumumule/clip-background-bias",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "created_date": "2026-11-08", "local_only": True,
        "sample": {**item, "image_path": str(image_path), "trimap_path": str(trimap_path),
                   "photographer_credit_visible_in_source": "Ralf Bitzer (visible signature; not independently verified)",
                   "cohort_size": 3662, "trimap_label_counts": {str(v): int((raw == v).sum()) for v in [1, 2, 3]}},
        "selection": {
            "outcome_blind": True, "existing_qc_samples_reviewed": 12,
            "qc_rejection_reasons": {
                "Abyssinian_2": "Thick dark retained halo; crop retains 89.7% of image, weak cropping illustration.",
                "basset_hound_70": "Body truncated at image edge; human/blanket pixels retained.",
                "Bombay_46": "Body truncated; dark animal has weak small-panel readability.",
                "Egyptian_Mau_214": "Truncated tail and costume-related semantic mask artifacts.",
                "german_shorthaired_60": "Wide retained grass boundary and occluded lower body.",
                "miniature_pinscher_79": "Tiny full-frame animal and conspicuous green boundary halo.",
                "leonberger_38": "Retained vegetation over animal; lower body obscured by grass.",
                "Sphynx_94": "Severe clothing-related mask failure removes torso.",
                "scottish_terrier_92": "Tiny full-frame animal; limited native crop resolution.",
                "basset_hound_197": "Tiny animal with snow occlusion and retained bright boundary.",
                "shiba_inu_61": "Tiny animal with retained leash/support context; limited crop resolution.",
                "leonberger_30": "Water occlusion; incomplete animal viewed from above."},
            "fallback_filters": {"minimum_image_short_side_px": 350, "foreground_fraction_inclusive": [.15, .55],
                                 "maximum_crop_area_fraction": .72, "minimum_foreground_margin_fraction_each_side": .025},
            "ranking": "Ascending label-3 / (label-1 + label-3) pixel count ratio, then image_id lexicographically",
            "eligible_count": len(ranked), "selected_rank_one_based": 1, "boundary_ratio": ratio,
            "first_ten_candidates": [{"image_id": e[1], "boundary_ratio": e[0]} for e in ranked[:10]],
            "visual_assessment": "Complete visible dog; recognizable face and body; modest retained floor-colored fringe documented, not retouched.",
            "prediction_outcomes_used": False},
        "geometry": {"box_xyxy": list(box), "coordinate_convention": "PIL left/top inclusive, right/bottom exclusive",
                     "padding": .10, "padding_rule": "Ceil 10% of unpadded mask extent on EACH side, clipped to source canvas",
                     "retained_labels": [1, 3], "fixed_across_all_variants": True},
        "donors_in_saved_order": [{**d, "path": str(ROOT / "data/natural_backgrounds" / d["image_id"]),
                                   "archive_url": manifest["archive_url"]} for d in donors],
        "compositing": {"policy": "hard", "function": "src.natural.composite",
                        "donor_transform": "PIL ImageOps.fit BICUBIC, centered (0.5, 0.5), to original canvas",
                        "retained_foreground_rgb_size_position": "Exactly unchanged before figure display resampling",
                        "boundary_cleanup_or_feathering": False},
        "source_provenance": {"oxford_dataset": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
                              "image_archive": "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz",
                              "annotations_archive": "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz",
                              "places_dataset": manifest["dataset"], "places_archive_md5": manifest["archive_md5"],
                              "asset_restoration_needed": False},
        "permissions": {
            "checked_date": "2026-11-08",
            "oxford": "Current official dataset page specifies CC BY-SA 4.0; original owners retain copyright. Provide dataset/owner credit where known, license link, and change notice; ShareAlike applies to adaptations when shared.",
            "oxford_license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "places": manifest["usage_restrictions"],
            "places_current_copyright_source": "https://github.com/CSAILVision/places365#license",
            "places_archived_terms_url": manifest["terms_url"],
            "verification_limit": "Current Oxford license and Places owner-copyright statement verified online. Places download page timed out and archived terms could not be fetched; research/education-only and no-redistribution restrictions are retained from the locked experiment manifest, not independently revalidated in this run.",
            "classroom_or_transfer": "Figure 2 and ZIP contain Places-derived imagery. No explicit redistribution/classroom-submission exception was verified. Do not treat the local ZIP as permission to send it, upload to an LMS, or publish it; establish permission or an applicable exception first. Figure 1 may be shared subject to the Oxford license conditions.",
            "public_commit": False, "zip_status": "Local packaging only; not cleared for redistribution"},
        "captions": CAPTIONS,
        "export": {"width_mm": 170, "png_dpi": DPI, "pdf_embedded_images": "Lossless source pixels, vector labels and box",
                   "trimap_display": "Categorical RGB lookup only; no interpolation between labels in embedded source",
                   "figure1_height_mm": 50, "figure2_height_mm": 89, "layout": layout},
        "verification": {"source_and_trimap_hashes_match_both_saved_manifests": True,
                         "donor_hashes_match_locked_pool": True, "assignments_match_protocol_and_saved_prediction_metadata": True,
                         "variant_pixel_checks": variant_checks, "protected_files_unchanged": True,
                         "protected_file_sha256_before_and_after": before, "inference_run": False,
                         "predictions_displayed": False},
        "outputs": {}}
    for name in ["figure1_object_cropping", "figure2_natural_background_swap"]:
        with Image.open(OUT / (name + ".png")) as im:
            im.load()
            assert min(im.info["dpi"]) >= 599
            provenance["outputs"][name + ".png"] = {"size_px": list(im.size), "dpi": list(im.info["dpi"]),
                                                        "sha256": sha256_file(OUT / (name + ".png"))}
        provenance["outputs"][name + ".pdf"] = {"sha256": sha256_file(OUT / (name + ".pdf"))}
    (OUT / "figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    readme = f"""# Local research brief figures

Both figures use **{image_id}** ({item['breed']}; zero-based class {item['label']}) from the saved 3,662-image evaluation cohort. Source: {image.width} x {image.height} pixels. Fixed PIL-exclusive box: **{list(box)}**; crop: {r-l} x {b-t} pixels. The foreground animal and uncertain boundary retain their original RGB values in every hard swap.

## Files and use

- `figure1_object_cropping.png` and `.pdf`: four panels, 170 x 50 mm.
- `figure2_natural_background_swap.png` and `.pdf`: 2 x 4 panels, 170 x 89 mm; saved donor order.
- `figure_provenance.json`: full paths, source/donor hashes, geometry, selection, output hashes and checks.
- `figures_preview.png`: combined PDF-rendered visual preview at 180 DPI.
- `brief_figures.zip`: exactly the two 600-DPI PNGs, provenance JSON and this README. LOCAL PACKAGING ONLY; not cleared for redistribution.

Place at 16-18 cm width. PDFs contain vector labels/box and losslessly embedded native image arrays. PNGs are 600 DPI. Source photographs are not enhanced, reconstructed, stretched, or subjected to an additional content crop. Lower donor resolution is intrinsic to the completed experiment (256 x 256), not an export defect. Panel image rectangles retain aspect ratio within consistent layout cells. No captions or model predictions are drawn inside the figures.

## Exact experimental inputs

Original: `{image_path}`

Trimap: `{trimap_path}`

Donors in actual saved order:
"""
    for i, d in enumerate(donors, 1):
        readme += f"\n{i}. `{d['image_id']}` - {d['category']} (`{d['places_category']}`)."
    readme += """

Assignments are loaded from `results/natural_backgrounds/sample_manifest.json` and checked against locked `protocol.json` and assignment fields in saved predictions. No new donors are selected. Existing `foreground_mask`, `bounding_box`, `make_conditions`, `fixed_geometry`, `composite` and `views` functions are reused. Padding is 10% of the unpadded extent on each side, rounded up and clipped. Label 1 is animal, label 2 background, label 3 uncertain boundary; labels 1 and 3 are retained. Figure 1 trimap colors map raw labels directly. Figure 1 bounding-box lines are overlays only. Figure 2 uses hard compositing, donor-only centered bicubic cover-resizing, and the identical original box for all crops. Photographic lighting and support surfaces can be implausible. The modest retained fringe is part of the actual hard-boundary experiment and has not been retouched.

## Outcome-blind selection

All 12 existing QC samples in `configs/natural_background_qc.json` and all three local QC grids were visually reviewed. None combined a complete, readable animal, limited boundary artifacts, sufficient native crop resolution and a useful reduction in context. Specific rejection reasons are in the JSON. Fallback filters, fixed before inspecting fallback composites: source short side >=350 px; foreground fraction 0.15-0.55 inclusive; padded crop <=72% of image; unpadded retained foreground at least 2.5% from every source edge. Rank the 259 eligible saved-cohort entries by label-3 / (label-1 + label-3) pixel ratio, then image ID. Select the first entry, havanese_55 (ratio 0.2207832295), subject to visual validation. No candidate was excluded on the basis of model performance. Prediction values are neither reported nor used; only assignment metadata is verified after selection.

## Attribution and permissions - checked 2026-11-08

Oxford-IIIT Pet: Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman and C. V. Jawahar, *Cats and Dogs*, CVPR 2012. Dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/ . The current official page states **CC BY-SA 4.0**: https://creativecommons.org/licenses/by-sa/4.0/ . Copyright remains with original image owners. The original Havanese photograph visibly bears a Ralf Bitzer signature (identity not independently verified). Preserve this credit where practical. Figure 1 changes: categorical trimap visualization, bounding-box overlay and crop/panel assembly. Shared Oxford-derived adaptations must satisfy attribution, license-link, change-notice and ShareAlike requirements. Figure 1 is provided under the dataset's stated CC BY-SA 4.0 terms to the extent those rights are available. This notice does not relicense Places content.

Places365: Bolei Zhou et al., *Places: A 10 Million Image Database for Scene Recognition*, IEEE TPAMI 2018. Source archive: https://data.csail.mit.edu/places/places365/val_256.tar . Scene IDs and hashes are in the JSON; individual photographer identities are not supplied by the locked manifest. Current project documentation confirms image copyrights belong to image owners: https://github.com/CSAILVision/places365#license . Code/model licenses do not grant an image redistribution license.

The locked experiment manifest records non-commercial research/education-only use and **no image redistribution**, citing https://web.archive.org/web/20210220220343/http://places2.csail.mit.edu/download.html . That archived page could not be fetched during this run; the current download page also timed out. These restrictions are retained conservatively from the saved protocol, with that verification limit disclosed rather than presenting them as newly retrieved terms.

**Classroom submission / transfer flag:** Educational use does not itself establish an exception to the recorded redistribution restriction. Figure 2 and the ZIP contain Places-derived image content. No express permission for sending these composites, uploading them to an LMS, or including them in a distributed submission was verified. Keep them local until permission or an applicable exception is established. A local ZIP is only packaging, not redistribution clearance. Do not publicly commit photographs, composites, PNGs, PDFs or the ZIP. All exports are under ignored `local_outputs/`. The image-free generation script may be committed separately. Nothing was committed or uploaded by this task.

## Verification and reproduction

Run from the repository root:

```powershell
.venv/Scripts/python.exe scripts/build_brief_figures.py
```

The script aborts on source/donor hash, saved-box or assignment mismatch. Pixel assertions check labels 1 AND 3 exactly across all full frames and fixed crops. Protected result, source, configuration, research-brief and local model-weight files are hashed before and after and must match. It does not run inference, train a model, modify original results, or edit the current brief. All delivered PNGs and PDFs are reopened; PDFs are rendered for visual QA. Review `figures_preview.png` at intended page width. Source resolution and output dimensions are recorded in the JSON.

## Captions (for insertion separately)

"""
    readme += "\n\n".join(CAPTIONS) + "\n"
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    with zipfile.ZipFile(OUT / "brief_figures.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in ["figure1_object_cropping.png", "figure2_natural_background_swap.png", "figure_provenance.json", "README.md"]:
            z.write(OUT / name, arcname=name)
    with zipfile.ZipFile(OUT / "brief_figures.zip") as z:
        assert z.testzip() is None
        assert len(z.namelist()) == 4
    assert read_json("local_outputs/brief_assets/figure_provenance.json")["sample"]["image_id"] == image_id
    assert (OUT / "README.md").read_text(encoding="utf-8")
    assert snapshot() == before
    print(json.dumps({"selected": image_id, "box": box, "donors": donors,
                      "protected_files_verified_unchanged": len(before), "outputs": provenance["outputs"],
                      "output_directory": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
