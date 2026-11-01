# Natural photographic background protocol

This extension preserves `results/main` and uses the identical saved 3,662-image cohort. No model is trained. This protocol, background pool and settings are locked before the first CLIP smoke or full run.

## Prespecified design

- Frozen OpenAI CLIP ViT-B/32, original 37 candidates and prompt, float32 inference and normal CLIP preprocessing.
- Twelve Places365-Standard validation photographs: three each from indoor living rooms, urban streets, forests and grassland/pasture. The exact pool and visual exclusions are in `configs/natural_background_manifest.json`.
- Seed-42 permutations of sorted validation identifiers propose candidates. Visually accept the first three per category without prominent animals/people or dominant foreground subjects; all decisions precede any model predictions.
- Each animal receives three distinct categories using a stable SHA256 hash of seed and image ID, then an independent within-category choice. Labels, breeds, mask properties and predictions are not consulted. Categories and assignments will not be changed after observing results.
- Donor-only bicubic cover-resize and centered crop fill the original canvas. Foreground location, scale and confidently labeled RGB pixels are invariant. Invariance is asserted for every composite.
- Primary hard compositing keeps trimap labels 1 and 3 unchanged. Sensitivity analysis smooths retained-mask alpha at sigma 1.5 source pixels, applies blending only at uncertain label 3, and forces label 1 to original RGB and label 2 to the donor.
- Compute the original 10%-padded animal box and dimension-matched center box once. Apply them after full-frame compositing for every donor and boundary policy. Record center truncation and retained confident-foreground fraction.

**Primary endpoint:** hard-boundary mean flip-rate difference, Object Crop minus Full Image, averaging the three swaps within each source animal before calculating paired differences.

Secondary endpoints: original/swap accuracy, accuracy drop, correct-to-incorrect and incorrect-to-correct transitions, stable incorrect predictions; Crop vs Center and Center vs Full; all four scene-specific analyses; boundary sensitivity. A prespecified supplementary subgroup tests Crop vs Center among images whose center control retains all confidently labeled foreground pixels. This is still not a perfect object-scale control.

## Statistics

Reuse the original 5,000-replicate seed-42 breed-stratified bootstrap on per-source-image means. Resampling one source retains all its swaps and method comparisons together. The effective primary sample size is 3,662 source images, not 10,986 independent swaps. Scene analyses use source means among the images assigned that category. Intervals condition on this model, fixed donor pool and protocol; donor-pool uncertainty is not estimated. Secondary analyses are exploratory and unadjusted for multiplicity. An interval containing zero is not equivalence.

## Provenance and permissions

Dataset: [Places365 development kit](https://github.com/zhoubolei/places_devkit); validation archive at the official MIT data server. Archive checksums match torchvision's documented metadata. Exact identifiers and SHA256 hashes are committed, photographs are not.

The [archived official download terms](https://web.archive.org/web/20210220220343/http://places2.csail.mit.edu/download.html) restrict use to non-commercial research/education and prohibit redistribution. [Current project documentation](https://github.com/CSAILVision/places365) states image copyrights remain with their owners. The current download page requests contact through a legacy-data form; the legacy archive remains publicly accessible. The model/code licenses do not license the images. All donors and composites stay locally under ignored `data/` and `local_outputs/`.

## Reproduction and quality control

```text
python scripts/prepare_natural_backgrounds.py
python scripts/qc_natural_backgrounds.py
python -m pytest -q
python scripts/run_natural_backgrounds.py --samples 5 --bootstrap 200 --output-dir local_outputs/natural-smoke
python scripts/run_natural_backgrounds.py
python scripts/analyze_natural_backgrounds.py --results results/natural_backgrounds
```

The download is about 501 MiB plus the devkit. Only selected photographs are extracted. Reconstruct the committed pool without reselection; `--candidates` is a development/QC tool, not part of the default reproduction. Missing/corrupt assets abort rather than silently substituting another donor or changing the cohort. Nonempty output directories and `results/main` are protected.

Local QC grids cover different breeds, high boundary-to-foreground ratios and small animal areas, with all assigned scenes, hard/feather examples and object crops. Inspect them before inference. QC does not guarantee physical plausibility, matching lighting, perfect masks or absence of every small animal. No donor or sample will be removed based on CLIP results.

Status at protocol definition: implementation and validation underway; no natural-background performance numbers have been recorded yet.
