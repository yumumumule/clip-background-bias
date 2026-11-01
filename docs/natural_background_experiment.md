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

## Executed results

The pool and protocol were committed as `76128f7` before the first smoke inference. Both the five-source smoke and full **3,662-source experiment** completed: **76,902 predictions**, consisting of 10,986 original views and 65,916 swap views. There are 10,986 animal/background assignments, two boundary policies and three methods. All original predictions match the preserved baseline. Every composite passed confident-foreground RGB and label-2 donor assertions; hard composites also preserved label 3 exactly.

| Hard-boundary method | Original accuracy | Swap accuracy | Flip rate | Accuracy drop |
| --- | ---: | ---: | ---: | ---: |
| Full image | 84.46% | 79.34% | 16.16% | 5.12 pp |
| Object crop | 84.35% | 81.31% | 13.13% | 3.04 pp |
| Center control | 84.05% | 80.20% | 14.60% | 3.85 pp |

| Hard-boundary paired comparison | Estimate (pp) | 95% cluster CI (pp) |
| --- | ---: | ---: |
| Crop minus Full flip rate, primary | **−3.02** | **[−3.78, −2.29]** |
| Crop minus Full swap accuracy | +1.98 | [+1.36, +2.62] |
| Crop minus Full accuracy drop | −2.08 | [−2.92, −1.27] |
| Crop minus Center flip rate | −1.47 | [−2.21, −0.77] |
| Crop minus Center swap accuracy | +1.11 | [+0.50, +1.73] |
| Center minus Full flip rate | −1.56 | [−2.26, −0.89] |

The primary contrast is about **18.7% relative reduction** in flips. This supports persistence of the original stability benefit within the fixed photographic pool. Natural-swap accuracy improves over swapped Full images, unlike the uncertain accuracy contrasts in the artificial experiment. Original-background accuracy remains essentially unchanged, and swapped Crop accuracy remains below its own original accuracy. Stability and correctness remain distinct outcomes.

### All four scene categories

Each category appears at most once per animal. Its n is the number of source animals assigned that scene. Forest and grassland are natural environments; urban streets are also outdoors. We report mutually exclusive selection categories rather than treating overlapping outdoor/natural tags as independent groups.

| Scene | Source animals | Crop minus Full flips (pp; 95% CI) | Crop minus Center flips (pp; 95% CI) |
| --- | ---: | ---: | ---: |
| Forest | 2,735 | −2.67 [−3.88, −1.50] | −1.61 [−2.71, −0.55] |
| Grassland | 2,768 | −3.72 [−4.95, −2.49] | −1.91 [−3.00, −0.83] |
| Indoor | 2,746 | −2.33 [−3.57, −1.06] | −0.51 [−1.64, +0.62] |
| Urban | 2,737 | −3.36 [−4.53, −2.23] | −1.83 [−2.89, −0.77] |

All Crop-minus-Full intervals exclude zero; no category was selected after seeing performance. Crop-minus-Center remains uncertain indoors. Different point estimates alone do not establish a category interaction; no confirmatory between-scene interaction test was specified. Full per-scene accuracy, flip and transition tables are saved.

### Boundary and geometry sensitivity

Feathered flip rates are 15.71% Full, 12.66% Crop and 14.14% Center. Crop-minus-Full is **−3.05 pp [−3.80, −2.34]**. The change in the paired effect from hard to feather is −0.03 pp [−0.38, +0.33]. The observed direction persists; an interval containing zero does not demonstrate policy equivalence.

The center box truncates confident foreground in **1,415 images**. Among the other **2,247**, Crop-minus-Center flips are −0.55 pp [−1.31, +0.21] for hard boundaries and −0.65 pp [−1.41, +0.09] for feathering. The extra crop advantage is unresolved in this subgroup. The flag concerns the explicit center box; CLIP's later standard center crop can remove further content. This observational subgroup is not a randomized scale-matching control.

### Correctness transitions

Rates below use all sources and average their three hard swaps, rather than conditioning on originally correct/incorrect predictions.

| Method | Correct → incorrect | Incorrect → correct | Stable but incorrect |
| --- | ---: | ---: | ---: |
| Full | 9.08% | 3.96% | 8.47% |
| Crop | 6.75% | 3.71% | 9.27% |
| Center | 7.46% | 3.61% | 8.81% |

Wrong answers can become correct, remain stable, or change between wrong breeds. Accuracy drop equals correct-to-incorrect minus incorrect-to-correct rate; the audit verifies this identity and checks that stability is not conflated with correctness.

## Interpretation and practical limits

The observed natural-swap effect is larger than the artificial-intervention point estimates, but this does not isolate naturalness: scene content and intervention strength differ. Settings were not changed to obtain favorable results. Crop versus Center supports an overall advantage, while the non-truncation subgroup and indoor results leave framing/context explanations open.

Only twelve screened donors were used, at 256-pixel resolution and resized to each canvas. The bootstrap does not sample unseen donors. Lighting, perspective and support surfaces can be implausible. Pre-inference QC documented retained grass, halos and semantic mask failures, including removal of clothing. A numerically valid trimap need not perfectly segment an animal. RGB invariance applies before CLIP interpolation. This does not establish causal removal of learned bias, universal natural-scene robustness, or performance with predicted masks.

Next, test larger independent donor pools, more plausible composites, explicit scale/framing controls, predicted masks and other frozen models. The original artificial experiment remains the primary baseline.

## Saved outputs and validation

New outputs are under [results/natural_backgrounds](../results/natural_backgrounds); `results/main` is unchanged. Files include locked protocol/config, source/donor manifests, original/swap predictions, all 37-class cosine scores, source means, accuracy/flip/transition tables, paired contrasts, boundary/geometry sensitivity, JSON results, metadata and three measured figures. Cosines are not calibrated probabilities. No photographs are published.

```text
python scripts/verify_natural_backgrounds.py --results results/natural_backgrounds
python -m pip install -r requirements-docs.txt
python scripts/build_research_brief.py
```

**42 tests passed**; both smoke and full-result audits passed. Verification checks all 76,902 predictions/scores, method/donor/boundary pairing, baseline originals, manifests and deterministic cluster-bootstrap regeneration. The [PDF](../output/pdf/research_brief.pdf) has two A4 text pages plus a separate reference page; all rendered pages were inspected. Smoke CIs degenerate with one source per represented breed and are not research evidence.
