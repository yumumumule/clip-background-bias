# Main experimental results: 1 November 2026

The complete test split was checked with fixed model, prompt, padding, mask boundary policy, blur radius and replacement colors. No setting was tuned to improve the result. This full-test evaluation is the repository's primary experiment; the earlier 370-image version is available in Git history.

## Coverage and annotation quality

- Test split checked: **3,669 images**.
- Invalid trimaps: **7**, containing only label 2 (background), so an animal crop cannot be defined.
- Paired evaluation: **3,662 images across all 37 breeds**, with exactly the same images in all 11 conditions: **40,282 predictions**.
- Excluded IDs: `Egyptian_Mau_20`, `japanese_chin_199`, `keeshond_7`, `Persian_259`, `saint_bernard_60`, `saint_bernard_78`, `wheaten_terrier_195`.
- All seven are excluded from every paired condition, including the full-image baseline. Exclusion depends only on invalid annotations, not model predictions. The default runner audits selected trimaps and records exclusions; `--no-exclude-invalid-masks` restores strict failure.
- Exact reasons, pixel counts and trimap hashes: [excluded_samples.json](../results/main/excluded_samples.json).

The complete valid cohort is not exactly class-balanced: there are 88–100 images per breed. Primary accuracies are image-weighted. Class-stratified bootstrap resampling retains these observed class counts. Excluding invalid annotations limits conclusions to the evaluable cohort.

## Accuracy and stability

| Processing | Original accuracy | Blur accuracy | Uniform accuracy | Blur flip rate | Uniform flip rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full image | 84.46% | 82.41% | 80.97% | 9.75% | 13.60% |
| Object crop | 84.35% | 83.01% | 81.70% | 8.30% | 11.63% |
| Centered control | 84.05% | 82.00% | 80.97% | 9.42% | 12.45% |

Foreground on neutral gray: **80.34%** accuracy. Background-only diagnostic: **7.54%** accuracy; the silhouette and residual boundary artifacts prevent interpreting it as pure natural-background information.

All intervals below are 95% percentile bootstrap intervals using 5,000 paired resamples within breeds, seed 42. Differences are **percentage points (pp)**. Negative flip-rate differences favor the object crop. Comparisons are exploratory and not adjusted for multiplicity.

| Crop minus full image | Difference (pp) | 95% CI (pp) |
| --- | ---: | ---: |
| Original accuracy | −0.11 | [−0.82, +0.60] |
| Blur accuracy | +0.60 | [−0.19, +1.42] |
| Uniform accuracy | +0.74 | [−0.08, +1.56] |
| Blur flip rate | **−1.45** | **[−2.38, −0.49]** |
| Uniform flip rate | **−1.97** | **[−2.95, −1.01]** |

Both primary flip-rate difference intervals now exclude zero. The effect is more precisely supported than in the 370-image subset, but its absolute magnitude remains modest. Neither original nor background-modified accuracy differences clearly exclude zero.

Accuracy drop under background blur is 2.05 pp for full images versus 1.34 pp for object crops. Uniform-background accuracy drop is 3.50 pp versus 2.65 pp. The paired differences between the accuracy changes are +0.71 pp [−0.19, +1.61] and +0.85 pp [−0.11, +1.80], respectively; these intervals include zero.

The centered control also reduces the observed uniform-background flip rate. Object crop minus centered-control flip rate is −1.12 pp [−1.99, −0.22] for blur and −0.82 pp [−1.72, +0.11] for uniform replacement. The latter remains uncertain. This partial geometry control does not justify attributing all observed robustness improvement specifically to accurate object localization.

## Interpretation

**Object cropping modestly reduces prediction flips under the two tested background interventions, while providing no clear classification accuracy gain.** This is evidence of a limited stability benefit under oracle segmentation, not a large performance improvement or a general solution to background bias.

The larger cohort improves precision; it does not make the effect large. The 370-image subset is contained in this cohort, so this is an expanded evaluation rather than an independent replication. All 4,070 predictions on that subset match the full-run predictions exactly. Model checkpoint hash and text-cache key also match the earlier run.

Oracle masks, boundary halos, changes in object scale, unknown pretraining overlap and limited model/prompt coverage remain limitations. The separate natural-background extension below tests photographic donors; explicit scale controls, predicted masks and other frozen models remain untested.

## Validation and reproduction

- **30 unit tests passed**, including two new invalid-trimap audit tests.
- Dependency consistency check passed.
- Result audit passed: 3,662 unique IDs, 40,282 predictions, matching labels, no excluded IDs in any condition, CSV/JSON agreement, deterministic bootstrap regeneration.
- Generated accuracy, robustness and consistency plots were opened and inspected.
- The first full-split run stopped on an empty trimap. All trimaps were then audited and a common-cohort exclusion policy was added. The completed rerun encountered no further inference errors; the release now uses the audit by default.

Run from the repository root, with the environment activated:

```text
python -m pytest -q
python scripts/run_experiment.py --download --output-dir local_outputs/full-test-reproduction
python scripts/verify_results.py --results local_outputs/full-test-reproduction
```

Choose an empty output directory. The measured run originally used `--output-dir results/full-test-2026-11-01`; its outputs were promoted to `results/main` during consolidation. Metadata preserves that historical command path and source-file hashes. Current default settings reproduce the same cohort and model conditions.

Measured files: [accuracy.csv](../results/main/accuracy.csv), [paired_metrics.csv](../results/main/paired_metrics.csv), [additional_paired_comparisons.csv](../results/main/additional_paired_comparisons.csv), [per_breed_accuracy.csv](../results/main/per_breed_accuracy.csv), [predictions.csv](../results/main/predictions.csv), [results.json](../results/main/results.json), [metadata.json](../results/main/metadata.json), and [sample_manifest.json](../results/main/sample_manifest.json).

Reproduce supplementary comparisons and the effect-size figure with `python scripts/analyze_results.py --results results/main`; the [analysis script](../scripts/analyze_results.py) reads saved predictions without model inference.

## Additional natural-background robustness test

The original experiment above remains the primary baseline and its saved files are unchanged. The extension uses the same 3,662 animals, three distinct scene categories per animal from a fixed pool of 12 Places365 photographs, and both hard and feathered boundaries. All 76,902 inferences completed; rerun original predictions match the baseline exactly. The primary sample size is 3,662 **source images**, not 10,986 independent swaps. Each animal's swaps are averaged before the 5,000-replicate breed-stratified paired cluster bootstrap.

| Hard-boundary method | Natural-swap accuracy | Flip rate | Accuracy drop from its own original |
| --- | ---: | ---: | ---: |
| Full image | 79.34% | 16.16% | 5.12 pp |
| Object crop | 81.31% | 13.13% | 3.04 pp |
| Center control | 80.20% | 14.60% | 3.85 pp |

| Paired natural comparison | Difference (pp) | 95% cluster CI (pp) |
| --- | ---: | ---: |
| Crop minus Full, flip rate (primary) | −3.02 | [−3.78, −2.29] |
| Crop minus Full, swap accuracy | +1.98 | [+1.36, +2.62] |
| Crop minus Center, flip rate | −1.47 | [−2.21, −0.77] |
| Crop minus Center, swap accuracy | +1.11 | [+0.50, +1.73] |
| Center minus Full, flip rate | −1.56 | [−2.26, −0.89] |
| Feather Crop minus Full, flip rate | −3.05 | [−3.80, −2.34] |

The extension supports persistence of the stability benefit in this fixed photographic pool. Its observed effect is larger than the artificial-intervention point estimates, but that does not isolate an effect of naturalness: intervention strength and scene content differ. Accuracy improves compared with swapped full images, while remaining below original-background crop accuracy. This does not show elimination of learned background bias.

Center truncation occurs for 1,415 images. Among 2,247 without confident-foreground truncation, Crop-minus-Center flips are −0.55 pp [−1.31, +0.21]. Geometry remains a plausible contributor. The indoor Crop-minus-Center interval also includes zero. Feathering changes the primary effect by −0.03 pp [−0.38, +0.33]; numerical similarity is not proof of equivalence.

The [extension report](natural_background_experiment.md) contains all four scene contrasts, transitions, provenance, local-only visual QC, locked design and exact commands. Numerical outputs are in [results/natural_backgrounds](../results/natural_backgrounds). The [three-page PDF](../output/pdf/research_brief.pdf) combines both experiments.
