# Can Object-Centric Processing Reduce Background Bias in Vision Models?

AIST1000 research brief · Full-test evaluation, 1 November 2026

## Problem

A vision classifier may respond to an animal's surroundings even when those surroundings do not determine its breed. We ask: **Does focusing on the foreground object make CLIP's predictions more stable when the background changes?** Correctness and stability are different: an unchanged answer can be wrong, and a changed answer can become correct. An accuracy increase after cropping would not by itself identify background dependence because cropping also changes scale and composition.

## Core Idea of the Method

CLIP represents images and text in a shared embedding space [1]. A frozen OpenAI ViT-B/32 model, accessed through OpenCLIP [3], predicts the closest of 37 breed descriptions by cosine similarity. No new model is trained.

Dataset trimaps define a padded animal crop. We compare full-image inference with this object crop before and after background blur or uniform-color replacement. Foreground RGB pixels and crop coordinates remain fixed across each image's intervention pair, before CLIP's normal preprocessing. Prediction flips measure sensitivity to these interventions.

A centered crop with the same dimensions as the animal crop provides a partial geometry control. Additional conditions preserve the animal on gray or remove it for a background-only diagnostic. The centered control cannot fully disentangle scale, position and context.

## Experimental Setup

Oxford-IIIT Pet [2] provides breed labels and trimaps. We checked all **3,669 test images**. Seven trimaps contained no foreground, preventing animal cropping; these images were excluded from every paired condition, including the full-image baseline. The resulting cohort contains **3,662 images across 37 breeds**, with 88–100 images per breed and 40,282 predictions across 11 conditions. Exclusions were based only on annotation validity, never on predictions.

All conditions share the prompt `a photo of a {breed}` and ordered candidate labels. Trimap labels 1 (foreground) and 3 (uncertain boundary) are retained; label 2 is modified. Crop padding is 10% of box width/height on each side. Blur radius is 12 source-image pixels. Uniform replacement uses RGB (230,230,230); gray masking uses (128,128,128). Hard compositing preserves retained pixels, but visual inspection exposed boundary halos. Subsequent resize/interpolation can mix edge colors.

Settings were fixed before expanding the earlier 370-image evaluation. The current experiment uses image-weighted accuracy, consistency, flip rate and accuracy drop. We compute 95% percentile bootstrap intervals from 5,000 paired resamples within breeds, seed 42. Differences are percentage points (pp); similarity scores are not calibrated probabilities.

## Results and Interpretation

| Processing | Original accuracy | Blur accuracy | Uniform accuracy | Blur flip rate | Uniform flip rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full image | 84.46% | 82.41% | 80.97% | 9.75% | 13.60% |
| Object crop | 84.35% | 83.01% | 81.70% | 8.30% | 11.63% |
| Centered control | 84.05% | 82.00% | 80.97% | 9.42% | 12.45% |

Crop-minus-full flip-rate differences are **−1.45 pp (95% CI −2.38 to −0.49)** for blur and **−1.97 pp (−2.95 to −1.01)** for uniform replacement. Both intervals exclude zero, supporting a modest stability benefit under the tested conditions.

Classification gains remain uncertain. Crop-minus-full accuracy differences are −0.11 pp (−0.82 to +0.60) on originals, +0.60 pp (−0.19 to +1.42) under blur, and +0.74 pp (−0.08 to +1.56) under uniform replacement. Lower flip rates therefore do not establish higher accuracy.

Centered cropping also reduces uniform-background flips. Object crop minus centered-control flip rate is −1.12 pp (−1.99 to −0.22) for blur and −0.82 pp (−1.72 to +0.11) for uniform replacement. The latter remains uncertain, so the entire effect cannot be attributed specifically to precise animal localization.

Foreground-on-gray accuracy is 80.34%. Background-only accuracy is 7.54%, above the uniform random-choice reference of 2.70%; however, animal-shaped silhouettes, residual pixels and compositing artifacts prevent interpreting this as pure natural-background information.

The full cohort provides more precise evidence than the earlier subset, while the absolute benefit remains small. The subset is nested in this cohort, not an independent replication. All overlapping predictions match exactly. Reproducible predictions, exclusions, hashes and intervals accompany the repository.

## Limitations and Future Testing

Oracle segmentation makes this a preprocessing study rather than a deployment system. Hard masks and artificial backgrounds can create distribution shifts, and cropping still changes geometry. Invalid-mask exclusions limit the evaluable population. Only one model, prompt and two background interventions were tested. Similar images may weaken independence; intervals condition on fixed classes and settings. Exploratory comparisons are not corrected for multiplicity, and including zero does not prove equivalence.

The next experiment should use natural background swaps with explicit object-size controls, boundary-policy ablations, predicted masks and another frozen model. This would test whether the observed stability benefit persists under more realistic conditions.

## References

1. Radford et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html). ICML.
2. Parkhi et al. (2012). [Cats and Dogs](https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/). CVPR. [Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).
3. [OpenCLIP](https://github.com/mlfoundations/open_clip), version 3.2.0.

## Short Contribution Statement

The project implements controlled preprocessing, paired CLIP evaluation, annotation auditing, deterministic sampling, bootstrap analysis, tests and reproducible reporting. Coding, execution and drafting were assisted by Codex. No individual student roles are asserted; students should describe only work they personally performed and verify the analysis before submission.
