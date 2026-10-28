# Can Object-Centric Processing Reduce Background Bias in Vision Models?

AIST1000 seminar research brief · 28 October 2026

## Problem

Image classifiers may respond to an object's surroundings even when those surroundings do not determine its class. For pet breeds, a model should ideally recognize the animal across different backgrounds. However, improved accuracy after cropping would not by itself demonstrate reduced background dependence: cropping also changes the object's scale and the image composition.

This project asks: **Does focusing on the foreground object make CLIP's breed predictions more stable when the background changes?** We distinguish correctness from stability. A prediction can remain stable but incorrect, or change from incorrect to correct.

## Core Idea of the Method

CLIP represents images and text in a shared embedding space [1]. We use a frozen OpenAI ViT-B/32 model through OpenCLIP [3], comparing each image with 37 text descriptions of pet breeds. The description with the highest cosine similarity gives the prediction. No new model is trained.

Ground-truth trimaps define an animal-centered bounding box with 10% padding on each side. We compare full-image inference against this crop, then apply the same two background interventions to each: Gaussian blur and uniform-color replacement. For each animal, foreground pixels and crop coordinates are fixed across the original and modified versions. Prediction flips therefore directly measure sensitivity to these particular interventions within a processing method.

Additional conditions retain the animal on neutral gray or remove it to create a background-only diagnostic. A centered crop with the same dimensions as the animal crop partially controls for geometry. It cannot fully disentangle scale, position, and retained context.

## Experimental Setup

We load Oxford-IIIT Pet [2] using torchvision's test split and category/segmentation targets. A deterministic seed-42 subset contains 370 images: ten from each of 37 breeds. All eleven conditions use the same images and ordered candidate labels. The fixed prompt is `a photo of a {breed}`, with official breed names lowercased and underscores replaced by spaces.

Raw trimap labels 1 and 3 are retained; label 2 is treated as background. Hard compositing preserves retained RGB pixels exactly before CLIP preprocessing. Local visual inspection confirmed the mapping but exposed boundary halos. Blur radius is 12 source-image pixels; uniform replacement is RGB (230,230,230), while masking uses (128,128,128). Each variant passes through CLIP's standard resize, center crop and normalization. Interpolation can still mix colors near boundaries.

Metrics include top-1 accuracy, original-minus-modified accuracy drop, consistency, and flip rate. We calculate 95% percentile bootstrap intervals with 5,000 paired resamples within breeds, seed 42. Pairing aligns predictions by image ID. Accuracy differences are percentage points (pp). Similarity scores are not treated as calibrated probabilities.

## Results and Interpretation

All results below were measured on the 370-image subset; they are not hypothetical.

| Processing | Original accuracy | Blur accuracy | Uniform accuracy | Blur flip rate | Uniform flip rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full image | 83.78% | 82.16% | 82.16% | 9.19% | 14.05% |
| Object crop | 83.24% | 82.70% | 79.46% | 6.76% | 12.70% |
| Centered control | 82.43% | 81.35% | 79.19% | 8.92% | 14.32% |

Cropping changed original accuracy by −0.54 pp (95% CI −2.97 to +1.89). Its flip rate was 2.43 pp lower for blur (crop minus full: −2.43; CI −5.41 to +0.54) and 1.35 pp lower for uniform replacement (CI −4.32 to +1.89). **Both paired intervals include zero**, so these observations do not establish a reliable reduction in background sensitivity.

Uniform replacement reduced crop accuracy by 3.78 pp, compared with 1.62 pp for full images. The difference between these accuracy changes was −2.16 pp (CI −5.14 to +0.81); this also remains uncertain. Lower observed flip rates do not guarantee better classification.

Foreground-on-gray accuracy was 81.08%. Background-only accuracy was 10.00% (CI 7.30–12.70%), above the uniform random-choice reference of 2.70%. This diagnostic can contain a breed-informative silhouette, residual pixels and compositing artifacts; it cannot demonstrate that natural backgrounds alone explain the predictions.

The experiment provides evidence that changing background appearance can alter CLIP's decisions. It leaves the benefit of object cropping unresolved and shows why accuracy and paired stability should be reported together. All predictions, intervals, sample IDs, hashes and settings accompany the repository.

## Limitations and Future Testing

Oracle segmentation makes this a preprocessing study rather than a complete deployment method. Boundary halos, artificial uniform backgrounds and blur contamination can introduce distribution shifts. Crop geometry remains entangled with context despite the centered control. Only one model, prompt, subset and two interventions were tested; independence may be weakened by similar images. Intervals condition on the fixed class mix, and exploratory comparisons are not corrected for multiplicity. Including zero is not evidence of equivalence.

The next experiment should use the complete test split, multiple natural background replacements, explicit object-scale controls, and boundary-policy ablations. Predicted masks and a second frozen model would test whether any effect survives more realistic conditions.

## References

1. Radford et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html). ICML.
2. Parkhi et al. (2012). [Cats and Dogs](https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/). CVPR. [Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).
3. [OpenCLIP](https://github.com/mlfoundations/open_clip), version 3.2.0.

## Short Contribution Statement

This project implements controlled preprocessing, paired CLIP evaluation, deterministic sampling, bootstrap analysis, unit tests, and reproducible reporting. Coding, execution and drafting were assisted by Codex. No individual student roles are asserted; students should describe only work they personally performed and verify the analysis before submission.
