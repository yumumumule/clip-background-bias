# Four-slide presentation: 3 minutes

Original artificial-background talk · 1 November 2026. Timings total 180 seconds. The English script is 351 words; rehearse with pauses and the figures. Photographic examples remain local. The updated research brief additionally reports the completed natural-background experiment.

## Slide 1 — Problem (0:00–0:30; 30 seconds)

On slide:

- Can changing a background alter a pet-breed prediction?
- Does focusing on the animal improve stability?
- Measure correctness and consistency separately.

Script:

“Imagine a model recognizing a dog partly because of the garden behind it. If we change the garden while keeping the dog, should its prediction change? Our project asks whether focusing on the animal makes CLIP more stable. We measure whether its answer is correct, and whether that answer survives a background change.”

## Slide 2 — Method (0:30–1:25; 55 seconds)

On slide:

- Frozen OpenAI CLIP ViT-B/32; no training.
- All 3,669 test images checked; 7 invalid masks excluded consistently.
- 3,662 evaluable images; same 37 breed descriptions.
- Background blur/uniform replacement; full image versus fixed animal crop.
- Centered crop as a partial geometry control.

Visual: [experimental_pipeline.png](../results/main/figures/experimental_pipeline.png).

Script:

“CLIP compares an image with text descriptions. We give it thirty-seven breed descriptions and choose the closest match. The model stays frozen, so there is no new training.

We checked the complete test set. Seven segmentation masks had no foreground, leaving three thousand six hundred and sixty-two images for every comparison.

Dataset segmentation tells us where the animal is. We compare the complete image with a padded animal crop. We then blur the background or replace it with a uniform color, preserving foreground pixels before normal preprocessing. Crop coordinates stay fixed across versions. A centered crop with matching dimensions partly checks the effect of changing image geometry.”

## Slide 3 — Experimental Evidence (1:25–2:30; 65 seconds)

On slide:

| Metric | Full image | Animal crop |
| --- | ---: | ---: |
| Original accuracy | 84.46% | 84.35% |
| Blur flip rate | 9.75% | 8.30% |
| Uniform flip rate | 13.60% | 11.63% |

- Crop-minus-full flip differences: −1.45 pp [−2.38,−0.49]; −1.97 pp [−2.95,−1.01].
- Paired 95% bootstrap intervals exclude zero.
- Accuracy differences remain uncertain; absolute stability gains are small.

Visual: [paired_flip_rate_difference.png](../results/main/figures/paired_flip_rate_difference.png). Explain that negative values favor object cropping.

Script:

“The full images achieved eighty-four point four six percent original accuracy. Cropping achieved eighty-four point three five percent, so original accuracy was essentially unchanged.

Under background blur, predictions flipped on nine point seven five percent of full images, compared with eight point three percent of crops. With uniform backgrounds, the rates were thirteen point six and eleven point six three percent.

Cropping reduced flips by about one and a half to two percentage points. We used five thousand paired bootstrap resamples. Both confidence intervals excluded zero, supporting a modest stability benefit.

However, accuracy differences still included zero. Centered cropping also helped in some comparisons, so we cannot attribute the whole effect to locating the animal precisely. The evidence supports a small improvement under these interventions, rather than a large performance gain.”

## Slide 4 — Conclusion and Limitation (2:30–3:00; 30 seconds)

On slide:

- Object cropping modestly reduces background-induced prediction flips.
- No clear classification accuracy gain.
- Oracle masks and artificial interventions limit generalization.
- Next: larger background pools, object-size controls and predicted masks.

Script:

“Our conclusion is that object cropping can modestly improve prediction stability under these background changes, without a clear accuracy gain. We used ideal dataset masks, artificial backgrounds, and crops that also change object scale. Next, we would expand photographic background pools, control object size, and evaluate predicted masks. The lesson is to assess correctness, stability, effect size, and uncertainty together.”
