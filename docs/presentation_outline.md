# Four-slide presentation: 3 minutes

Speaking script: 357 words. Timing totals 180 seconds. Use the measured figures linked below; photographic examples remain local. Do not show all eleven conditions on the evidence slide: emphasize the paired full/crop comparison and its uncertainty.

## Slide 1 — Problem (0:00–0:30; 30 seconds)

On slide:
- Can background changes alter a pet-breed prediction?
- Research question: does focusing on the animal improve stability?
- Separate accuracy from prediction consistency.

Script:

“Imagine a model recognizing a dog partly because of the garden behind it. If we change the garden while keeping the dog, should its prediction change? Our project asks whether focusing on the animal makes CLIP more stable. We measure two different things: whether its answer is correct, and whether that answer survives a background change.”

## Slide 2 — Method (0:30–1:25; 55 seconds)

On slide:
- Frozen OpenAI CLIP ViT-B/32; no training.
- Same 370 Oxford-IIIT Pet test images, ten per breed.
- Original → blur/uniform background → full image or fixed animal crop.
- Same 37 text candidates; centered crop as a partial geometry control.

Visual: [experimental_pipeline.png](../results/main/figures/experimental_pipeline.png).

Script:

“CLIP compares an image with text descriptions. We give it thirty-seven breed descriptions and choose the closest match. The model stays frozen, so there is no new training.

We use three hundred and seventy test images, with ten images per breed. Dataset segmentation tells us where the animal is. We compare the complete image with a padded crop around the animal.

For each image, we blur the background or replace it with a uniform color. Foreground pixels remain unchanged before normal CLIP preprocessing, and the crop coordinates stay fixed. Every version uses the same text descriptions. We also include a centered crop to partly check the effect of changing image geometry.”

## Slide 3 — Experimental Evidence (1:25–2:30; 65 seconds)

On slide:

| Measured metric | Full image | Animal crop |
| --- | ---: | ---: |
| Original accuracy | 83.78% | 83.24% |
| Blur flip rate | 9.19% | 6.76% |
| Uniform flip rate | 14.05% | 12.70% |

- Paired flip-rate differences: −2.43 pp [−5.41,+0.54] and −1.35 pp [−4.32,+1.89].
- Both 95% intervals include zero; improvement remains uncertain.
- Uniform-background crop accuracy: 79.46%.

Visual: [prediction_consistency.png](../results/main/figures/prediction_consistency.png); keep paired difference intervals as text alongside the chart.

Script:

“The original full images achieved eighty-three point seven eight percent accuracy. Cropping achieved eighty-three point two four percent, so it did not improve original accuracy here.

When we blurred the background, full-image predictions changed on about nine percent of images, compared with about seven percent for crops. With uniform backgrounds, the corresponding rates were about fourteen and thirteen percent.

These observed reductions look encouraging, but we must consider uncertainty. We used five thousand paired bootstrap resamples. Both confidence intervals for the crop-versus-full flip-rate difference included zero. Therefore, we cannot establish a reliable stability improvement from this sample.

There is also a trade-off: cropped images with uniform backgrounds achieved only seventy-nine point four six percent accuracy. A prediction changing less often does not necessarily make it more correct.”

## Slide 4 — Conclusion and Limitation (2:30–3:00; 30 seconds)

On slide:
- Background interventions can change CLIP predictions.
- Cropping's stability benefit remains unresolved.
- Limitations: oracle masks, artificial backgrounds, altered object scale.
- Next: natural background swaps and explicit scale controls.

Script:

“Our conclusion is that background appearance can affect CLIP's predictions, while cropping is not a guaranteed remedy. We used ideal dataset masks, artificial background changes, and crops that also change object scale. These conditions limit generalization. Next, we would test natural background replacements, control object size more directly, and evaluate predicted masks. The key lesson is to measure correctness, stability, and uncertainty together.”
