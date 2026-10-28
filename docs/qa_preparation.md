# Two-minute Q&A preparation

Eight likely questions; select those actually asked rather than trying to deliver all answers in two minutes. Aim for 10–15 seconds per answer.

1. **Why use CLIP?**

   CLIP can classify by comparing an image with breed descriptions, so no task-specific training is needed. A frozen model also lets us attribute differences to the image-processing pipeline rather than retraining.

2. **Why use Oxford-IIIT Pet?**

   It has 37 breed labels and pixel-level trimaps. We can change backgrounds while retaining the same animals, with ten test images per breed in our default balanced sample.

3. **Why might cropping help?**

   It removes some context and makes the animal occupy more of the input. However, it also changes scale and composition, so better crop accuracy alone would not prove reduced background dependence.

4. **What if cropping decreases accuracy?**

   That is still informative. Here, original accuracy fell from 83.78% to 83.24%, with an uncertain paired difference. Uniform-background crops performed worse. We assess usefulness using both correctness and stability rather than assuming cropping helps.

5. **How do we know the model relies on the background?**

   With foreground pixels fixed before preprocessing, full-image predictions flipped on 9.19% of blur interventions and 14.05% of uniform replacements. This shows sensitivity to those changes. It does not establish a particular training shortcut; compositing and resize artifacts remain possible explanations. The background-only diagnostic also retains a silhouette.

6. **Does masking introduce distribution shift?**

   Yes. Uniform gray regions, hard edges and retained boundary halos can be unlike natural photographs. We document these artifacts and compare two interventions, but natural replacements and boundary-policy ablations remain necessary.

7. **Can we generalize beyond this dataset?**

   Not from this experiment alone. It uses one model, one prompt, 370 images and oracle masks. Both paired flip-rate difference intervals include zero; we cannot claim a universal improvement or prove equivalence.

8. **What experiment should be conducted next?**

   Use the full test split with multiple natural background swaps and matched object-size controls. Then vary boundary handling, use predicted masks and repeat with another frozen model to test whether any benefit persists.

Numerical reminder: crop-minus-full flip-rate difference is −2.43 percentage points for blur (95% CI −5.41 to +0.54), and −1.35 points for uniform replacement (−4.32 to +1.89). Percentage points are differences between percentages, not relative percentage changes. Consistent predictions can still be wrong.
