# Two-minute Q&A preparation

Original artificial-background Q&A · 1 November 2026. Eight likely questions; answer those actually asked. Aim for approximately 10–15 seconds per answer.

1. **Why use CLIP?**

   CLIP classifies by comparing images with breed descriptions, so no task-specific training is needed. Keeping the model frozen makes differences attributable to the input-processing pipeline rather than retraining.

2. **Why use Oxford-IIIT Pet?**

   It has 37 breed labels and pixel-level trimaps, enabling controlled background changes. We checked all 3,669 test images and used the same 3,662 valid-mask images in every condition; seven masks had no foreground.

3. **Why might cropping help?**

   It removes some context and makes the animal occupy more of the input. It also changes scale and composition, so the centered-crop control is useful. It only partially separates geometry from localization.

4. **What if cropping decreases accuracy?**

   Accuracy and stability are separate outcomes. Original accuracy was 84.46% for full images and 84.35% for crops, with an uncertain difference. Lower flip rates support a modest stability benefit, not a clear classification gain.

5. **How do we know the model responds to the background?**

   Holding foreground RGB pixels fixed before preprocessing, full-image predictions flipped on 9.75% of blur interventions and 13.60% of uniform replacements. This demonstrates sensitivity to those changes, although compositing and resizing artifacts complicate causal interpretation. The background-only diagnostic retains a silhouette.

6. **Does masking introduce distribution shift?**

   Yes. Hard edges, uniform regions and retained boundary halos can be unlike natural photographs. Gray masking reached 80.34% accuracy, below original full images. Natural replacements and boundary-policy ablations remain necessary.

7. **Can we generalize beyond this dataset?**

   Not yet. Both crop-minus-full flip-rate intervals exclude zero, but the effect is only about 1.5–2 percentage points under two artificial interventions. Oracle masks, one model/prompt and an imperfect geometry control limit generalization.

8. **What experiment should be conducted next?**

   Expand the completed photographic-background test to larger independent pools with explicit object-size controls. Then vary boundary handling, evaluate predicted masks and repeat with another frozen model. The complete evaluable test split has already been run.

Numerical reminder: crop-minus-full flip-rate differences are −1.45 percentage points for blur (95% CI −2.38 to −0.49) and −1.97 points for uniform replacement (−2.95 to −1.01). These exploratory intervals are not corrected for multiple comparisons. Stable predictions can still be incorrect.
