# Can Object-Centric Processing Reduce Background Bias in Vision Models?

**Page 1 - Problem and Method | AIST1000 research seminar**

## Background bias and shortcut learning

An image classifier can exploit correlations between an object and its surroundings instead of learning features that transfer reliably. Such shortcuts can work on a benchmark and fail when testing conditions change [1]. Background-intervention research shows that object-recognition models can respond strongly to context, even with an unchanged foreground [2]. For pet breeds, a useful model should identify the animal across different settings.

This project asks: **Can object-centric preprocessing reduce CLIP's background sensitivity without retraining it?** We distinguish prediction stability from classification correctness. An unchanged prediction can be wrong; a changed prediction can become correct. Reduced flip rates alone do not establish higher accuracy or elimination of learned background bias.

## Core idea: focus on the object

The method uses a segmentation trimap to locate the animal, computes its bounding box, and adds 10% padding on each side. Cropping removes some context and changes the animal's representation in the input. We compare the crop with complete-image inference and a centered crop of exactly the same dimensions. The centered control partially matches resizing and framing, but it is not a perfect control for animal scale or final model cropping.

## CLIP and the evaluation framework

Frozen OpenAI CLIP ViT-B/32 compares normalized image embeddings with 37 breed descriptions [3,8]. Every condition uses the same checkpoint, class order, standard preprocessing, and prompt, `a photo of a {breed}`. No model is trained or fine-tuned. Oxford-IIIT Pet supplies breed labels and trimaps [4]. All 3,669 test records were checked; seven masks contained no foreground and were excluded consistently, leaving 3,662 paired animals.

## Controlled background interventions

The original experiment applies background blur or uniform-color replacement. The extension uses twelve Places365 validation photographs: three each from living rooms, streets, forests and grassland [5,6]. Donors were visually screened and locked before inference. Each animal receives three different scene categories through a seed-42, image-ID-based assignment that does not consult labels or predictions.

Only the donor is resized and cropped to the original canvas. The animal stays at its source position and size. Primary hard compositing retains trimap labels 1 and 3 exactly; a sensitivity analysis blends only uncertain label 3 while preserving confident foreground label 1. RGB invariance is asserted for every composite. Both crop coordinates remain fixed across backgrounds and boundary policies.

## Paired statistical design

The prespecified primary endpoint is natural-swap flip rate, Object Crop minus Full Image. Each animal's three swaps are averaged first. A 5,000-resample, seed-42, breed-stratified paired cluster bootstrap resamples source animals with all repeated measurements together. Secondary comparisons address accuracy, center cropping, scene category and boundary handling. Intervals are conditional on this model, donor pool and protocol; exploratory comparisons are unadjusted for multiplicity.

<!-- PAGE BREAK -->

# Page 2 - Evidence and Limitations

## Original complete-test experiment

Across 3,662 animals, original-image accuracy was 84.46% for full images and 84.35% for object crops. Cropping reduced blur-induced flips by 1.45 percentage points (pp; 95% CI 0.49 to 2.38) and uniform-background flips by 1.97 pp (1.01 to 2.95). These support modest stability improvements; the paired accuracy differences remained uncertain. The original numerical results are preserved unchanged.

## Natural photographic backgrounds

The extension completed 10,986 animal-background assignments, evaluated under hard and feathered boundaries with three methods: 76,902 inferences including rerun originals. All original predictions matched the baseline. Hard-boundary results, averaged within source animals, were:

| Method | Original accuracy | Swap accuracy | Flip rate |
| --- | ---: | ---: | ---: |
| Full image | 84.46% | 79.34% | 16.16% |
| Object crop | 84.35% | 81.31% | 13.13% |
| Center control | 84.05% | 80.20% | 14.60% |

The primary Crop-minus-Full flip-rate difference was **-3.02 pp (95% CI -3.78 to -2.29)**, about a 19% relative reduction. Crop-minus-Full swap accuracy increased by **1.98 pp (1.36 to 2.62)**. Original accuracy was essentially unchanged, while cropping still lost 3.04 pp after swapping, compared with 5.12 pp for full images. Thus robustness improved without making predictions invariant.

Against the same-size center control, crops reduced flips by 1.47 pp (0.77 to 2.21) and improved swap accuracy by 1.11 pp (0.50 to 1.73). These exploratory results support an additional advantage over generic centered cropping.

## Boundary, scene and geometry checks

Feathered-boundary Crop-minus-Full flips were -3.05 pp (-3.80 to -2.34), similar to the primary estimate. The boundary effect's change was -0.03 pp (-0.38 to +0.33); this does not prove equivalence. All four scene categories favored Crop over Full, with point reductions of 2.33-3.72 pp, although Crop versus Center remained uncertain indoors.

The center control truncated confident foreground in 1,415 images. Among the 2,247 without that truncation, Crop-minus-Center flips were -0.55 pp (-1.31 to +0.21). Framing therefore remains a plausible contributor; the subgroup is not a randomized scale control.

## Limitations, further testing and conclusion

Twelve screened donors do not represent all photographic backgrounds. Bootstrap intervals condition on this fixed pool, not unseen scenes. Donor resizing, inconsistent lighting, retained boundary halos and semantic mask errors can create artifacts; visual QC found clothing removed by a nominally valid mask. Oracle annotations also differ from deployed segmentation. Photo redistribution is restricted [7].

Next, test larger independent donor pools, physically more plausible composites, explicit object-scale controls and predicted masks. **Object-centric cropping reduces measured background sensitivity in these experiments, but does not eliminate it or establish general background invariance.** Implementation and drafting were assisted by Codex; individual student roles are not assigned.

<!-- PAGE BREAK -->

# Page 3 - References

[1] Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., and Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2, 665-673. [doi:10.1038/s42256-020-00257-z](https://doi.org/10.1038/s42256-020-00257-z).

[2] Xiao, K., Engstrom, L., Ilyas, A., and Madry, A. (2021). Noise or Signal: The Role of Image Backgrounds in Object Recognition. *International Conference on Learning Representations (ICLR)*. [Paper](https://arxiv.org/abs/2006.09994); [authors' code and dataset](https://github.com/MadryLab/backgrounds_challenge).

[3] Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139, 8748-8763. [Paper and proceedings](https://proceedings.mlr.press/v139/radford21a.html).

[4] Parkhi, O. M., Vedaldi, A., Zisserman, A., and Jawahar, C. V. (2012). Cats and Dogs. *IEEE Conference on Computer Vision and Pattern Recognition*, 3498-3505. [doi:10.1109/CVPR.2012.6248092](https://doi.org/10.1109/CVPR.2012.6248092); [official Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

[5] Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., and Torralba, A. (2018). Places: A 10 Million Image Database for Scene Recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40(6), 1452-1464. [doi:10.1109/TPAMI.2017.2723009](https://doi.org/10.1109/TPAMI.2017.2723009).

[6] Zhou, B., and the Places dataset team. Places365 Data Development Kit: category definitions, validation identifiers and dataset documentation. [Official development kit](https://github.com/zhoubolei/places_devkit); [official validation archive](https://data.csail.mit.edu/places/places365/val_256.tar). The twelve donor IDs, selection decisions and SHA256 hashes are provided in the repository's background manifest.

[7] MIT Places2. Official dataset download terms, archived 20 February 2021: use restricted to educational or noncommercial research; image redistribution prohibited. Image copyrights remain with their owners. [Archived terms](https://web.archive.org/web/20210220220343/http://places2.csail.mit.edu/download.html); [current project documentation](https://github.com/CSAILVision/places365). Donor photographs and composites are not published.

[8] OpenCLIP contributors. OpenCLIP, version 3.2.0. [Official implementation](https://github.com/mlfoundations/open_clip/tree/v3.2.0). Software reference for the frozen CLIP inference pipeline.
