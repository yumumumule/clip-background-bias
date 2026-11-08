# Research brief figure assets

Publication-quality illustrations for *Can Object-Centric Processing Reduce Background Bias in Vision Models?* Published to this repository at the owner's explicit request on 2026-11-08. This publication does not establish or expand third-party image rights.

## Downloads

| Figure | PNG (600 DPI) | PDF |
| --- | --- | --- |
| Object-centric cropping | [PNG](figure1_object_cropping.png) | [PDF](figure1_object_cropping.pdf) |
| Natural background swap | [PNG](figure2_natural_background_swap.png) | [PDF](figure2_natural_background_swap.pdf) |

[PNG transfer package and provenance](brief_figures.zip) | [Visual preview](figures_preview.png) | [Machine-readable provenance](figure_provenance.json)

Figures are 170 mm wide; heights are 50 mm and 89 mm. PDFs embed native image arrays losslessly and use vector labels. PNG dimensions are 4016 x 1182 and 4016 x 2103. The ZIP contains both PNGs, provenance JSON and this README. Full paths in the local provenance have been replaced with repository-relative paths for publication. Original photographs, donor files, masks, model weights and the complete dataset are not included in this asset directory.

## Experimental fidelity and selection

Both figures use **havanese_55**, breed **Havanese**, zero-based class 16, from the saved 3,662-image evaluation cohort. Original dimensions: 500 x 375 px. Fixed bounding box: **[35, 1, 386, 375]**, with PIL-exclusive right/bottom; crop dimensions: 351 x 374 px.

The saved ordered donor assignments are:

1. `Places365_val_00024175.jpg`: indoor (`/l/living_room`).
2. `Places365_val_00015568.jpg`: urban (`/s/street`).
3. `Places365_val_00021776.jpg`: forest (`/f/forest/broadleaf`).

The existing project functions define the foreground, 10%-padded bounding box, hard compositing and fixed crops. Trimap labels 1 (animal) and 3 (uncertain boundary) retain exact original RGB values and full-frame location/size; label 2 is replaced by the donor. Only donors undergo the experiment's centered bicubic cover-resize. All crops use the same original box. The visible retained fringe is part of the completed experiment and was not retouched. Photographic donor resolution and implausible lighting/support surfaces remain experimental limitations.

All 12 original outcome-blind QC selections were inspected. Their limitations and individual rejection reasons are recorded in the provenance. The fallback selected the first source after filtering for short side >=350 px, foreground fraction 0.15-0.55 inclusive, padded crop <=72% of source area, and unpadded foreground at least 2.5% from every edge, then ranking the 259 eligible entries by boundary-to-retained-foreground ratio and image ID. No CLIP prediction outcome was used for selection or displayed in the figures.

## Credits and third-party image terms

**Oxford-IIIT Pet:** Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman and C. V. Jawahar, *Cats and Dogs*, CVPR 2012. [Official dataset page](https://www.robots.ox.ac.uk/~vgg/data/pets/). The current official page specifies [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and states that image copyright remains with original owners. The source photograph visibly carries a **Ralf Bitzer** signature; the photographer's identity was not independently verified. Changes in Figure 1: trimap color mapping, bounding-box overlay, crop and panel assembly. Figure 1 is provided under the dataset's stated CC BY-SA 4.0 terms to the extent those rights are available. Retain attribution, license link and change notice; ShareAlike applies to shared adaptations. This does not relicense Places imagery.

**Places365:** Bolei Zhou et al., *Places: A 10 Million Image Database for Scene Recognition*, IEEE TPAMI 2018. [Project documentation](https://github.com/CSAILVision/places365#license) confirms that image copyrights belong to their owners. [Source archive](https://data.csail.mit.edu/places/places365/val_256.tar). Exact donor identifiers and hashes are in the provenance. Individual photographer names are not supplied by the experiment manifest.

**Redistribution status remains unresolved.** The locked experiment manifest records non-commercial research/education-only use and no image redistribution, citing [archived download terms](https://web.archive.org/web/20210220220343/http://places2.csail.mit.edu/download.html). The current download page timed out and the archived terms could not be fetched during verification. These recorded restrictions have not been independently revalidated or superseded by a verified redistribution grant. The owner requested public publication, but that request is not third-party permission. Do not infer that Figure 2, its Places-derived content or the ZIP is licensed for further distribution, public reuse or classroom/LMS transfer. Establish relevant permission or an applicable exception before further distribution. The repository's MIT software license does not apply to third-party photographs or composites. No blanket open-image license is asserted for this directory.

## Reproduction and verification

With the original dataset assets restored locally and the existing experiment dependencies plus `requirements-docs.txt` installed, run from the repository root:

```powershell
.venv/Scripts/python.exe scripts/build_brief_figures.py
```

The script regenerates local exports under ignored `local_outputs/brief_assets/`; it does not publish or overwrite this tracked snapshot. It verifies source/mask/donor hashes, saved bounding boxes, the locked protocol and donor-assignment metadata. Foreground invariance is asserted for labels 1 and 3 across all full frames and crops. Source/result/configuration/brief/model-weight files are hashed before and after; no inference is run and no original result or brief is modified.

Visual QA passed on 2026-11-08: correct panel order, readable labels, faithful trimap legend, no unintended clipping, overlap or stretching. PDF-embedded image arrays were independently checked against the original and all three composites and fixed crops. Output hashes and detailed verification are in the provenance JSON.

## Captions

Figure 1. Object-centric preprocessing using ground-truth segmentation. The animal is localized using its trimap, and a padded bounding box is used to reduce surrounding context before CLIP inference.

Figure 2. Controlled natural-background interventions. The foreground animal is preserved while its surrounding background is replaced with three photographic scenes. Identical crop coordinates are applied across all conditions.
