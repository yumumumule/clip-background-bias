# Execution and validation record

Executed locally on 28 October 2026. This record describes observed runs, not intended checks.

## Environment

- Windows 11; existing Miniconda Python 3.12.9 used to create `.venv`.
- NVIDIA GeForce RTX 5070 Ti Laptop GPU; CUDA detection and tensor computation succeeded.
- PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, OpenCLIP 3.2.0.
- Direct pins are in `requirements.txt`; the complete executed environment is in `requirements-lock-windows-cu128.txt`.
- `python -m pip check`: **No broken requirements found.**
- Exact lock installation command was checked with pip's `--dry-run`; all packages were already satisfied. A separate clean-environment reinstall was not performed.

PowerShell commands used for setup (the interpreter location is specific to this machine):

```powershell
& 'D:\Apps\miniconda\python.exe' -m venv 'clip-background-bias\.venv'
cd clip-background-bias
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Data and model integrity

Official Oxford archives were downloaded, checked against torchvision's published MD5 values, and extracted locally:

| Archive | Verified MD5 |
| --- | --- |
| images.tar.gz | `5c4f3ee8e5d25df40f4fd59a7f44e54c` |
| annotations.tar.gz | `95a8c909bbe2e81eed6a22bccdf3f68f` |

The test split contains 3,669 records. The measured subset contains exactly ten images from each of 37 breeds. Every selected image and trimap also has a SHA256 in `sample_manifest.json`.

OpenCLIP downloaded the official OpenAI TorchScript checkpoint. Verified SHA256:

```text
40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
```

The main metadata records code commit `25f5db5ca8e36b1cb76dbee89781e406069086a3`, which contains the executed inference and metric implementation. Its `git_dirty: true` reflects newly written, then-untracked result artifacts when the post-inference snapshot was taken; it is not a claim of a clean working tree. Later commits add documentation, measured artifacts and the result audit script. The inference implementation was unchanged for the repeat run.

## Unit tests and encountered issues

```text
python -m pytest -q
```

Final result: **28 passed**, in 3.94 seconds on the latest run. Tests cover raw trimap/boundary conversion, malformed and empty masks, exclusive crop bounds, padding/clipping, one-pixel foregrounds, pixel preservation, consistent paired cropping, geometry controls, image/mask dimension mismatches, missing files, balanced sampling, accuracy, consistency, paired alignment, and bootstrap shapes/reproducibility.

The first test run had one failure because dimension validation occurred after an empty-foreground check. Dimension validation was moved first, and subsequent runs passed. The first model smoke run exposed PyTorch's changed `weights_only` behavior for TorchScript archives. Loading was corrected for the SHA256-verified official OpenAI file, and real GPU/CPU inference subsequently succeeded. A benign TorchScript-dispatch warning remains visible; the model loads successfully.

## Real-data commands and outcomes

Commands below were executed with `.venv/Scripts/python.exe` (Windows path syntax) as `python`.

```text
python scripts/run_experiment.py --download --samples 5 --bootstrap 200 --output-dir results/smoke
python scripts/run_experiment.py --download --samples 370 --seed 42 --batch-size 32 --device auto --output-dir results/main
python scripts/run_experiment.py --download --samples 370 --seed 42 --batch-size 32 --device auto --output-dir local_outputs/reproduction
python scripts/run_experiment.py --samples 2 --device cpu --bootstrap 200 --output-dir local_outputs/cpu-smoke
python scripts/verify_results.py --results results/main
python scripts/plot_results.py --results results/main
python scripts/run_experiment.py --help
```

| Check | Observed outcome |
| --- | --- |
| CUDA smoke | Completed, 5 images × 11 conditions = 55 predictions |
| Main experiment | Completed, 370 images × 11 conditions = 4,070 predictions |
| Full repeat | Completed with the same settings and cached weights/text embeddings |
| Repeat comparison | `predictions.csv`, `accuracy.csv`, `paired_metrics.csv`, `results.json` and `sample_manifest.json` were **byte-identical** |
| Repeat provenance | Model SHA256 and text-cache key matched |
| CPU smoke | Completed, 2 images × 11 conditions = 22 predictions |
| Result audit | Passed: count, IDs, label mapping, cosine-score bounds, CSV/JSON agreement, bootstrap regeneration |
| Plot regeneration | Completed using saved measured CSVs, without model inference |
| CLI help | Completed and listed supported arguments |

Smoke-test CIs degenerate because each sampled breed has only one image; smoke numbers are not used as research evidence. CPU verification demonstrates successful execution, not equality of CPU/CUDA floating-point scores on all inputs.

## Visual inspection and publication contents

All four public figures were opened and inspected: accuracy, accuracy drop, prediction consistency and the experimental pipeline. Local main-run and smoke-run example grids were also opened. The crop, mask and background transformations behaved as intended; retained boundary halos and silhouette leakage were visible and are discussed in the brief and README. These checks do not establish perfect segmentation for all 370 images.

Public figures contain no dataset photographs. Photographic example grids stay under ignored `local_outputs/`; smoke results, environments, data archives, dataset images/annotations, weights and caches are excluded. A Git index audit checks that only the four intended chart/schematic PNGs are included and that no large or credential files are staged. Markdown relative links are checked before publication.

Not tested: the complete 3,669-image split, natural background swaps, predicted segmentation, other prompts/models, boundary-exclusion ablation, macOS, or Linux runtime. The GitHub Actions workflow provides a Linux CPU unit-test configuration; this local record does not claim a remote CI pass.
