# Execution and validation record

Main evidence: full-test evaluation on 1 November 2026. Initial environment setup and smoke checks were completed on 28 October 2026. Only observed checks are recorded here.

## Environment and integrity

- Windows 11; Miniconda Python 3.12.9 used to create the isolated `.venv`.
- NVIDIA GeForce RTX 5070 Ti Laptop GPU; CUDA detection and tensor computation succeeded.
- PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, OpenCLIP 3.2.0.
- Direct dependency pins: `requirements.txt`; complete executed Windows environment: `requirements-lock-windows-cu128.txt`.
- `python -m pip check`: **No broken requirements found.**
- The platform-specific lock was checked with pip's `--dry-run`. A second clean Windows environment installation was not performed.

Official data archives matched torchvision's MD5 values:

| Archive | Verified MD5 |
| --- | --- |
| images.tar.gz | `5c4f3ee8e5d25df40f4fd59a7f44e54c` |
| annotations.tar.gz | `95a8c909bbe2e81eed6a22bccdf3f68f` |

Official OpenAI checkpoint SHA256:

```text
40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
```

## Main experiment and annotation audit

All **3,669 test records** were audited. Seven trimaps contained only background pixels. Their IDs, reasons, counts and hashes are in [excluded_samples.json](../results/main/excluded_samples.json). The same seven are excluded from all 11 conditions, leaving **3,662 images and 40,282 predictions**. Input annotations were not altered; no exclusion depends on model correctness.

The first full-split attempt stopped on an empty foreground mask. The runner gained an explicit audit/exclusion policy, with tests confirming common-cohort selection and that missing files still fail. The completed rerun encountered no further inference errors. The full-test release now enables this policy by default; `--no-exclude-invalid-masks` selects strict failure.

The completed main inference command was:

```text
python scripts/run_experiment.py --download --samples 3669 --seed 42 --batch-size 32 --device auto --exclude-invalid-masks --output-dir results/full-test-2026-11-01
```

That directory was subsequently promoted to the canonical `results/main` during repository consolidation. Recorded metadata preserves the original command/output path rather than rewriting execution history. `git_commit` identifies the prior published base; `git_dirty: true` and per-source-file SHA256 hashes record that the annotation-audit implementation was added locally. Current source files contain the executed audit, preprocessing, model and metric implementations.

The earlier 370-image release is preserved in Git history. Its 4,070 overlapping predictions matched this full run exactly. Checkpoint SHA256 and text-cache key also matched. Earlier subset repeat runs produced byte-identical predictions, metrics and manifests; this statement does not imply a separate independent dataset.

During consolidation, the new default command was executed as `python scripts/run_experiment.py --download --output-dir local_outputs/release-reproduction-2026-11-01`. It completed the full 3,662-image cohort. Predictions, accuracy and paired-metric CSVs, combined result JSON, sample manifests and exclusions were byte-identical to the promoted main results. The published inference source files also matched the SHA256 hashes recorded in the measured-run metadata.

## Tests and numerical checks

```text
python -m pytest -q
python scripts/verify_results.py --results results/main
python scripts/plot_results.py --results results/main
python scripts/analyze_results.py --results results/main
```

**30 unit tests passed**, including invalid-mask exclusion and missing-file handling. Coverage also includes raw trimap conversion, padding, tiny/empty foregrounds, clipping, pixel preservation, paired crop geometry, dimension mismatches, deterministic sampling, accuracy, consistency, paired alignment and bootstrap reproducibility.

The result audit passed: unique IDs, exclusions, label mapping, cosine-score bounds, CSV/JSON agreement and deterministic regeneration of the 5,000-replicate bootstrap tables. Supplementary comparisons and per-breed accuracies are regenerated from saved predictions. Charts use measured values only.

Initial execution exposed the PyTorch `weights_only` change for OpenAI's TorchScript archive. The loader explicitly handles the verified official file. A benign TorchScript-dispatch warning remains visible; real inference succeeds.

## Smoke checks, reproducibility and publication

- GPU smoke: five images × 11 conditions; completed.
- CPU smoke: two images × 11 conditions; completed.
- Smoke intervals degenerate with one image per selected breed and are not used as research evidence.
- Reproduction commands use new output directories; nonempty destinations are refused.
- The CLI supports configuration overrides, automatic CUDA/CPU selection and explicit strict/exclusion policies.
- Full inference was tested on Windows. The [initial Linux CPU CI run](https://github.com/yumumumule/clip-background-bias/actions/runs/40701042369) passed all 28 tests then present; the current [unit-test workflow](https://github.com/yumumumule/clip-background-bias/actions/workflows/tests.yml) runs the expanded suite on subsequent pushes.

Five public figures contain only charts and a schematic. Accuracy, robustness, consistency and effect-size charts were opened and inspected. Initial local photographic grids were also inspected; visible halos and silhouette leakage are documented. Photographs stay in ignored local outputs. Dataset images/annotations, environments, weights, caches and credentials are excluded from Git. Relative document links and staged file contents are audited before publication.

The repository uses `results/main` as the sole published result set. No duplicate follow-up result directory is retained. Academic documents and default commands use the full evaluable cohort.

Not tested: natural background swaps, predicted masks, other prompts/models, boundary-exclusion ablation, macOS inference, or full CLIP inference on Linux. The seven background-only trimaps cannot support paired object-crop evaluation.
