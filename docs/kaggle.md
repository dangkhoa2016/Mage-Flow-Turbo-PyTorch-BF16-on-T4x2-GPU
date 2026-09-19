# Kaggle

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](kaggle.vi.md)

## Target environment

The public demo targets a fresh Kaggle Notebook with:

- Accelerator: **Tesla T4 x2**
- Internet: **ON**
- Attached model: `dangkhoa2016/mage-flow-community-mage-flow-turbo`
- No additional dataset required for the canonical text-to-image demo
- GPU peaks and showcase evidence recorded under `artifacts/public-showcase/`

Expected model path:

```text
/kaggle/input/models/dangkhoa2016/
mage-flow-community-mage-flow-turbo/
pytorch/default/1
```

## Source bootstrap

The notebook clones this repository into `/kaggle/working` and resolves the committed `SOURCE_REF` (default `v1.0.0`, a release tag or an exact 40-hex commit SHA) before importing project modules.

The runtime then fetches the exact pinned upstream Mage commit and verifies both:

- upstream commit: `76bec2bb3818863f470de7e867c2dc7f1d0bfd83`
- upstream `mage_flow` tree: `946b91bcb2cac75e6cfe8399f0f7f330a2280adf`

The bootstrap dependency `loguru==0.7.3` is also verified by exact wheel size and SHA-256 before installation into the isolated bootstrap target.

## Recommended run procedure

1. Create a brand-new Kaggle Notebook.
2. Select **GPU T4 x2**.
3. Enable Internet.
4. Attach the required Kaggle model.
5. Import/open `notebooks/kaggle-production-demo-mage-flow-turbo-bf16-t4x2.ipynb`.
6. Start with a fresh kernel.
7. Run **Run All exactly once**.
8. Preserve notebook outputs and the generated evidence directory.

Do not manually upload a publication ZIP or identity JSON. The public workflow uses Git source bootstrap.

## Lane A — canonical qualification

A successful Lane A run establishes, among other checks:

- exactly two Tesla T4 devices;
- model load count of one;
- one T2I trajectory;
- GPU0 and GPU1 participation;
- four transformer invocations for the four-step profile;
- four expected 0->1 transformer boundary transfers;
- four 1->0 transformer returns;
- one VAE-input 0->1 transfer;
- BF16 materialization PASS;
- CPU fallback not observed;
- valid 512x512 RGB output.

## Lane B — production showcase

`run_public_showcase` (from `public_demo.showcase`) is a separate lane with its own single model load. It applies the same explicit dual-T4 placement and keeps the model hot across the deterministic case table at `512 / 768 / 1024`. Per-image evidence (timing, GPU allocated/reserved peaks, routing facts, SHA-256) is persisted under `artifacts/public-showcase/<run_id>/` together with `showcase-summary.json`, `case-results.json` and a manifest. The fail-closed final key is `PUBLIC_SHOWCASE_FINAL_VERDICT`.

## Failed runs

If a heavy cell fails, preserve that failed attempt. For strict publication provenance, start another **new** Kaggle session instead of editing state, deleting runtime directories, or rerunning only the failed cell.

## Notebook versus authority

The public notebook is a reproducibility/publication surface. The closed qualified authority evidence remains the formal routing/runtime qualification record and is not rewritten by a new public run.