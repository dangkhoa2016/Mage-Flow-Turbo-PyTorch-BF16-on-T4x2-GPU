# Usage

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](usage.vi.md)

## Recommended path: Kaggle notebook

Use the repository notebook:

```text
notebooks/kaggle-production-demo-mage-flow-turbo-bf16-t4x2.ipynb
```

Prepare a fresh Kaggle session with:

- Accelerator: **Tesla T4 x2**
- Internet: **ON**
- Attached model: `dangkhoa2016/mage-flow-community-mage-flow-turbo`
- No extra dataset required for the canonical text-to-image demo

Expected model mount:

```text
/kaggle/input/models/dangkhoa2016/mage-flow-community-mage-flow-turbo/pytorch/default/1
```

Use a fresh kernel and **Run All exactly once**. Do not rerun a failed heavy cell and present the mixed session as final publication evidence.

## CLI entry point

The public CLI is:

```bash
python scripts/public_kaggle_demo.py
```

Optional arguments:

```bash
python scripts/public_kaggle_demo.py \
  --project-root /path/to/repository \
  --output-root /path/to/output
```

The CLI calls `public_demo.runner.run_public_demo()`, prints every gate verdict plus the run ID and summary path, and returns exit code `1` only when the final status is `FAIL`.

## Canonical run contract

The public demo is intentionally narrow:

| Setting | Value |
|---|---|
| Prompt | `a red fox in a snowy forest at golden hour, high detail` |
| Seed | `42` |
| Steps | `4` |
| CFG | `1.0` |
| Resolution | `512x512` |
| Transformer blocks | `12` |
| `split_block` | `1` |
| Attention | `sdpa` |
| CPU fallback | disabled |

The public runner accepts no overrides: prompt, seed, steps, CFG, resolution, transformer-block count, `split_block`, attention backend, dtype, model ID/revision and upstream commit/tree are all fixed by the canonical contract, and the model path is resolved only to the owner-qualified Kaggle attachment. Any other input fails closed. Its purpose is reproducible qualification/public demonstration, not a general prompt playground.

## Output

A successful run writes an evidence directory under the selected output root. The returned result includes:

- `run_id`
- final `status`
- `first_failed_gate`
- `error`
- `summary_path`
- `verdict_lines`
- output/evidence paths and reduced live facts

Keep the complete failed or successful evidence directory when reviewing a run.

## Related documentation

- [Architecture](architecture.md)
- [Python and CLI API](api.md)
- [Kaggle](kaggle.md)
- [Limitations](limitations.md)
- [Provenance](provenance.md)
