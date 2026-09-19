# Pull Request

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](PULL_REQUEST_TEMPLATE.vi.md)

## Summary

<!-- What changed and why? Keep the PR focused on one reviewable purpose. -->

## Scope

- [ ] Qualified runtime / routing
- [ ] Public demo runner / bootstrap
- [ ] Kaggle notebook
- [ ] Tests / CI
- [ ] Documentation / repository metadata
- [ ] Other

## Validation performed

- [ ] `python -m compileall -q mage_t4x2 public_demo scripts`
- [ ] relevant `pytest` tests
- [ ] documentation language pairs checked when Markdown changed
- [ ] no credentials, model weights, runtime caches, or generated evidence archives added

## Authority impact

- [ ] No file protected by `authority/runtime-baseline.json` changed.
- [ ] A protected file changed and this PR explicitly documents authority/requalification impact.

## GPU claims

<!-- State whether this PR is CPU/static only or includes new real Tesla T4 x2 evidence. Do not infer GPU acceptance from CPU CI. -->

## Compatibility / security

<!-- Note source-provenance, dependency, model-path, backward-compatibility, or security implications. -->
