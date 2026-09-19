# Security Policy

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](SECURITY.vi.md)

## Reporting

Do not open a public issue for a suspected vulnerability, exposed credential, unsafe artifact handling, dependency compromise, command-injection path, or other security-sensitive finding.

Use GitHub private vulnerability reporting when enabled. Otherwise email `i.am@dangkhoa.dev` with:

- concise issue description;
- affected component and revision;
- reproduction steps or proof of concept;
- expected impact;
- suggested mitigation if known.

Do not send real production credentials or private third-party data.

## Project-specific boundaries

This repository is not a hosted internet service. Its primary public execution surface is a Kaggle notebook plus a local Python/CLI runner.

Important boundaries:

- model and project source paths are validated explicitly;
- CPU fallback is forbidden by the canonical GPU contract;
- upstream source and bootstrap wheel identities are pinned/verified;
- generated evidence may contain environment paths and should be reviewed before public sharing;
- notebook Internet access means GitHub/PyPI availability and supply-chain assumptions matter.

## Supported versions

Security fixes target the current `main` branch and, after stable release, the latest supported stable release when practical. Historical commits are not guaranteed to receive backports.
