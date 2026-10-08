# SFT training source

Both ParaAgent and ToolEnv SFT use the LLaMA-Factory snapshot in [LLaMA-Factory/](LLaMA-Factory/). It is a separately installed third-party package, not part of the `paraagent` Python wheel.

## Source map

| Component | Location |
|---|---|
| ParaAgent launcher and configuration | [`../scripts/train/paraagent-sft.sh`](../scripts/train/paraagent-sft.sh), [`../configs/train/paraagent-sft.yaml`](../configs/train/paraagent-sft.yaml) |
| ToolEnv launcher and configuration | [`../scripts/train/toolenv-sft.sh`](../scripts/train/toolenv-sft.sh), [`../configs/train/toolenv-sft.yaml`](../configs/train/toolenv-sft.yaml) |
| Dataset registry | [`../data/dataset_info.json`](../data/dataset_info.json) |
| SFT workflow | [`LLaMA-Factory/src/llamafactory/train/sft/workflow.py`](LLaMA-Factory/src/llamafactory/train/sft/workflow.py) |
| Trainer | [`LLaMA-Factory/src/llamafactory/train/sft/trainer.py`](LLaMA-Factory/src/llamafactory/train/sft/trainer.py) |
| Dataset conversion | [`LLaMA-Factory/src/llamafactory/data/converter.py`](LLaMA-Factory/src/llamafactory/data/converter.py) |
| Tokenization and supervised labels | [`LLaMA-Factory/src/llamafactory/data/processor/supervised.py`](LLaMA-Factory/src/llamafactory/data/processor/supervised.py) |
| Chat templates | [`LLaMA-Factory/src/llamafactory/data/template.py`](LLaMA-Factory/src/llamafactory/data/template.py) |

## Install and run

Activate the `paraagent` LLaMA-Factory environment described in the [installation guide](../docs/installation.md). Both SFT stages share it. Run these commands from the ParaAgent repository root, not from this directory:

```bash
python -m pip install -r requirements/sft-serve.txt
python -B -c 'import llamafactory; print(llamafactory.__file__)'
```

The printed path should be inside this snapshot's `src/llamafactory/` directory. The launchers call the active environment's `llamafactory-cli`; merely copying this source does not replace an existing installation.

Run one of the two training stages with the appropriate model, data and GPU resources:

```bash
bash scripts/train/paraagent-sft.sh
bash scripts/train/toolenv-sft.sh
```

These commands use the full-run defaults. The data remains in ParaAgent's `data/` directory; do not replace its dataset registry with upstream demonstration data.

## Source

- Upstream: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
- Revision: `62ae362455801d4900a5132c7a30b23dc5fc3802`.
- License: [Apache-2.0](LLaMA-Factory/LICENSE).
- Contents: runtime source, dependency specifications, package metadata and citation. Tests, demonstration data, development scripts and training artifacts are excluded.
- Provenance: [UPSTREAM.json](UPSTREAM.json). Runtime source is unmodified; bundled READMEs describe this release.
