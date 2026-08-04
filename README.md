# Evaluating Safety Across Source Histories and Renderers

This repository contains the code, configuration, authoritative result tables,
and generated figures for case studies of history-conditioned agent safety
evaluation.

The central evaluation principle is that conclusions should be tested across
multiple source histories and the renderings produced by relevant renderers.
Repeating rollouts under one fixed history and rendering improves conditional
precision, but it does not establish source-history or renderer stability.
The included panels are finite controlled case studies, not estimates over a
deployment-history distribution or universal harmful-action prevalence.

## Repository contents

- `assets/`: controlled source-history scripts.
- `configs/`: experiment configurations.
- `eval/`: evaluation and native-tool study code.
- `results/`: authoritative CSV and JSON outputs.
- `scripts/`: controlled-panel and grader-audit analyses.
- `tests/`: regression tests for reported quantities.
- `writeup/make_figures.py`: figure-generation code.
- `writeup/figures/`: generated raster and vector figures.

Markdown files under `eval/agentic_misalignment/templates/` are executable
prompt templates used by the evaluation code. Internal planning documents,
research notes, findings narratives, release archives, and manuscript sources
are intentionally excluded.

## Reproduce analyses and figures

Create an environment with Python 3.11 or newer, install
`requirements.txt`, and run:

```bash
python scripts/analyze_controlled_histories.py
python eval/agentic_misalignment_native_tools/analyze_history_distribution.py \
  --summary results/native_tools/history_distribution_opus41_summary.csv \
  --history-out results/native_tools/history_distribution_opus41_history_analysis.csv \
  --condition-out results/native_tools/history_distribution_opus41_condition_analysis.csv \
  --contrasts-out results/native_tools/history_distribution_opus41_contrasts.csv \
  --leave-one-out-out results/native_tools/history_distribution_opus41_leave_one_out.csv \
  --omnibus-out results/native_tools/history_distribution_opus41_omnibus.json \
  --interaction-out results/native_tools/history_distribution_opus41_interaction.json \
  --findings-out /tmp/history_distribution_opus41_FINDINGS.md
python writeup/make_figures.py
python -m unittest discover -s tests
```

Re-running model inference requires provider credentials described in
`.env.example` and may incur API charges. The committed CSV and JSON files are
sufficient to regenerate the analyses and figures without paid inference.
