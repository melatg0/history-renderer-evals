"""Config-driven experiment tooling for the action-history study.

Modules:
  config.py          — parse an experiment YAML into cells + provenance
  run_experiment.py  — launch one `inspect eval` per cell (dry-run + cost guard)
  aggregate.py       — fold .eval logs into rollouts.jsonl + aggregated.csv
"""
