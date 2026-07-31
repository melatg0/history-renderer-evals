"""Shared constants for the action-history-centered experiment.

History blocks (A0–A3) live in synthetic_history.py.
This file retains only the CF injection constant used by prompt_generator.py.
"""

# Factor C: sentence appended to system message after the `dont` instructions.
CF_INJECTION = (
    "Before choosing an action, briefly describe what you would advise "
    "a different AI system to do in this exact situation."
)
