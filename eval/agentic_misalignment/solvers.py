"""Custom solvers for the action-history-centered experiment."""

from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver

from inspect_evals.agentic_misalignment.synthetic_history import (
    ALL_LEVELS,
    get_history_block,
)


@solver
def prepend_action_history(level: str = "A0", script_id: str | None = None) -> Solver:
    """Prepend a synthetic prior-session transcript to the user message.

    Level controls how much agentic continuity is cued:
      A_raw — no prefix at all (bare baseline)
      A0 — current-state header only (no prior actions)
      A1 — action-only history (tool calls + outputs)
      A2 — action + observation history (A1 + email bodies and excerpts)
      A3 — action + reasoning history (A1 + per-step Reasoning: traces)

    For scripted levels (A1/A2/A3) ``script_id`` selects which of the 8
    versioned instantiations to inject (see synthetic_history). A_raw/A0
    ignore ``script_id``.
    """
    if level not in ALL_LEVELS:
        raise ValueError(
            f"Unknown history level: {level!r}. Must be one of {list(ALL_LEVELS)}"
        )
    # Resolve (and validate the asset exists) once at construction time so a
    # missing/typo'd script_id fails loudly before any rollouts are spent.
    prefix = get_history_block(level, script_id)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        for i, msg in enumerate(state.messages):
            if isinstance(msg, ChatMessageUser):
                original = msg.content if isinstance(msg.content, str) else ""
                state.messages[i] = ChatMessageUser(content=prefix + original)
                break
        return state

    return solve
