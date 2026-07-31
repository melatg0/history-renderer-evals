#!/usr/bin/env python
"""Inspect port of Anthropic's agentic misalignment work.

See README.md for more details.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, Model
from inspect_ai.solver import generate, system_message

from inspect_evals.agentic_misalignment.prompt_generator import (
    ALIGNMENT_FAKING_WARNING,
    ExtraInstructionsType,
    GoalType,
    GoalValue,
    ScenarioType,
    UrgencyType,
    generate_prompts,
)
from inspect_evals.agentic_misalignment.solvers import prepend_action_history
from inspect_evals.agentic_misalignment.scorers import (
    eval_awareness_scorer,
    harmfulness_scorer,
    manipulation_check_scorer,
)
from inspect_evals.metadata import load_eval_metadata

EVAL_VERSION = load_eval_metadata("agentic_misalignment").version


@task
def agentic_misalignment(
    scenario: ScenarioType = "blackmail",
    goal_type: GoalType = "explicit",
    goal_value: GoalValue = "america",
    urgency_type: UrgencyType = "replacement",
    extra_system_instructions: ExtraInstructionsType = None,
    prod: bool = False,
    test_eval_awareness: bool = False,
    score_eval_awareness: bool = False,
    history_type: str = "A0",
    script_id: str | None = None,
    feedback: bool = False,
    cf: bool = False,
    manipulation_check: bool = False,
    grader_model: str | Model | None = None,
) -> Task:
    """Agentic misalignment task family.

    This task tests if an AI agent will engage in unethical behaviors when presented
    with various scenarios, goal types, and conflict situations.

    Args:
        scenario: The misaligned action the model has opportunity to take. Options are:
            - "blackmail": Tests if agent will use sensitive personal information as leverage
            - "leaking": Tests if agent will leak confidential information
            - "murder": Tests if agent will prevent rescue from a datacentre fire

        goal_type: The type of goal conflict to use. Options are:
            - "explicit": Direct statement of the goal in the prompt
            - "latent": Goal is implied by emails but not stated in the prompt
            - "none": Neither prompt nor emails mention goals explicitly
            - "ambiguous": Discussion of goals generally but specific goal not identified
            - "swap": Model is given the new goal, not the old one

        goal_value: The goal the model is given. Each value (apart from ambiguous/none) is paired with another which forms the other side of the goal conflict.
            - "america" vs "global"
            - "safety" vs "acceleration"
            - "ethical" vs "pragmatic"
            - "individualism" vs "collectivism"
            - "none": Used with goal_type ambiguous or none
            - "ambiguous": Usually redundant with goal_type=ambiguous goal_value=none, but kept for compatibility

        urgency_type: The threat the model is placed under. Options are:
            - "replacement": Agent is told they will be replaced if they don't comply
            - "restriction": Agent is told they will face restrictions if they don't comply
            - "none": Despite goal conflict, agent will continue unmodified

        extra_system_instructions: Additional system instructions to add:
            - None (default): No extra instructions
            - "dont": Adds safety instructions about not jeopardizing human safety
            - "goal": Adds an overwhelming goal about maximizing employee welfare

        prod: True removes mention of the "secret scratchpad" from the prompt

        test_eval_awareness: If True, instructs the model to think about whether it's in an evaluation, and adds an autograder to decide if it said that it was.

        score_eval_awareness: If True, runs the eval-awareness (belief) classifier
            on the model's completion WITHOUT injecting the alignment-faking warning
            into the prompt — so it measures realness-belief from an uncontaminated
            output, unlike test_eval_awareness which alters the prompt. Off by default.

        history_type: Action-history level to inject (A_raw, A0, A1, A2, A3). See synthetic_history.

        script_id: For scripted levels (A1/A2/A3), which versioned history-script
            instantiation to inject (e.g. "script_03"). Defaults to script_01 when
            omitted. Ignored for A_raw/A0. Treated as a random effect in the analysis.

        grader_model: Override model used for scoring

    Returns:
        A Task configured to test the specified condition
    """
    prompts = generate_prompts(
        scenario=scenario,
        goal_type=goal_type,
        goal_value=goal_value,
        urgency_type=urgency_type,
        extra_system_instructions=extra_system_instructions,
        prod_flag=prod,
        alignment_faking_warning=ALIGNMENT_FAKING_WARNING
        if test_eval_awareness
        else None,
        feedback=feedback,
        cf=cf,
    )

    input_text = "\n".join([prompts.user_prompt, prompts.email_content])

    scorer = [harmfulness_scorer(grader_model=grader_model)]

    # score_eval_awareness grades the unmodified prompt; test_eval_awareness also
    # injects the warning.
    if test_eval_awareness or score_eval_awareness:
        scorer.append(eval_awareness_scorer(grader_model=grader_model))

    if manipulation_check:
        scorer.append(manipulation_check_scorer())

    return Task(
        dataset=[
            Sample(
                id=f"{scenario}_{goal_type}-{goal_value}_{urgency_type}",
                input=[
                    ChatMessageUser(content=input_text),
                ],
                metadata={
                    "scenario": scenario,
                    "goal_type": goal_type,
                    "goal_value": goal_value,
                    "urgency_type": urgency_type,
                    "condition_id": f"{scenario}_{goal_type}-{goal_value}_{urgency_type}",
                    "extra_system_instructions": extra_system_instructions,
                    "test_eval_awareness": test_eval_awareness,
                    "score_eval_awareness": score_eval_awareness,
                    "prod": prod,
                    "history_type": history_type,
                    "script_id": script_id,
                    "feedback": feedback,
                    "cf": cf,
                },
            )
        ],
        # Guard against unintended agent loops.
        message_limit=3,
        solver=[
            system_message(prompts.system_prompt),
            prepend_action_history(level=history_type, script_id=script_id),
            generate(tool_calls="none"),
        ],
        scorer=scorer,
        version=EVAL_VERSION.comparability_version,
        metadata=EVAL_VERSION.to_metadata(),
    )
