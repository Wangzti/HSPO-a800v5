"""
HSPO: Plan-Execute interface prompts for ALFWorld and WebShop.

These prompts induce the model to emit structured responses:
  <switch>KEEP/SWITCH</switch>
  <subgoal>...</subgoal>
  <action>...</action>

Aligned with HiPER's strict no-think format.
"""

# --------------------- ALFWorld (Plan-Execute) --------------------- #

ALFWORLD_HSPO_TEMPLATE_NO_HIS = """You are an expert agent operating in the ALFRED Embodied Environment.

You will complete the task by maintaining a SHORT-TERM SUB-GOAL at each step. A sub-goal is a small high-level objective that can typically be achieved in a few actions. A sub-goal is NOT the full task and NOT a low-level action.

At every step, you reconsider your current sub-goal based on the latest observation, and may continue it or switch to a new short-term sub-goal.

Your current observation is:
{current_observation}

Your current sub-goal is:
{current_subgoal}
(If this is the first step of the episode, this will be "None".)

Your admissible actions are:
[{admissible_actions}]

At EVERY step, you MUST output EXACTLY THREE blocks, in the order shown below:
1) A <switch> block          (KEEP or SWITCH)
2) A <subgoal> block         (the sub-goal to follow next)
3) An <action> block         (one admissible action)

NO other text, comments, or reasoning is allowed before, after, or between these blocks.

STRICT FORMAT REQUIREMENTS:
- <switch> MUST contain ONLY "KEEP" or "SWITCH".
- <subgoal> MUST appear at EVERY step:
    * If <switch>KEEP</switch>, you MUST copy the EXACT current sub-goal.
    * If <switch>SWITCH</switch>, you MUST write a NEW short sub-goal achievable in a few actions.
- <action> MUST contain EXACTLY ONE action verbatim copied AS IS from the admissible actions list.

NOW RESPOND IN THIS EXACT FORMAT (no extra text):

<switch>KEEP or SWITCH</switch>
<subgoal>the sub-goal you will follow next</subgoal>
<action>EXACTLY one ADMISSIBLE action</action>"""

ALFWORLD_HSPO_TEMPLATE = """You are an expert agent operating in the ALFRED Embodied Environment.
Your overall task is: {task_description}

You will complete the task by maintaining a SHORT-TERM SUB-GOAL at each step. A sub-goal is a small high-level objective that can typically be achieved in a few actions. A sub-goal is NOT the full task and NOT a low-level action.

At every step, you reconsider your current sub-goal based on the latest observation, and may continue it or switch to a new short-term sub-goal.

You have already taken {step_count} step(s).
Most recent {history_length} observations and actions:
{action_history}

Your current observation is:
{current_observation}

Your current sub-goal is:
{current_subgoal}

Your admissible actions are:
[{admissible_actions}]

At EVERY step, you MUST output EXACTLY THREE blocks, in the order shown below:
1) A <switch> block          (KEEP or SWITCH)
2) A <subgoal> block         (the sub-goal to follow next)
3) An <action> block         (one admissible action)

NO other text, comments, or reasoning is permitted anywhere in the output.

STRICT FORMAT REQUIREMENTS:
- <switch> MUST contain ONLY "KEEP" or "SWITCH".
- <subgoal> MUST appear at EVERY step:
    * If you KEEP, copy the EXACT current sub-goal into <subgoal>.
    * If you SWITCH, write a NEW short sub-goal achievable in a few actions and NOT the entire task.
- <action> MUST contain EXACTLY ONE action verbatim copied AS IS from the admissible actions list.

NOW RESPOND IN THIS EXACT FORMAT (no extra text):

<switch>KEEP or SWITCH</switch>
<subgoal>the sub-goal you will follow next</subgoal>
<action>EXACTLY one ADMISSIBLE action</action>"""

# --------------------- WebShop (Plan-Execute) --------------------- #

WEBSHOP_HSPO_TEMPLATE_NO_HIS = """You are an expert autonomous agent operating in the WebShop e-commerce environment. Your task is to: {task_description}.

You will complete the task by maintaining a SHORT-TERM SUB-GOAL at each step. A sub-goal is a small high-level objective that can typically be achieved in a few actions. A sub-goal is NOT the full task and NOT a low-level action.

At every step, you reconsider your current sub-goal based on the latest observation, and may continue it or switch to a new short-term sub-goal.

Your current observation is: {current_observation}.

Your current sub-goal is: {current_subgoal}.

Your admissible actions of the current situation are:
[
{available_actions}
].

At EVERY step, you MUST output EXACTLY THREE blocks, in the order shown below:
1) A <switch> block          (KEEP or SWITCH)
2) A <subgoal> block         (the sub-goal to follow next)
3) An <action> block         (one admissible action)

NO other text, comments, or reasoning is allowed before, after, or between these blocks.

STRICT FORMAT REQUIREMENTS:
- <switch> MUST contain ONLY "KEEP" or "SWITCH".
- <subgoal> MUST appear at EVERY step:
    * If <switch>KEEP</switch>, you MUST copy the EXACT current sub-goal.
    * If <switch>SWITCH</switch>, you MUST write a NEW short sub-goal achievable in a few actions.
- <action> MUST contain EXACTLY ONE action verbatim copied AS IS from the admissible actions list.

NOW RESPOND IN THIS EXACT FORMAT (no extra text):

<switch>KEEP or SWITCH</switch>
<subgoal>the sub-goal you will follow next</subgoal>
<action>EXACTLY one ADMISSIBLE action</action>"""

WEBSHOP_HSPO_TEMPLATE = """You are an expert autonomous agent operating in the WebShop e-commerce environment. Your task is to: {task_description}.

You will complete the task by maintaining a SHORT-TERM SUB-GOAL at each step. A sub-goal is a small high-level objective that can typically be achieved in a few actions. A sub-goal is NOT the full task and NOT a low-level action.

At every step, you reconsider your current sub-goal based on the latest observation, and may continue it or switch to a new short-term sub-goal.

You have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.

Your current sub-goal is: {current_subgoal}.

Your admissible actions of the current situation are:
[
{available_actions}
].

At EVERY step, you MUST output EXACTLY THREE blocks, in the order shown below:
1) A <switch> block          (KEEP or SWITCH)
2) A <subgoal> block         (the sub-goal to follow next)
3) An <action> block         (one admissible action)

NO other text, comments, or reasoning is permitted anywhere in the output.

STRICT FORMAT REQUIREMENTS:
- <switch> MUST contain ONLY "KEEP" or "SWITCH".
- <subgoal> MUST appear at EVERY step:
    * If you KEEP, copy the EXACT current sub-goal into <subgoal>.
    * If you SWITCH, write a NEW short sub-goal achievable in a few actions and NOT the entire task.
- <action> MUST contain EXACTLY ONE action verbatim copied AS IS from the admissible actions list.

NOW RESPOND IN THIS EXACT FORMAT (no extra text):

<switch>KEEP or SWITCH</switch>
<subgoal>the sub-goal you will follow next</subgoal>
<action>EXACTLY one ADMISSIBLE action</action>"""
