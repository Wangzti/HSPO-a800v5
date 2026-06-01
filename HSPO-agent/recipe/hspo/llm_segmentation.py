"""
HSPO: LLM-based online trajectory segmentation.

Converts raw (observation, action) trajectories into the Plan-Execute format
by using an external LLM to propose subgoal boundaries and subgoal descriptions.

Supports:
- ALFWorld: 6 task types (FindPick, Clean, Heat, Cool, Place, Examine)
- WebShop: 4 segment types (Search, Inspect, SelectOption, Purchase)

Uses the OpenAI-compatible API for segmentation requests.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Default API configuration
DEFAULT_API_KEY = "sk-5fJS9kecWR0uZqcQFbD1171e66D048E8A14364D1656e991c"
DEFAULT_BASE_URL = "https://aihubmix.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
#  Segmentation prompts
# ---------------------------------------------------------------------------

SEGMENTATION_SYSTEM_PROMPT = """You are an expert trajectory segmenter. Your job is to analyze a sequence of agent-environment interactions and identify natural subgoal boundaries.

A subgoal is a meaningful subtask that the agent is trying to accomplish. Examples:
- ALFWorld: "find the cup", "pick up the cup", "heat the cup in microwave", "put the cup on the desk"
- WebShop: "search for the product", "inspect product details", "select the right option", "complete purchase"

For each step, you must determine:
1. Whether this step starts a NEW subgoal (SWITCH) or continues the current one (KEEP)
2. A descriptive subgoal label for the segment

Output a JSON list where each element corresponds to one step:
```json
[
  {"step": 0, "switch": "SWITCH", "subgoal": "find the mug"},
  {"step": 1, "switch": "KEEP", "subgoal": "find the mug"},
  {"step": 2, "switch": "SWITCH", "subgoal": "pick up the mug"},
  ...
]
```

RULES:
1. The first step is always SWITCH (starts the first subgoal).
2. SWITCH when the agent's objective clearly changes (new object, new location, new action type).
3. KEEP when the same subgoal continues across steps.
4. Subgoal descriptions should be short (2-8 words), action-oriented phrases.
5. Each subgoal segment typically spans 1-5 steps.
6. A subgoal is complete when its objective is achieved (e.g., object picked up, object cleaned, purchase made)."""

SEGMENTATION_USER_PROMPT = """Segment the following trajectory into subgoals:

Task: {task_description}
Environment: {env_name}

Trajectory steps:
{trajectory}

Output the segmentation as a JSON list:"""


# ---------------------------------------------------------------------------
#  Dataclass
# ---------------------------------------------------------------------------

@dataclass
class SegmentationResult:
    step: int
    switch: str           # "SWITCH" or "KEEP"
    subgoal: str          # descriptive label
    observation: str = ""
    action: str = ""


# ---------------------------------------------------------------------------
#  Client
# ---------------------------------------------------------------------------

class LLMSegmenter:
    """Uses an external LLM to segment trajectories into subgoal-bounded segments."""

    def __init__(
        self,
        api_key: str = DEFAULT_API_KEY,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _call_api(self, messages: List[Dict]) -> str:
        """Call the OpenAI-compatible API with retries."""
        import openai

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=120,
            max_retries=0,
        )

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[LLMSegmenter] API error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise

        raise RuntimeError("All API retries exhausted")

    def _parse_response(self, response_text: str, num_steps: int) -> List[SegmentationResult]:
        """Parse the LLM JSON response into SegmentationResults."""
        # Extract JSON array from response
        json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"Could not extract JSON from response: {response_text[:200]}")

        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            # Try to fix common issues
            cleaned = json_match.group(0)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            cleaned = re.sub(r",\s*}", "}", cleaned)
            parsed = json.loads(cleaned)

        results = []
        for item in parsed:
            if int(item["step"]) < num_steps:
                results.append(SegmentationResult(
                    step=int(item["step"]),
                    switch=item.get("switch", "KEEP").strip().upper(),
                    subgoal=item.get("subgoal", "").strip(),
                ))

        # Ensure we have an entry for every step
        all_results = []
        for step in range(num_steps):
            match = [r for r in results if r.step == step]
            if match:
                all_results.append(match[0])
            else:
                # Fallback: use last known subgoal or "continue"
                prev = all_results[-1] if all_results else None
                all_results.append(SegmentationResult(
                    step=step,
                    switch="KEEP",
                    subgoal=prev.subgoal if prev else "unknown",
                ))

        return all_results

    def segment_trajectory(
        self,
        observations: List[str],
        actions: List[str],
        task_description: str,
        env_name: str,
    ) -> List[SegmentationResult]:
        """Segment a single trajectory into subgoal-bounded sequences.

        Args:
            observations: List of observation strings (one per step)
            actions: List of action strings (one per step)
            task_description: Overall task description
            env_name: Environment name (alfworld or webshop)

        Returns:
            List of SegmentationResult, one per step
        """
        assert len(observations) == len(actions), "observations and actions must have equal length"

        # Truncate if too long
        max_steps = 30
        if len(observations) > max_steps:
            observations = observations[:max_steps]
            actions = actions[:max_steps]

        # Build trajectory text
        lines = []
        for i, (obs, act) in enumerate(zip(observations, actions)):
            # Truncate each observation to 500 chars
            obs_short = obs[:500] + "..." if len(obs) > 500 else obs
            act_short = act[:200] + "..." if len(act) > 200 else act
            lines.append(f"Step {i}: [Observation] {obs_short} [Action] {act_short}")

        trajectory_text = "\n".join(lines)

        messages = [
            {"role": "system", "content": SEGMENTATION_SYSTEM_PROMPT},
            {"role": "user", "content": SEGMENTATION_USER_PROMPT.format(
                task_description=task_description,
                env_name=env_name,
                trajectory=trajectory_text,
            )},
        ]

        response_text = self._call_api(messages)
        results = self._parse_response(response_text, len(observations))

        # Enrich with original data
        for r in results:
            if r.step < len(observations):
                r.observation = observations[r.step]
            if r.step < len(actions):
                r.action = actions[r.step]

        return results


# ---------------------------------------------------------------------------
#  Batch segmentation
# ---------------------------------------------------------------------------

def segment_batch_trajectories(
    batch_observations: List[List[str]],
    batch_actions: List[List[str]],
    task_descriptions: List[str],
    env_name: str,
    segmenter: Optional[LLMSegmenter] = None,
) -> List[List[SegmentationResult]]:
    """Segment a batch of trajectories.

    Args:
        batch_observations: List of trajectories, each a list of observations
        batch_actions: List of trajectories, each a list of actions
        task_descriptions: Task descriptions per trajectory
        env_name: Environment name
        segmenter: LLMSegmenter instance (creates default if None)

    Returns:
        List of segmentation result lists
    """
    if segmenter is None:
        segmenter = LLMSegmenter()

    all_results = []
    for obs_list, act_list, task_desc in zip(
        batch_observations, batch_actions, task_descriptions,
    ):
        try:
            results = segmenter.segment_trajectory(
                observations=obs_list,
                actions=act_list,
                task_description=task_desc,
                env_name=env_name,
            )
            all_results.append(results)
        except Exception as e:
            print(f"[segment_batch] Failed for task '{task_desc[:50]}...': {e}")
            # Create a fallback segmentation: all KEEP with empty subgoal
            fallback = [
                SegmentationResult(step=i, switch="KEEP" if i > 0 else "SWITCH", subgoal="")
                for i in range(len(obs_list))
            ]
            all_results.append(fallback)

    return all_results


# ---------------------------------------------------------------------------
#  Convert segmentation to Plan-Execute format
# ---------------------------------------------------------------------------

def convert_to_plan_execute_format(
    observations: List[str],
    actions: List[str],
    segmentation: List[SegmentationResult],
) -> List[str]:
    """Convert a segmented trajectory into Plan-Execute response strings.

    Each step produces: <switch>SWITCH/KEEP</switch> <subgoal>...</subgoal> <action>...</action>
    """
    formatted = []
    for seg in segmentation:
        switch_tag = f"<switch>{seg.switch}</switch>"
        subgoal_tag = f"<subgoal>{seg.subgoal}</subgoal>"
        action_tag = f"<action>{seg.action}</action>"
        formatted.append(f"{switch_tag} {subgoal_tag} {action_tag}")

    return formatted


# ---------------------------------------------------------------------------
#  Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    segmenter = LLMSegmenter()

    test_obs = [
        "You are in a kitchen. You see a mug on the table, a microwave, and a sink.",
        "You are in a kitchen. You are holding a mug. You see a microwave.",
        "You are in a kitchen. The mug is now heated. You see a table.",
    ]
    test_acts = [
        "go to table 1",
        "use microwave 1",
        "go to table 1",
    ]
    task = "heat the mug and put it on the table"

    results = segmenter.segment_trajectory(
        observations=test_obs,
        actions=test_acts,
        task_description=task,
        env_name="alfworld",
    )

    for r in results:
        print(f"Step {r.step}: {r.switch} | {r.subgoal}")
