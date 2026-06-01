"""
HSPO v2: LLM-as-Judge milestone scorer with subgoal parser c(g) and rank functions.

Architecture:
  c(g):  LLM-based subgoal classifier  → canonical type ∈ Σ
         Σ = {FindPick, Clean, Heat, Cool, Place, ExamineLight}  (6 types for ALFWorld)
  rank:  LLM-as-Judge milestone assessor → integer 0..max_rank
  score: delta_rank + completion_bonus + step_cost - C_side       (C_side ≡ 0)

Key properties enforced by validation:
  - Monotonicity: rank(s_{t+1}) ≥ rank(s_t) on any valid expert trajectory

Model: gpt-5  (user-specified)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  API client (shared with llm_segmentation, duplicated for self-containedness)
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_API_KEY = "sk-5fJS9kecWR0uZqcQFbD1171e66D048E8A14364D1656e991c"
DEFAULT_BASE_URL = "https://aihubmix.com/v1"
DEFAULT_MODEL = "gpt-5"


class LLMClient:
    """Thin wrapper around the OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_key: str = DEFAULT_API_KEY,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _call(self, messages: List[Dict[str, str]]) -> str:
        import openai

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,  # we handle retries ourselves
        )

        last_error = None
        for attempt in range(max(1, self.max_retries)):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                print(f"[LLMClient] API error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
        raise RuntimeError(f"All API retries exhausted. Last error: {last_error}")

    def call_json(self, messages: List[Dict[str, str]]) -> Optional[dict]:
        """Call the API and parse the response as JSON. Returns None on failure."""
        try:
            raw = self._call(messages)
        except Exception as e:
            print(f"[LLMClient] call_json failed (API error): {e}")
            return None
        # Try to extract JSON block from markdown / surrounding text
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match is None:
            print(f"[LLMClient] No JSON object found in response: {raw[:200]}")
            return None
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*}", "}", json_match.group(0))
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                print(f"[LLMClient] Could not parse JSON: {raw[:200]}")
                return None


# ═══════════════════════════════════════════════════════════════════════════
#  1.  LLM Subgoal Parser  c(g) : text → canonical type ∈ Σ
# ═══════════════════════════════════════════════════════════════════════════

# Canonical set Σ for ALFWorld (6 primitive subgoal types)
ALFWORLD_SUBGOAL_TYPES = [
    "FindPick",         # locate + pick up target object
    "Clean",            # wash/clean target object at sink
    "Heat",             # heat target object in microwave
    "Cool",             # cool target object in fridge
    "Place",            # put target object in/on receptacle
    "ExamineLight",     # examine target object under light source
]

SUBGOAL_PARSER_SYSTEM = f"""You are a subgoal classifier for the ALFWorld embodied environment.

ALFWorld has exactly 6 canonical subgoal types:
{json.dumps(ALFWORLD_SUBGOAL_TYPES, indent=2)}

Definitions:
- FindPick: Locate, navigate to, pick up, take, or retrieve an object.
- Clean: Wash, rinse, scrub, or clean an object (requires sink interaction).
- Heat: Heat, warm, microwave, or cook an object (requires microwave interaction).
- Cool: Cool, chill, refrigerate, or freeze an object (requires fridge interaction).
- Place: Put, place, store, set down, or move an object to a receptacle.
- ExamineLight: Examine, look at, inspect, or view an object (requires light source interaction).

Output ONLY a JSON object with the format:
{{"subgoal_type": "<one of the 6 canonical types>", "target": "<target object>", "confidence": <0.0-1.0>}}"""


class LLMSubgoalParser:
    """Maps a free-text subgoal to a canonical type c(g) ∈ Σ."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient()
        # Simple LRU cache to reduce API calls
        self._cache: Dict[str, str] = {}

    def classify(self, subgoal: str) -> str:
        """Return canonical subgoal type ∈ Σ."""
        sg = subgoal.strip()
        if not sg:
            return "FindPick"

        if sg in self._cache:
            return self._cache[sg]

        messages = [
            {"role": "system", "content": SUBGOAL_PARSER_SYSTEM},
            {"role": "user", "content": f"Classify this subgoal: \"{sg}\""},
        ]

        result = self.client.call_json(messages)
        if result is not None:
            parsed_type = result.get("subgoal_type", "FindPick")
            if parsed_type in ALFWORLD_SUBGOAL_TYPES:
                self._cache[sg] = parsed_type
                return parsed_type

        # Fallback to keyword-based heuristic
        return _fallback_subgoal_classify(sg)


def _fallback_subgoal_classify(subgoal: str) -> str:
    """Deterministic fallback when LLM is unavailable."""
    sg = subgoal.lower().strip()
    if any(kw in sg for kw in ["clean", "wash", "rinse", "scrub"]):
        return "Clean"
    if any(kw in sg for kw in ["heat", "microwave", "warm", "cook"]):
        return "Heat"
    if any(kw in sg for kw in ["cool", "fridge", "refrigerate", "chill", "freeze"]):
        return "Cool"
    if any(kw in sg for kw in ["place", "put", "store", "drop", "set down", "move to", "deliver"]):
        return "Place"
    if any(kw in sg for kw in ["examine", "look at", "inspect", "check", "view"]):
        return "ExamineLight"
    return "FindPick"


# ═══════════════════════════════════════════════════════════════════════════
#  2.  LLM Rank Judge  —  milestone progress assessment
# ═══════════════════════════════════════════════════════════════════════════

# Per-type max ranks (same as HSPO paper Table 1)
MAX_RANKS = {
    "FindPick": 2,
    "Clean": 4,
    "Heat": 4,
    "Cool": 4,
    "Place": 3,
    "ExamineLight": 3,
}

RANK_SYSTEM_PROMPT = """You are a milestone-progress assessor for the ALFWorld embodied AI environment.
Your job is to evaluate, given a subgoal and the current environment state, what rank (integer milestone) the agent has reached.

The rank must be an integer from 0 (no progress) to max_rank (subgoal complete).
CRITICAL RULE: On any valid trajectory toward a subgoal, the rank MUST increase monotonically — once a milestone is reached, the rank must never decrease.

You will receive:
- subgoal_type: one of {FindPick, Clean, Heat, Cool, Place, ExamineLight}
- subgoal: the natural-language subgoal description
- max_rank: the maximum possible rank for this subgoal type
- state_observation: the text observation of the current environment state
- task_instance: optional structured info (gamefile path, admissible commands, etc.)

Output ONLY a compact JSON object:
{"rank": <int>, "rationale": "<one-sentence reason>"}"""


def _rank_prompt(
    subgoal_type: str,
    subgoal: str,
    max_rank: int,
    state_obs: str,
    instance_info: Optional[str] = None,
) -> str:
    """Build a user prompt for the rank judge."""
    lines = [
        f"Subgoal type: {subgoal_type}",
        f"Subgoal: {subgoal}",
        f"Max rank (0 = no progress, {max_rank} = complete): {max_rank}",
        "",
        f"State observation: \"{state_obs[:2000]}\"",
    ]
    if instance_info:
        lines.append(f"\nTask instance info: {instance_info[:500]}")
    return "\n".join(lines)


# Per-type milestone descriptions (used by LLM + for documentation)
MILESTONE_DESCRIPTIONS = {
    "FindPick": (
        "Rank 0 = object not visible, not in inventory.\n"
        "Rank 1 = object visible in scene.\n"
        "Rank 2 = object picked up and in inventory."
    ),
    "Clean": (
        "Rank 0 = object not visible.\n"
        "Rank 1 = object visible.\n"
        "Rank 2 = object in inventory.\n"
        "Rank 3 = object in inventory AND agent at/near sinkbasin.\n"
        "Rank 4 = object cleaned (washed/rinsed)."
    ),
    "Heat": (
        "Rank 0 = object not visible.\n"
        "Rank 1 = object visible.\n"
        "Rank 2 = object in inventory.\n"
        "Rank 3 = object in inventory AND agent at/near microwave.\n"
        "Rank 4 = object heated/hot/warm."
    ),
    "Cool": (
        "Rank 0 = object not visible.\n"
        "Rank 1 = object visible.\n"
        "Rank 2 = object in inventory.\n"
        "Rank 3 = object in inventory AND agent at/near fridge.\n"
        "Rank 4 = object cooled/cold/chilled."
    ),
    "Place": (
        "Rank 0 = object not visible, not in inventory.\n"
        "Rank 1 = object visible.\n"
        "Rank 2 = object in inventory.\n"
        "Rank 3 = object placed/put/stored in target receptacle."
    ),
    "ExamineLight": (
        "Rank 0 = object not visible.\n"
        "Rank 1 = object visible.\n"
        "Rank 2 = object in inventory AND light source visible/accessible.\n"
        "Rank 3 = object examined/looked at/ inspected under light."
    ),
}


class LLMRankJudge:
    """LLM-as-judge milestone rank assessor.

    Given (subgoal_type, subgoal, state_observation, instance_info),
    returns an integer rank in [0, max_rank].
    """

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient()
        self._cache: Dict[Tuple, int] = {}

    def assess(
        self,
        subgoal_type: str,
        subgoal: str,
        state_obs: str,
        instance_info: Optional[str] = None,
    ) -> int:
        """Return milestone rank ∈ [0, max_rank]."""
        max_rank = MAX_RANKS.get(subgoal_type, 2)

        # Cache key
        cache_key = (subgoal_type, subgoal, state_obs[:500], str(instance_info)[:200])
        if cache_key in self._cache:
            return self._cache[cache_key]

        user_prompt = _rank_prompt(subgoal_type, subgoal, max_rank, state_obs, instance_info)

        messages = [
            {"role": "system", "content": RANK_SYSTEM_PROMPT + "\n\nMilestone definitions for this type:\n"
             + MILESTONE_DESCRIPTIONS.get(subgoal_type, "")},
            {"role": "user", "content": user_prompt},
        ]

        result = self.client.call_json(messages)
        if result is not None:
            rank = result.get("rank", 0)
            if isinstance(rank, (int, float)):
                rank = int(rank)
                rank = max(0, min(rank, max_rank))
                self._cache[cache_key] = rank
                return rank

        # Fallback: deterministic heuristic
        return _fallback_rank(subgoal_type, subgoal, state_obs)


def _fallback_rank(subgoal_type: str, subgoal: str, state_obs: str) -> int:
    """Deterministic fallback rank using regex matching."""
    from recipe.hspo.milestone_scorer import AlfWorldMilestoneScorer
    scorer = AlfWorldMilestoneScorer()
    target = _extract_target(subgoal)
    return scorer._milestone_rank(subgoal_type, subgoal_type, target, state_obs, subgoal)


def _extract_target(subgoal: str) -> str:
    """Extract target object from subgoal string."""
    sg = subgoal.lower().strip()
    verbs = ["find", "pick up", "pick", "get", "take", "clean", "wash",
             "heat", "microwave", "warm", "cool", "chill", "place", "put",
             "store", "drop", "examine", "look at", "inspect", "search"]
    for verb in sorted(verbs, key=len, reverse=True):
        idx = sg.find(verb)
        if idx >= 0:
            remainder = sg[idx + len(verb):].strip()
            for art in ["the ", "a ", "an ", "some "]:
                if remainder.startswith(art):
                    remainder = remainder[len(art):]
            for sep in [" and ", ", ", " in ", " on ", " at ", " to ", " from ", " with ", " for "]:
                stop_idx = remainder.find(sep)
                if stop_idx > 0:
                    remainder = remainder[:stop_idx]
            remainder = remainder.strip().strip('.,!?;:"\'')
            if len(remainder) > 1:
                return remainder
            break
    return sg.strip().strip('.,!?;:"\'')


# ═══════════════════════════════════════════════════════════════════════════
#  3.  MilestoneScorerV2  —  main scorer
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScoreV2Config:
    step_cost: float = -0.1
    completion_bonus: float = 2.0
    c_side: float = 0.0           # side-effect penalty (set to 0 per spec)


class MilestoneScorerV2:
    """LLM-as-judge milestone scorer for HSPO (v2).

    score = delta_rank + completion_bonus + step_cost - C_side

    Where:
      delta_rank = rank(state_after) - rank(state_before)
      rank(.)    = LLMRankJudge.assess(subgoal_type, subgoal, state)
      subgoal_type = subgoal_parser.classify(subgoal)
      C_side     = 0 (reserved for future side-effect modeling)
    """

    def __init__(
        self,
        env_name: str,
        config: Optional[ScoreV2Config] = None,
        rank_judge: Optional[LLMRankJudge] = None,
        subgoal_parser: Optional[LLMSubgoalParser] = None,
    ):
        self.env_name = env_name.lower()
        self.config = config or ScoreV2Config()

        self.rank_judge = rank_judge or LLMRankJudge()
        self.subgoal_parser = subgoal_parser or LLMSubgoalParser()

        # Statistics
        self._n_calls: int = 0
        self._n_cache_hits: int = 0

    # ------------------------------------------------------------------
    #  Score
    # ------------------------------------------------------------------

    def score(
        self,
        subgoal: str,
        state_before: str,
        action: str,
        state_after: str,
        task_type: Optional[str] = None,
        gamefile: Optional[str] = None,
        instance_info: Optional[str] = None,
    ) -> float:
        """Score one transition's milestone progress.

        Returns a scalar. Positive = progress toward subgoal.
        """
        self._n_calls += 1

        # 1. Classify subgoal type  c(g)
        sg_type = self.subgoal_parser.classify(subgoal)

        # 2. Rank before / after
        rank_before = self.rank_judge.assess(
            sg_type, subgoal, state_before, instance_info,
        )
        rank_after = self.rank_judge.assess(
            sg_type, subgoal, state_after, instance_info,
        )

        # 3. Delta
        delta = rank_after - rank_before

        # 4. Completion bonus
        max_rank = MAX_RANKS.get(sg_type, 2)
        bonus = self.config.completion_bonus if rank_after >= max_rank else 0.0

        # 5. Score = delta + bonus + step_cost - C_side
        return delta + bonus + self.config.step_cost - self.config.c_side

    # ------------------------------------------------------------------
    #  Batch scoring (reduces API calls via caching)
    # ------------------------------------------------------------------

    def score_batch(
        self,
        subgoals: List[str],
        states_before: List[str],
        actions: List[str],
        states_after: List[str],
        gamefiles: Optional[List[str]] = None,
        instance_infos: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Score a batch of transitions. Returns numpy array of scores."""
        n = len(subgoals)
        scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            scores[i] = self.score(
                subgoal=subgoals[i],
                state_before=states_before[i],
                action=actions[i],
                state_after=states_after[i],
                gamefile=gamefiles[i] if gamefiles is not None else None,
                instance_info=instance_infos[i] if instance_infos is not None else None,
            )
        return scores


# ═══════════════════════════════════════════════════════════════════════════
#  4.  Rank Function Validation  (monotonicity check)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExpertTrajectory:
    """A single expert trajectory for validation."""
    task_id: str
    subgoal_type: str
    subgoal: str
    states: List[str]            # per-step state observations
    actions: List[str]           # per-step actions
    instance_info: Optional[str] = None


@dataclass
class ValidationReport:
    passed: bool
    total_trajectories: int
    failures: List[Dict[str, Any]] = field(default_factory=list)
    per_task_results: Dict[str, bool] = field(default_factory=dict)


def validate_rank_fn(
    rank_judge: LLMRankJudge,
    expert_trajs: List[ExpertTrajectory],
    subgoal_parser: Optional[LLMSubgoalParser] = None,
) -> ValidationReport:
    """Validate that a rank function is monotonic on expert trajectories.

    For each expert trajectory, the rank along time must be monotonically
    non-decreasing: rank(state_{t+1}) >= rank(state_t) for all t.

    Args:
        rank_judge: The LLMRankJudge to validate.
        expert_trajs: List of expert demonstration trajectories.
        subgoal_parser: Optional; if provided, uses LLM subgoal parser to
                        auto-classify the subgoal_type.

    Returns:
        ValidationReport with pass/fail status and per-task results.
    """
    report = ValidationReport(passed=True, total_trajectories=len(expert_trajs))
    failures = []

    for traj in expert_trajs:
        sg_type = traj.subgoal_type
        if subgoal_parser is not None and not sg_type:
            sg_type = subgoal_parser.classify(traj.subgoal)
        if not sg_type:
            sg_type = _fallback_subgoal_classify(traj.subgoal)

        ranks = []
        for state in traj.states:
            r = rank_judge.assess(sg_type, traj.subgoal, state, traj.instance_info)
            ranks.append(r)

        # Monotonicity check
        is_monotonic = all(ranks[i] >= ranks[i - 1] for i in range(1, len(ranks)))
        report.per_task_results[traj.task_id] = is_monotonic

        if not is_monotonic:
            report.passed = False
            failures.append({
                "task_id": traj.task_id,
                "subgoal_type": sg_type,
                "subgoal": traj.subgoal,
                "ranks": ranks,
                "violations": [
                    {"step": i, "rank_before": ranks[i - 1], "rank_after": ranks[i]}
                    for i in range(1, len(ranks))
                    if ranks[i] < ranks[i - 1]
                ],
            })

    report.failures = failures
    return report


def generate_rank_fn_llm(
    subgoal_type: str,
    state_schema: str,
    instance_fields: str,
    max_rank: int,
    expert_traj_example: str,
    client: Optional[LLMClient] = None,
) -> Optional[str]:
    """Use LLM to generate a Python rank function for a given subgoal type.

    This implements the recommended flow: LLM generates → human validates.

    Args:
        subgoal_type: One of Σ = {FindPick, Clean, Heat, Cool, Place, ExamineLight}
        state_schema: Description of the state dict structure
        instance_fields: Description of instance fields
        max_rank: Maximum rank value
        expert_traj_example: Example expert trajectory for reference

    Returns:
        Python code for the rank function, or None if generation failed.
    """
    client = client or LLMClient()

    gen_prompt = f"""Write a Python rank function for the ALFWorld environment.

Subgoal type: {subgoal_type}
Max rank: {max_rank}

State schema:
{state_schema}

Instance fields: {instance_fields}

Reference expert trajectory:
{expert_traj_example}

Write a function with signature:
def rank_{subgoal_type.lower()}(state: dict, instance: dict) -> int:
    '''
    Return integer rank from 0 to {max_rank}.
    Higher = closer to completing the subgoal.
    MUST be monotonic: rank(s_{{t+1}}) >= rank(s_t) on any valid trajectory.
    '''
    # your implementation

Requirements:
1. The function must accept (state: dict, instance: dict) and return int.
2. Must return values in [0, {max_rank}].
3. Must be monotonically non-decreasing on any valid trajectory.
4. Use only the state and instance dicts; no external API calls.
5. The state dict contains text observations under key "text_obs".

Provide only the Python code, no explanation."""

    messages = [
        {"role": "system", "content": "You are an expert Python programmer specialized in RL environment scoring."},
        {"role": "user", "content": gen_prompt},
    ]

    raw = client._call(messages)
    # Extract Python code block
    code_match = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    # If no code block, try to extract the function definition
    fn_match = re.search(r"(def rank_\w+\(.*?\).*?)(?:\n\n|\Z)", raw, re.DOTALL)
    if fn_match:
        return fn_match.group(1).strip()
    return raw.strip()


# ═══════════════════════════════════════════════════════════════════════════
#  5.  Factory  (default: v2)
# ═══════════════════════════════════════════════════════════════════════════
#
#  NOTE: The `get_milestone_scorer` factory in milestone_scorer.py should be
#  updated to return MilestoneScorerV2 by default.  See the update below.
# ═══════════════════════════════════════════════════════════════════════════

def create_v2_scorer(env_name: str, **kwargs) -> MilestoneScorerV2:
    """Create a v2 milestone scorer with LLM-as-judge."""
    return MilestoneScorerV2(env_name=env_name, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
#  6.  Self-test / smoke
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test subgoal parser (offline fallback + LLM if API available)
    parser = LLMSubgoalParser()
    for sg in ["find the mug", "clean the mug", "heat the apple", "cool the wine",
               "put the plate on the table", "look at the statue under the light"]:
        result = parser.classify(sg)
        print(f"c({sg!r}) = {result}")

    # Test rank judge (offline fallback)
    judge = LLMRankJudge()
    for sg, state in [
        ("find the mug", "You see a mug on the table."),             # rank 1
        ("find the mug", "You are carrying a mug."),                  # rank 2
        ("clean the mug", "You are carrying a clean mug."),           # rank 4
    ]:
        sg_type = parser.classify(sg)
        r = judge.assess(sg_type, sg, state)
        print(f"rank({sg_type}, {sg!r}, ...) = {r}")

    # Test scorer
    scorer = MilestoneScorerV2("alfworld")
    s = scorer.score(
        subgoal="find the mug",
        state_before="You are in a kitchen. You see a table.",
        action="go to table 1",
        state_after="You are at the table. You see a mug on the table.",
    )
    print(f"\nscore('find the mug', go to table) = {s:.2f}")
    print("(Fallback mode — no API calls made)")
