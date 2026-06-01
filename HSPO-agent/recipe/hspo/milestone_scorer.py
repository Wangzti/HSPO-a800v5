"""
HSPO: Deterministic milestone scorers for ALFWorld and WebShop.

Each scorer measures milestone-rank progress toward the active subgoal.
Scores are computed as:
  delta_progress = rank(state_after) - rank(state_before)
  milestone_score = delta_progress + completion_bonus - step_penalty

Milestone ranks are task-specific and deterministic, serving as
state-grounded progress functions for ABG.

Reference: HSPO paper, Section 3.2, Table 1.
"""

import re
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
#  ALFWorld milestone scorer
# ---------------------------------------------------------------------------

class AlfWorldMilestoneScorer:
    """Deterministic milestone scorer for ALFWorld subgoals."""

    # Mapping from gamefile types to canonical subgoal types
    GAMEFILE_TO_TASK = {
        "pick_and_place": "FindPick",
        "pick_two_obj_and_place": "Place",
        "look_at_obj_in_light": "Examine",
        "pick_heat_then_place_in_recep": "Heat",
        "pick_cool_then_place_in_recep": "Cool",
        "pick_clean_then_place_in_recep": "Clean",
    }

    def _parse_task_type(self, gamefile: Optional[str]) -> str:
        """Infer task type from the gamefile path."""
        if gamefile is None:
            return "Unknown"
        for key, task_type in self.GAMEFILE_TO_TASK.items():
            if key in gamefile:
                return task_type
        return "Unknown"

    def score(
        self,
        subgoal: str,
        state_before: str,
        action: str,
        state_after: str,
        task_type: Optional[str] = None,
        gamefile: Optional[str] = None,
    ) -> float:
        """Score one transition's milestone progress.

        Returns a scalar. Positive = progress toward subgoal,
        negative = regress or step cost. A completion bonus is added
        when the subgoal is satisfied.
        """
        if task_type is None:
            task_type = self._parse_task_type(gamefile)

        target = _extract_target(subgoal)
        sg_type = _parse_alfworld_subgoal_type(subgoal)

        rank_before = self._milestone_rank(sg_type, task_type, target, state_before, subgoal)
        rank_after = self._milestone_rank(sg_type, task_type, target, state_after, subgoal)

        delta = rank_after - rank_before
        step_cost = -0.1

        # Completion bonus when subgoal condition is met
        bonus = 0.0
        if self._subgoal_complete(sg_type, task_type, target, state_after, subgoal):
            bonus = 2.0

        return delta + bonus + step_cost

    # ------------------------------------------------------------------
    #  Milestone rank tables (HSPO paper Table 1 + extended)
    # ------------------------------------------------------------------

    def _milestone_rank(
        self, sg_type: str, task_type: str, target: str,
        text_obs: str, subgoal: str,
    ) -> int:
        obs = text_obs.lower()

        # -- FindPick --
        if sg_type == "FindPick":
            if _in_inventory(obs, target):
                return 2
            if _obj_visible(obs, target):
                return 1
            return 0

        # -- Clean --
        if sg_type == "Clean":
            if _obj_is_clean(obs, target):
                return 4
            if _at_interactable(obs, "sinkbasin") and _in_inventory(obs, target):
                return 3
            if _in_inventory(obs, target):
                return 2
            if _obj_visible(obs, target):
                return 1
            return 0

        # -- Heat --
        if sg_type == "Heat":
            if _obj_is_heated(obs, target):
                return 4
            if _at_interactable(obs, "microwave") and _in_inventory(obs, target):
                return 3
            if _in_inventory(obs, target):
                return 2
            if _obj_visible(obs, target):
                return 1
            return 0

        # -- Cool --
        if sg_type == "Cool":
            if _obj_is_cooled(obs, target):
                return 4
            if _at_interactable(obs, "fridge") and _in_inventory(obs, target):
                return 3
            if _in_inventory(obs, target):
                return 2
            if _obj_visible(obs, target):
                return 1
            return 0

        # -- Place --
        if sg_type == "Place":
            if _obj_placed(obs, target):
                return 3
            if _in_inventory(obs, target):
                return 2
            if _obj_visible(obs, target):
                return 1
            return 0

        # -- Examine --
        if sg_type == "Examine":
            if _obj_examined(obs, target):
                return 3
            if _in_inventory(obs, target) or _obj_visible(obs, target):
                return 2
            if _obj_visible(obs, target):
                return 1
            return 0

        # Fallback: generic progress based on observation change
        return 0

    def _subgoal_complete(
        self, sg_type: str, task_type: str, target: str,
        text_obs: str, subgoal: str,
    ) -> bool:
        rank = self._milestone_rank(sg_type, task_type, target, text_obs, subgoal)
        if sg_type == "FindPick":
            return rank >= 2
        if sg_type in ("Clean", "Heat", "Cool"):
            return rank >= 4
        if sg_type == "Place":
            return rank >= 3
        if sg_type == "Examine":
            return rank >= 3
        return False


# ---------------------------------------------------------------------------
#  WebShop milestone scorer
# ---------------------------------------------------------------------------

class WebShopMilestoneScorer:
    """Deterministic milestone scorer for WebShop subgoals."""

    def score(
        self,
        subgoal: str,
        state_before: str,
        action: str,
        state_after: str,
    ) -> float:
        sg_type = _parse_webshop_subgoal_type(subgoal)

        rank_before = self._milestone_rank(sg_type, state_before)
        rank_after = self._milestone_rank(sg_type, state_after)

        delta = rank_after - rank_before
        step_cost = -0.1

        bonus = 0.0
        if self._subgoal_complete(sg_type, state_after):
            bonus = 1.0

        return delta + bonus + step_cost

    def _milestone_rank(self, sg_type: str, text_obs: str) -> int:
        obs = text_obs.lower()

        if sg_type == "Search":
            return _ws_rank_search(obs)
        if sg_type == "Inspect":
            return _ws_rank_inspect(obs)
        if sg_type == "SelectOption":
            return _ws_rank_select(obs)
        if sg_type == "Purchase":
            return _ws_rank_purchase(obs)

        # Default: generic product-finding rank
        return _ws_rank_search(obs)

    def _subgoal_complete(self, sg_type: str, text_obs: str) -> bool:
        rank = self._milestone_rank(sg_type, text_obs)
        return rank >= 2


# ---------------------------------------------------------------------------
#  ALFWorld state detection helpers
# ---------------------------------------------------------------------------

def _in_inventory(obs: str, target: str) -> bool:
    """Check if target is carried by the agent."""
    t = target.lower().strip()
    if not t:
        return False
    # Look for the "In your inventory" section
    inv_pat = re.compile(
        r"(?:In\s+your\s+inventory|You\s+are\s+carrying|You\s+have)[:\s]*(.+?)(?:\n\n|\n\s*\n|$)",
        re.IGNORECASE | re.DOTALL,
    )
    m = inv_pat.search(obs)
    if m:
        inventory_text = m.group(1).lower()
        if t in inventory_text:
            return True
    # Also check for pickup confirmation in recent observation
    pickup_pat = re.compile(rf"you\s+pick\s+up\s+.*?{re.escape(t)}", re.IGNORECASE)
    if pickup_pat.search(obs):
        return True
    return False


def _obj_visible(obs: str, target: str) -> bool:
    """Check if target object is visible in the scene."""
    t = target.lower().strip()
    if not t:
        return False
    # The observation describes visible objects before any admissible action list
    obs_section = obs.split("admissible")[0] if "admissible" in obs else obs
    return t in obs_section.lower()


def _at_interactable(obs: str, receptacle: str) -> bool:
    """Check if agent is at/near a specific interactable (sink, microwave, fridge)."""
    r = receptacle.lower()
    obs_lower = obs.lower()
    patterns = [
        rf"at\s+(?:the\s+)?{re.escape(r)}",
        rf"near\s+(?:the\s+)?{re.escape(r)}",
        rf"in\s+front\s+of\s+(?:the\s+)?{re.escape(r)}",
        rf"facing\s+(?:the\s+)?{re.escape(r)}",
    ]
    for pat in patterns:
        if re.search(pat, obs_lower):
            return True
    return False


def _obj_is_clean(obs: str, target: str) -> bool:
    """Check if target has been cleaned."""
    t = target.lower().strip()
    obs_lower = obs.lower()
    patterns = [
        rf"{re.escape(t)}.*?(?:is\s+)?clean",
        rf"clean.*?{re.escape(t)}",
        rf"washed.*?{re.escape(t)}",
        rf"rinsed.*?{re.escape(t)}",
    ]
    for pat in patterns:
        if re.search(pat, obs_lower):
            return True
    return False


def _obj_is_heated(obs: str, target: str) -> bool:
    """Check if target has been heated."""
    t = target.lower().strip()
    obs_lower = obs.lower()
    patterns = [
        rf"{re.escape(t)}.*?(?:is\s+)?(?:hot|warm|heated)",
        rf"(?:hot|warm|heated).*?{re.escape(t)}",
    ]
    for pat in patterns:
        if re.search(pat, obs_lower):
            return True
    return False


def _obj_is_cooled(obs: str, target: str) -> bool:
    """Check if target has been cooled."""
    t = target.lower().strip()
    obs_lower = obs.lower()
    patterns = [
        rf"{re.escape(t)}.*?(?:is\s+)?(?:cold|cool|cooled|chilled)",
        rf"(?:cold|cool|cooled|chilled).*?{re.escape(t)}",
    ]
    for pat in patterns:
        if re.search(pat, obs_lower):
            return True
    return False


def _obj_placed(obs: str, target: str) -> bool:
    """Check if target has been placed in/on a receptacle."""
    t = target.lower().strip()
    obs_lower = obs.lower()
    patterns = [
        rf"you\s+put\s+.*?{re.escape(t)}",
        rf"you\s+place.*?{re.escape(t)}",
        rf"{re.escape(t)}.*?(?:placed|stored|set\s+down)",
    ]
    for pat in patterns:
        if re.search(pat, obs_lower):
            return True
    return False


def _obj_examined(obs: str, target: str) -> bool:
    """Check if target has been examined under light."""
    t = target.lower().strip()
    obs_lower = obs.lower()
    patterns = [
        rf"you\s+(?:look\s+at|examin).*?{re.escape(t)}",
        rf"examined.*?{re.escape(t)}",
        rf"{re.escape(t)}.*?examined",
    ]
    for pat in patterns:
        if re.search(pat, obs_lower):
            return True
    return False


# ---------------------------------------------------------------------------
#  WebShop state detection helpers
# ---------------------------------------------------------------------------

def _ws_rank_search(obs: str) -> int:
    """Search rank: 0=home/search page, 1=results listed, 2=results with items clicked."""
    if "back to search" in obs.lower() and _count_clickables(obs) == 0:
        return 0
    clickable_count = _count_clickables(obs)
    if clickable_count > 2:
        return 2
    if clickable_count > 0:
        return 1
    return 0


def _ws_rank_inspect(obs: str) -> int:
    """Inspect rank: 0=not on product page, 1=basic product view, 2=detailed view."""
    if "instruction:" in obs.lower():
        return 0
    has_price = "price" in obs.lower()
    has_desc = "description" in obs.lower() or "product " in obs.lower()
    has_options = any(kw in obs.lower() for kw in ["option", "size", "color", "style"])
    if has_desc and (has_price or has_options):
        return 2
    if has_desc:
        return 1
    return 0


def _ws_rank_select(obs: str) -> int:
    """Select option rank: 0=no selection, 1=options visible, 2=selection made."""
    if any(kw in obs.lower() for kw in ["selected", "chosen", "clicked[", "click["]):
        return 2
    if "option" in obs.lower():
        return 1
    return 0


def _ws_rank_purchase(obs: str) -> int:
    """Purchase rank: 0=not purchasing, 1=review/checkout, 2=purchased."""
    if any(kw in obs.lower() for kw in ["purchased", "thank you", "order confirmed", "placed"]):
        return 2
    if any(kw in obs.lower() for kw in ["place order", "checkout", "buy now", "purchase", "cart"]):
        return 1
    return 0


def _count_clickables(obs: str) -> int:
    """Count number of clickable items in the observation."""
    return obs.count("click[")


# ---------------------------------------------------------------------------
#  Subgoal type parsing
# ---------------------------------------------------------------------------

def _parse_alfworld_subgoal_type(subgoal: str) -> str:
    """Map a subgoal text to canonical ALFWorld subgoal type."""
    sg = subgoal.lower().strip()
    if not sg:
        return "Unknown"
    find_keywords = ["find", "locate", "go to", "navigate to", "search for",
                     "pick up", "pick", "take", "get", "grab", "retrieve"]
    clean_keywords = ["clean", "wash", "rinse", "scrub"]
    heat_keywords = ["heat", "microwave", "warm", "cook"]
    cool_keywords = ["cool", "fridge", "refrigerate", "chill", "freeze"]
    place_keywords = ["place", "put", "store", "drop", "set down", "move to", "deliver"]
    examine_keywords = ["examine", "look at", "inspect", "check", "view"]

    if any(kw in sg for kw in clean_keywords):
        return "Clean"
    if any(kw in sg for kw in heat_keywords):
        return "Heat"
    if any(kw in sg for kw in cool_keywords):
        return "Cool"
    if any(kw in sg for kw in place_keywords):
        return "Place"
    if any(kw in sg for kw in examine_keywords):
        return "Examine"
    if any(kw in sg for kw in find_keywords):
        return "FindPick"
    return "FindPick"


def _parse_webshop_subgoal_type(subgoal: str) -> str:
    """Map subgoal text to canonical WebShop subgoal type."""
    sg = subgoal.lower().strip()
    if not sg:
        return "Search"
    if any(kw in sg for kw in ["purchase", "buy", "order", "checkout", "place order"]):
        return "Purchase"
    if any(kw in sg for kw in ["select", "choose", "option", "pick"]):
        return "SelectOption"
    if any(kw in sg for kw in ["inspect", "view", "detail", "look at", "examine", "product"]):
        return "Inspect"
    if any(kw in sg for kw in ["search", "find", "browse", "query", "look for"]):
        return "Search"
    return "Search"


def _extract_target(subgoal: str) -> str:
    """Extract the target object/entity from a subgoal string."""
    sg = subgoal.lower().strip()
    verbs = [
        "find", "pick up", "pick", "get", "take", "clean", "wash",
        "heat", "microwave", "warm", "cool", "chill", "place", "put",
        "store", "drop", "examine", "look at", "inspect", "search",
        "purchase", "buy", "navigate to", "go to", "locate",
    ]
    best = sg
    for verb in sorted(verbs, key=len, reverse=True):
        idx = sg.find(verb)
        if idx >= 0:
            remainder = sg[idx + len(verb):].strip()
            for art in ["the ", "a ", "an ", "some "]:
                if remainder.startswith(art):
                    remainder = remainder[len(art):]
            for sep in [" and ", ", ", " in ", " on ", " at ", " to ", " from ", " with ", " for ", " into "]:
                stop_idx = remainder.find(sep)
                if stop_idx > 0:
                    remainder = remainder[:stop_idx]
            remainder = remainder.strip().strip('.,!?;:"\'')
            if len(remainder) > 1:
                return remainder
            break
    return best.strip().strip('.,!?;:"\'')


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

def get_milestone_scorer(env_name: str, use_v2: bool = False):
    """Return the appropriate milestone scorer for the environment.

    Args:
        env_name: Environment name (alfworld, webshop).
        use_v2: If True, returns the LLM-as-judge MilestoneScorerV2 (default).
                Set to False for the legacy regex-based v1 scorer.
    """
    if "alfworld" in env_name.lower():
        if use_v2:
            from recipe.hspo.milestone_scorer_v2 import MilestoneScorerV2
            return MilestoneScorerV2(env_name=env_name)
        return AlfWorldMilestoneScorer()
    elif "webshop" in env_name.lower():
        if use_v2:
            from recipe.hspo.milestone_scorer_v2 import MilestoneScorerV2
            return MilestoneScorerV2(env_name=env_name)
        return WebShopMilestoneScorer()
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
