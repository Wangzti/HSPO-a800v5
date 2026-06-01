# Copyright 2025 HSPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
HSPO – Hierarchical Sub-goal conditioned Process Optimization.

Package layout:
  hspo.config       – HSPOConfig dataclass (all hyper-parameters in one place)
  hspo.types        – Shared data contracts (StepRecord, SegmentRecord, MacroTransition)
  hspo.parser       – PlanExecuteParser  (<switch>/<subgoal>/<action> extractor)
  hspo.token_mask   – TokenMaskBuilder  (per-token span masks for training)
  hspo.advantages   – compute_process_return, compute_macro_gae
  hspo.prm          – Rule-based Process Reward Models (ALFWorld, WebShop)
  hspo.curriculum   – Training phase scheduler (low_level → high_level → joint)
"""

from hspo.config import HSPOConfig
from hspo.parser import PlanExecuteParser
from hspo.token_mask import TokenMaskBuilder

__all__ = ["HSPOConfig", "PlanExecuteParser", "TokenMaskBuilder"]
