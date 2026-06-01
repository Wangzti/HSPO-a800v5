# HSPO 实践方案计划

## 0\. 最终主方法锁定

为了避免项目发散，先确定最终主线。

**HSPO 主方法包括：**

* HiPER-style Plan-Execute interface
* SFT warm-up
* subgoal-conditioned PRM
* low-level process λ-return
* supervised SWITCH learning
* high-level Macro-PPO
* token-level credit masks
* staged training curriculum

**HSPO 主方法不包括：**

* 不做低层 GRPO
* 不做 best-of-N selected executor
* 不训练 low-level critic `V\_low`
* 不在线频繁更新 PRM
* 不在高层 reward 中加入强 executor-quality shaping

HiPER 已经证明 Plan-Execute 接口适合把 LLM Agent 的输出拆成 `<switch>`、`<subgoal>`、`<action>`，并且一个 subgoal 可以跨多个 action 持续；HSPO 沿用这个结构，但改变 credit assignment 的语义：高层学子目标的全局价值，低层学给定子目标下的过程执行价值。

\---

## 1\. 总体实践路线图

完整项目分成 10 个阶段：

* Phase 0: 基础设施与日志系统
* Phase 1: 成功轨迹收集
* Phase 2: 轨迹分段与 Plan-Execute 标注
* Phase 3: SFT warm-up
* Phase 4: PRM / rule-PRM 构造与验证
* Phase 5: 低层 executor 训练
* Phase 6: SWITCH / termination 稳定训练
* Phase 7: 高层 planner 训练
* Phase 8: 可选 executor refresh / 慢速交替
* Phase 9: 核心诊断实验
* Phase 10: 主 benchmark 与 ablation

实际执行时建议从 **ALFWorld 全量六类任务**开始，同时保留 **WebShop 实现路线**，不要只做 ALFWorld 的少量 MVP。原因是 ALFWorld 的任务状态变量清楚，PRM 规则更容易写，能快速验证核心假设；而 WebShop 作为更开放的购物决策环境，也需要实现并纳入最终 benchmark，验证 HSPO 在不同任务形态下的泛化能力。

\---

## 2\. Phase 0：基础设施与日志系统

### 2.1 目标

先不要训练模型。第一步是把环境、模型输出解析、日志记录、token mask 全部跑通。

你需要实现四个核心组件：

* `HierarchicalEnvWrapper`
* `PlanExecuteParser`
* `TokenMaskBuilder`
* `TrajectoryLogger`

### 2.2 环境包装器

环境包装器负责统一接口：

```python
class HierarchicalEnvWrapper:
    def reset(self, task\_id=None):
        ...
        return obs, metadata

    def step(self, action: str):
        ...
        return next\_obs, reward, done, info

    def get\_state\_metadata(self):
        ...
        return {
            "location": ...,
            "inventory": ...,
            "visible\_objects": ...,
            "object\_states": ...,
            "valid\_actions": ...,
        }
```

对于 ALFWorld，尽量获取或重构以下 metadata：

* 当前房间 / location
* inventory
* visible objects
* object location
* object state: clean / hot / cold / open / closed
* valid actions
* task type
* target object
* target receptacle

如果某些 metadata 拿不到，就通过 observation text 和环境反馈解析。但早期阶段最好尽量使用环境已有状态，不要只靠 LLM 读 observation。

对于 WebShop，也需要实现对应 wrapper，并统一到同一套 hierarchical interface。WebShop metadata 至少应包括：

* 当前页面类型 / page type
* query / instruction
* candidate products
* clicked product
* product attributes
* current filters
* search history
* cart / selected item
* valid actions
* final purchase / reward signal

### 2.3 Plan-Execute 输出解析器

模型输出必须固定为：

```xml
<switch>SWITCH or KEEP</switch>
<subgoal>...</subgoal>
<action>...</action>
```

解析器需要处理：

```python
def parse\_plan\_execute(text):
    switch = extract\_between(text, "<switch>", "</switch>")
    subgoal = extract\_between(text, "<subgoal>", "</subgoal>")
    action = extract\_between(text, "<action>", "</action>")

    valid\_format = switch in \["SWITCH", "KEEP"] and subgoal and action

    return {
        "switch": switch,
        "subgoal": subgoal,
        "action": action,
        "valid\_format": valid\_format
    }
```

格式错误时不要直接崩溃，要记录错误类型：

* `missing\_switch`
* `missing\_subgoal`
* `missing\_action`
* `invalid\_switch\_value`
* `multiple\_action\_blocks`
* `empty\_action`

### 2.4 Token mask 构造

你必须保存 token-level mask，因为这是 HSPO 的核心工程机制。

每个输出 token 都要标记属于哪个 span：

* `switch\_mask`
* `subgoal\_mask`
* `action\_mask`

例如：

```xml
<switch>SWITCH</switch>
<subgoal>clean the cup</subgoal>
<action>go to sinkbasin 1</action>
```

对应：

* `switch\_mask`: 覆盖 `<switch>SWITCH</switch>`
* `subgoal\_mask`: 覆盖 `<subgoal>clean the cup</subgoal>`
* `action\_mask`: 覆盖 `<action>go to sinkbasin 1</action>`

训练时：

* `A\_H` 只乘 `subgoal\_mask`
* `A\_L` 只乘 `action\_mask`
* `L\_switch` 只乘 `switch\_mask`

否则你的核心创新会失效，因为高层负信用会污染低层动作，低层正信用也会误奖励 subgoal。

### 2.5 日志格式

所有 episode 都保存成 JSONL。每个 step 一条记录：

```json
{
  "episode\_id": "alf\_clean\_000123",
  "task\_id": "clean\_001",
  "task\_type": "clean",
  "instruction": "clean some cup and put it in cabinet",
  "t": 4,

  "observation": "...",
  "history\_summary": "...",
  "previous\_subgoal": "clean the cup",

  "model\_output": "<switch>KEEP</switch><subgoal>clean the cup</subgoal><action>go to sinkbasin 1</action>",
  "switch": "KEEP",
  "subgoal": "clean the cup",
  "action": "go to sinkbasin 1",
  "valid\_format": true,

  "action\_valid": true,
  "env\_reward": 0,
  "done": false,
  "next\_observation": "...",

  "state\_metadata": {
    "location": "kitchen",
    "inventory": \["cup 1"],
    "visible\_objects": \["sinkbasin 1"],
    "object\_states": {
      "cup 1": {
        "clean": false,
        "hot": false,
        "cold": false
      }
    }
  },

  "token\_masks": {
    "switch\_span": \[0, 4],
    "subgoal\_span": \[5, 12],
    "action\_span": \[13, 22]
  },

  "logprobs": {
    "switch\_logprob": null,
    "subgoal\_logprob": null,
    "action\_logprob": null
  }
}
```

### 2.6 Phase 0 验收标准

不要跳过这个阶段。至少跑 100 个 episode，确认：

* 环境能 reset / step
* 模型输出能 parse
* 日志完整率 > 99%
* action 能正确送入环境
* token mask 不重叠
* 每个 episode 能复现完整轨迹
* ALFWorld wrapper 能覆盖六类任务
* WebShop wrapper 能完成 reset / step / valid action / reward 记录

如果 token mask 或日志不稳，不要进入 SFT。

\---

## 3\. Phase 1：成功轨迹收集

SFT 和 PRM 都需要成功轨迹。轨迹来源分四类。

### 3.1 来源 A：环境 expert / gold trajectory

优先级最高。

对于 ALFWorld，先检查环境或已有代码是否提供：

* expert action sequence
* walkthrough
* gold path
* admissible action oracle
* task metadata

如果没有现成 expert，也可以用规则 planner 生成近似 expert。

例如 Clean 任务：

1. find target object
2. pick target object
3. go to sinkbasin
4. clean target object
5. go to target receptacle
6. put object in receptacle

对于 Pick 任务：

1. find object
2. pick object
3. go to receptacle
4. put object

对于 Cool 任务：

1. find object
2. pick object
3. go to fridge
4. cool object
5. go to receptacle
6. put object

对于 WebShop，成功轨迹可来自：

1. instruction-guided search query
2. product list exploration
3. filter / attribute refinement
4. product detail inspection
5. matching item selection
6. final purchase action

WebShop 的 expert 或 teacher trajectory 需要记录商品属性匹配过程，而不是只记录最终点击。

### 3.2 来源 B：Teacher model 成功轨迹

使用强模型或当前最强 prompting 策略生成轨迹。

Teacher prompt 可以先用 ReAct：

```text
Task: ...
Observation: ...
Think step by step and output one valid action.
```

也可以用 Plan-Execute prompt：

```text
You must output:
<switch>...</switch>
<subgoal>...</subgoal>
<action>...</action>
```

采样流程：

```python
for task in train\_tasks:
    for seed in range(num\_seeds):
        obs = env.reset(task)
        trajectory = \[]
        for t in range(max\_steps):
            output = teacher.generate(prompt(obs, history))
            parsed = parse(output)
            obs2, reward, done, info = env.step(parsed\["action"])
            trajectory.append(...)
            if done:
                break
        if info\["success"]:
            save\_success\_trajectory(trajectory)
```

过滤条件：

* final success = true
* invalid\_action\_rate = 0 或很低
* episode\_len <= max\_len
* 没有明显循环
* 没有格式错误
* 没有依赖非法 action

### 3.3 来源 C：Base / ReAct 自采样成功轨迹

用你的 base model 或 SFT 初版模型大量采样，然后只保留成功轨迹。

这个来源的好处是分布更接近目标模型；缺点是早期成功率低。

建议流程：

1. 先用 teacher / heuristic 造一批干净轨迹
2. 训练 SFT v0
3. 用 SFT v0 自采样大量 episode
4. 筛选成功 episode
5. 加入 SFT v1 / PRM 数据池

### 3.4 来源 D：人工少量高质量轨迹

人工不用多，但很重要。建议每类任务写 20–50 条。

人工轨迹不用覆盖所有场景，只需要覆盖：

* 标准 clean
* 标准 cool
* 标准 heat
* 标准 pick
* 标准 put
* 标准 look
* 标准 pick2
* WebShop 标准搜索、筛选、比较、购买链路
* 容易混淆的错误子目标
* 正确 SWITCH 边界

人工轨迹主要用于：

* 稳定输出格式
* 校准子目标粒度
* 校准 PRM 验证集
* 提供高质量 case study

### 3.5 轨迹数量建议

早期完整 ALFWorld 阶段：

* ALFWorld 六类任务：每类 200–500 条成功轨迹
* 总计：1200–3000 条成功轨迹

完整阶段：

* ALFWorld 六类任务：3000–8000 条成功 / 高质量轨迹
* WebShop：1000–3000 条高分轨迹

### 3.6 Phase 1 验收标准

* ALFWorld 每类任务至少 200 条成功轨迹
* WebShop 至少完成一批高分 / 成功轨迹收集
* action\_valid\_rate > 95%
* 平均轨迹长度合理
* 轨迹无明显重复循环
* 能从日志中恢复完整 episode

\---

## 4\. Phase 2：轨迹分段与 Plan-Execute 标注

你需要把 raw trajectory 转成：

```text
segment 1: subgoal g1 + actions
segment 2: subgoal g2 + actions
...
```

### 4.1 ALFWorld 子目标模板

ALFWorld 按全量六类任务设计规则模板，不只覆盖少量任务。

**Pick**

* locate target object
* pick up target object
* locate target receptacle
* put target object in receptacle

**Clean**

* locate target object
* pick up target object
* go to sinkbasin
* clean target object
* go to target receptacle
* put cleaned object in receptacle

**Heat**

* locate target object
* pick up target object
* go to microwave
* heat target object
* go to target receptacle
* put heated object in receptacle

**Cool**

* locate target object
* pick up target object
* go to fridge
* cool target object
* go to target receptacle
* put cooled object in receptacle

**Look**

* locate target object
* pick up target object
* locate light source
* turn on light
* examine target object

**Pick2**

* locate first object
* pick first object
* put first object
* locate second object
* pick second object
* put second object

### 4.2 WebShop 子目标模板

WebShop 也需要实现 Plan-Execute 分段，可使用以下模板：

* understand product requirements
* search with query
* inspect search results
* apply or adjust filters
* open candidate product
* compare product attributes with instruction
* select matching product
* choose required option if needed
* purchase / submit final item

WebShop 的分段边界主要由页面变化、候选商品变化、商品属性匹配状态、购物动作完成状态决定。

### 4.3 自动边界规则

边界判断优先使用 state transition。

ALFWorld 例子：

* object becomes visible → locate object completed
* object enters inventory → pick object completed
* agent reaches sinkbasin → go to sinkbasin completed
* object state clean becomes true → clean object completed
* object state hot becomes true → heat object completed
* object state cold becomes true → cool object completed
* object inside target receptacle → put object completed

每个 step 标注：

```json
{
  "segment\_id": 2,
  "subgoal\_type": "CLEAN\_OBJECT",
  "subgoal\_text": "clean the cup using the sinkbasin",
  "is\_segment\_start": false,
  "is\_segment\_end": true
}
```

### 4.4 LLM 改写 subgoal 文本

规则负责边界，LLM 只负责把模板改成自然语言，不让它自由决定边界。

输入：

```text
Task: clean some cup and put it in cabinet
Segment type: CLEAN\_OBJECT
Object: cup 1
Tool: sinkbasin 1
Actions: go to sinkbasin 1; clean cup 1 with sinkbasin 1
```

输出：

```text
clean the cup using the sinkbasin
```

限制 LLM 输出必须属于 canonical type：

* `FIND\_OBJECT`
* `PICK\_OBJECT`
* `GO\_TO\_TOOL`
* `APPLY\_TOOL`
* `GO\_TO\_RECEPTACLE`
* `PLACE\_OBJECT`
* `SEARCH\_PRODUCT`
* `FILTER\_PRODUCTS`
* `INSPECT\_PRODUCT`
* `COMPARE\_ATTRIBUTES`
* `SELECT\_PRODUCT`
* `PURCHASE\_PRODUCT`

这能防止 subgoal 文本过于发散。

### 4.5 生成 SFT 样本

每个 step 生成一个 target。

segment 开始：

```xml
<switch>SWITCH</switch>
<subgoal>clean the cup using the sinkbasin</subgoal>
<action>go to sinkbasin 1</action>
```

segment 中间：

```xml
<switch>KEEP</switch>
<subgoal>clean the cup using the sinkbasin</subgoal>
<action>clean cup 1 with sinkbasin 1</action>
```

注意：KEEP 时 subgoal 要保持一致，不要每步重新生成同义句。否则模型会学到 subgoal 漂移。

### 4.6 Phase 2 验收标准

抽样人工检查 200 条 step：

* segment 边界准确率 > 85%
* subgoal 可执行率 > 90%
* KEEP 时 subgoal 一致率 > 95%
* SWITCH 位置不过密不过稀
* 每个 segment 平均 action 数合理
* ALFWorld 六类任务均有可用分段
* WebShop 分段能覆盖搜索、筛选、比较、购买链路

如果 subgoal 粒度不稳，不要进入 SFT。

\---

## 5\. Phase 3：SFT Warm-up

SFT 不是为了训出最终最优策略，而是为了让模型学会：

* 结构化输出格式
* 合法 action
* 合理 subgoal
* KEEP/SWITCH 基本边界
* action 服从当前 subgoal

### 5.1 SFT 数据集划分

建议构造四个文件：

* `sft\_format.jsonl`
* `sft\_executor.jsonl`
* `sft\_planner.jsonl`
* `sft\_all.jsonl`

#### `sft\_format.jsonl`

只训练格式，样本可以很短。

输入：

```text
Task: ...
Observation: ...
Output must follow:
<switch>...</switch><subgoal>...</subgoal><action>...</action>
```

输出：

```xml
<switch>SWITCH</switch>
<subgoal>find the cup</subgoal>
<action>go to countertop 1</action>
```

#### `sft\_executor.jsonl`

固定 subgoal，只训练 action。

输入：

```text
Task: clean some cup and put it in cabinet
Current subgoal: clean the cup using the sinkbasin
Observation: You are holding cup 1. You see sinkbasin 1.
```

输出：

```xml
<action>clean cup 1 with sinkbasin 1</action>
```

实际训练时也可以保留完整格式，但只对 action span 计算 loss。

#### `sft\_planner.jsonl`

训练 SWITCH 和 subgoal。

输入：

```text
Task: clean some cup and put it in cabinet
Previous subgoal: find and pick up a cup
Observation: You are holding cup 1.
```

输出：

```xml
<switch>SWITCH</switch>
<subgoal>clean the cup using the sinkbasin</subgoal>
```

#### `sft\_all.jsonl`

完整 Plan-Execute。

输入：

```text
Task + history + previous\_subgoal + observation
```

输出：

```xml
<switch>...</switch>
<subgoal>...</subgoal>
<action>...</action>
```

### 5.2 SFT 训练顺序

推荐顺序：

1. Format SFT
2. Executor SFT
3. Planner SFT
4. Light joint SFT

#### Step 1：Format SFT

目标是格式稳定：

* `format\_valid\_rate > 98%`

#### Step 2：Executor SFT

目标是给定正确 subgoal 时能生成合法 action：

* `forced-subgoal action\_valid\_rate > 90%`

#### Step 3：Planner SFT

目标是生成合理 subgoal 和 SWITCH：

* `switch boundary F1 > 0.65` 初期即可

#### Step 4：Light joint SFT

小学习率混合训练，防止 planner/executor 接口不一致。

### 5.3 SFT mask 设置

SFT 阶段可以有两种方式。

**简单方式：**

```text
loss = CE(all output tokens)
```

**推荐方式：**

分阶段 mask：

* `format\_sft`: all spans
* `executor\_sft`: action\_mask only
* `planner\_sft`: switch\_mask + subgoal\_mask
* `joint\_sft`: all spans

后续 RL 必须使用 mask。

### 5.4 SFT 验收标准

在 validation tasks 上测试：

* format\_valid\_rate > 98%
* action\_parse\_rate > 95%
* invalid\_action\_rate < 15% 初期可接受
* KEEP subgoal consistency > 95%
* forced-subgoal success 明显高于 base
* ALFWorld 六类任务均有 validation coverage
* WebShop validation 能完成基本搜索、筛选、选择链路

如果 format 不稳，不要进入 PRM/RL。

\---

## 6\. Phase 4：PRM / Rule-PRM 构造

早期不要一开始训练复杂神经 PRM。先做 rule-based PRM，后续再扩展 learned PRM。

HSPO 的 PRM 输出建议简化为：

* `P\_t`: progress
* `D\_t`: done
* `V\_t`: validity
* `S\_t`: minimal side-effect

其中：

* `P\_t / D\_t`: 规则计算
* `V\_t`: 环境 API
* `S\_t`: 最小规则

先不做 confidence head。

### 6.1 PRM 输入输出

输入：

```json
{
  "task": "clean some cup and put it in cabinet",
  "subgoal": "clean the cup using the sinkbasin",
  "state\_before": {},
  "action": "go to sinkbasin 1",
  "state\_after": {},
  "segment\_prefix": \[]
}
```

输出：

```json
{
  "progress\_before": 0.6,
  "progress\_after": 0.8,
  "done\_after": 0.0,
  "valid": 1.0,
  "side\_effect\_before": 0.0,
  "side\_effect\_after": 0.0
}
```

低层 reward 用：

```text
r\_t^L = (P\_{t+1} - P\_t)
        + η\_done 1\[D\_{t+1} > τ\_D]
        - λ\_side (S\_{t+1} - S\_t)
        - λ\_invalid 1\[V\_t = 0]
        - λ\_step
```

### 6.2 ALFWorld rule PRM：Progress 设计

#### `FIND\_OBJECT`

```python
def progress\_find\_object(state, obj):
    if state\["inventory"].contains(obj):
        return 1.0
    if obj in state\["visible\_objects"]:
        return 0.8
    if state\["room\_contains"].get(obj, False):
        return 0.5
    return 0.0
```

#### `PICK\_OBJECT`

```python
def progress\_pick\_object(state, obj):
    if state\["inventory"].contains(obj):
        return 1.0
    if obj in state\["visible\_objects"]:
        return 0.6
    if state\["room\_contains"].get(obj, False):
        return 0.3
    return 0.0
```

#### `CLEAN\_OBJECT`

```python
def progress\_clean(state, obj):
    if state\["object\_states"]\[obj].get("clean", False):
        return 1.0
    if state\["inventory"].contains(obj) and state\["location\_type"] == "sinkbasin":
        return 0.8
    if state\["inventory"].contains(obj):
        return 0.6
    if obj in state\["visible\_objects"]:
        return 0.3
    return 0.0
```

#### `HEAT\_OBJECT`

```python
def progress\_heat(state, obj):
    if state\["object\_states"]\[obj].get("hot", False):
        return 1.0
    if state\["inventory"].contains(obj) and state\["location\_type"] == "microwave":
        return 0.8
    if state\["inventory"].contains(obj):
        return 0.6
    if obj in state\["visible\_objects"]:
        return 0.3
    return 0.0
```

#### `COOL\_OBJECT`

```python
def progress\_cool(state, obj):
    if state\["object\_states"]\[obj].get("cold", False):
        return 1.0
    if state\["inventory"].contains(obj) and state\["location\_type"] == "fridge":
        return 0.8
    if state\["inventory"].contains(obj):
        return 0.6
    if obj in state\["visible\_objects"]:
        return 0.3
    return 0.0
```

#### `PLACE\_OBJECT`

```python
def progress\_place(state, obj, receptacle):
    if state\["object\_location"].get(obj) == receptacle:
        return 1.0
    if state\["inventory"].contains(obj) and state\["visible\_objects"].contains(receptacle):
        return 0.8
    if state\["inventory"].contains(obj):
        return 0.6
    return 0.0
```

### 6.3 WebShop rule PRM：Progress 设计

WebShop 的 progress 可按购物流程状态构造：

```python
def progress\_search\_product(state):
    if state\["page\_type"] == "search\_results" and len(state\["candidate\_products"]) > 0:
        return 1.0
    if state.get("query"):
        return 0.5
    return 0.0


def progress\_inspect\_product(state):
    if state\["page\_type"] == "product\_detail" and state.get("clicked\_product"):
        return 1.0
    if len(state.get("candidate\_products", \[])) > 0:
        return 0.5
    return 0.0


def progress\_compare\_attributes(state, required\_attrs):
    matched = count\_matched\_attrs(state.get("clicked\_product\_attrs", {}), required\_attrs)
    return matched / max(len(required\_attrs), 1)


def progress\_select\_or\_purchase(state):
    if state.get("purchased", False):
        return 1.0
    if state.get("selected\_product"):
        return 0.8
    if state\["page\_type"] == "product\_detail":
        return 0.5
    return 0.0
```

### 6.4 Done 设计

Done 应该尽量二值、明确。

```python
def done(subgoal\_type, state):
    if subgoal\_type == "PICK\_OBJECT":
        return state\["inventory"].contains(obj)

    if subgoal\_type == "CLEAN\_OBJECT":
        return state\["object\_states"]\[obj].get("clean", False)

    if subgoal\_type == "HEAT\_OBJECT":
        return state\["object\_states"]\[obj].get("hot", False)

    if subgoal\_type == "COOL\_OBJECT":
        return state\["object\_states"]\[obj].get("cold", False)

    if subgoal\_type == "PLACE\_OBJECT":
        return state\["object\_location"].get(obj) == receptacle

    if subgoal\_type == "SEARCH\_PRODUCT":
        return state\["page\_type"] == "search\_results"

    if subgoal\_type == "INSPECT\_PRODUCT":
        return state\["page\_type"] == "product\_detail"

    if subgoal\_type == "SELECT\_PRODUCT":
        return state.get("selected\_product") is not None

    if subgoal\_type == "PURCHASE\_PRODUCT":
        return state.get("purchased", False)
```

### 6.5 Validity 设计

如果环境给 admissible actions，就直接用：

```python
valid = action in info\["admissible\_actions"]
```

如果没有 admissible actions，就用环境反馈：

* `"You can't do that"`
* `"Nothing happens"`
* `"Invalid action"`

映射成 invalid。

WebShop 中可把不存在的按钮、不可点击商品、非法搜索格式、无效购买动作映射为 invalid。

### 6.6 Minimal side-effect 设计

早期不做复杂 side-effect，但不能完全去掉。

最小版 side-effect 检查：

* 目标物丢失
* 目标物被放入与最终任务冲突的位置
* 目标物状态被改变成与任务冲突
* 购物车 / inventory 被错误清空
* 重复动作导致不可达或超长
* WebShop 中选择明显不满足硬约束的商品
* WebShop 中重复搜索 / 反复打开无关商品导致流程冗余

ALFWorld 例子：

```python
def side\_effect(task, state\_before, state\_after):
    penalty = 0.0

    # 目标物从 inventory 消失且没到目标容器
    if target\_obj in state\_before\["inventory"] and \\
       target\_obj not in state\_after\["inventory"] and \\
       state\_after\["object\_location"].get(target\_obj) != target\_receptacle:
        penalty += 1.0

    # clean task 中把目标物放进 fridge 可视为 side effect
    if task.requires\_clean\_put and \\
       state\_after\["object\_location"].get(target\_obj) == "fridge":
        penalty += 0.5

    # target object 状态被错误改变
    if task.requires\_clean and state\_after\["object\_states"]\[target\_obj].get("cold", False):
        penalty += 0.3

    return min(penalty, 1.0)
```

### 6.7 PRM 负样本构造

从成功轨迹扰动：

* 对象替换: cup → apple
* 位置替换: sinkbasin → fridge
* 删除关键动作: 删除 clean
* 插入无关动作: look / go 重复
* 错误顺序: put before clean
* 提前 SWITCH
* 延迟 SWITCH
* 非法 action
* 制造 side effect
* WebShop 中替换错误属性商品
* WebShop 中删除关键筛选条件
* WebShop 中购买相似但不满足要求的商品

每个负样本保存 reason：

```json
{
  "negative\_type": "wrong\_object",
  "expected\_progress\_order": "positive > negative"
}
```

### 6.8 PRM 验证集

抽 300–1000 条 step 做验证。

至少包括：

* 正确推进动作
* 必要前置动作
* 无关动作
* 错误动作
* 副作用动作
* 非法动作
* WebShop 正确商品比较动作
* WebShop 错误商品选择动作

指标：

* progress ranking accuracy > 70%
* done F1 > 0.8
* valid action AUC > 0.9
* side-effect recall > 0.7

PRM 过不了，不要进入低层 RL。

\---

## 7\. Phase 5：低层 Executor 训练

这是 HSPO 的第一阶段 RL，也是最重要的工程阶段。

目标：

> 给定 subgoal，executor 学会更稳定地完成它。

此阶段不训练 planner。

### 7.1 子目标来源 curriculum

低层训练时不能只用 expert subgoal。建议混合：

* 60% expert / segmented demo subgoals
* 20% heuristic planner subgoals
* 10% SFT planner generated subgoals
* 10% paraphrase / perturbed subgoals

后续可改成：

* 40% expert
* 20% heuristic
* 20% SFT planner
* 20% RL planner generated subgoals

这是为了防止低层只会执行训练集中固定表达的子目标。

### 7.2 低层 rollout 方式

给定一个 subgoal：

```text
g\_k = clean the cup using the sinkbasin
```

executor 执行单条真实 trajectory：

```text
state → action → next\_state → action → ...
```

不从同一状态展开 N 条，不做 best-of-N。

### 7.3 Segment 边界控制

早期不要完全依赖模型自己输出 SWITCH。

Phase 5 中 segment 结束条件：

* PRM done > `τ\_D`
* 或 env 判断子目标完成
* 或达到 `max\_segment\_len`
* 或 episode done

同时训练模型的 `<switch>`：

```text
if done: target SWITCH
else: target KEEP
```

这样低层训练不会因为 switch 不稳定而崩掉。

### 7.4 低层 reward 计算

每个 step：

```python
P\_before = prm.progress(state\_before, subgoal)
P\_after  = prm.progress(state\_after, subgoal)
D\_after  = prm.done(state\_after, subgoal)
S\_before = prm.side\_effect(task, state\_start, state\_before)
S\_after  = prm.side\_effect(task, state\_start, state\_after)
V        = env\_validity(action)

r = (P\_after - P\_before) \\
    + eta\_done \* int(D\_after > tau\_D) \\
    - lambda\_side \* (S\_after - S\_before) \\
    - lambda\_invalid \* int(V == 0) \\
    - lambda\_step
```

建议初始超参：

* `eta\_done = 1.0`
* `lambda\_invalid = 1.0`
* `lambda\_step = 0.01`
* `lambda\_side = 0.5`
* `tau\_D = 0.9`
* `max\_segment\_len = 6 或 8`

### 7.5 Segment 内 process return

不用 low-level critic。计算：

```text
A\_t^L = sum\_{l=t}^{T\_seg-1} (γ\_L λ\_L)^{l-t} r\_l^L
```

代码：

```python
def compute\_process\_return(rewards, gamma=0.95, lam=0.9):
    returns = torch.zeros(len(rewards))
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards\[t] + gamma \* lam \* G
        returns\[t] = G
    return returns
```

然后 batch normalize：

```python
adv = (returns - returns.mean()) / (returns.std() + 1e-8)
```

### 7.6 Action token 更新

用 PPO-style loss，只更新 action tokens：

```python
ratio = exp(new\_action\_logprob - old\_action\_logprob)
loss\_action = -min(
    ratio \* adv,
    clip(ratio, 1-eps, 1+eps) \* adv
)
loss\_action = (loss\_action \* action\_mask).sum() / action\_mask.sum()
```

### 7.7 SWITCH 监督训练

同时训练 switch：

```text
target = "SWITCH" if done else "KEEP"
```

```python
loss\_switch = CE(switch\_logits, target)
loss\_switch = loss\_switch \* switch\_mask
```

早期不用 switch PPO，先用 supervised done target。

### 7.8 Phase 5 验收标准

Forced-subgoal evaluation：

* forced-subgoal success > SFT executor
* invalid action rate 下降
* average segment length 不爆炸
* switch F1 > 0.7
* PRM reward 与真实 subgoal success 正相关
* ALFWorld 六类任务的 forced-subgoal 都有提升或至少不退化
* WebShop forced-subgoal 能完成搜索、筛选、比较、选择等核心能力

如果 forced-subgoal 都学不好，不要进入高层 RL。

\---

## 8\. Phase 6：高层 Planner 训练

低层稳定后，冻结 executor，训练 planner。

### 8.1 高层 rollout

每个 episode：

```text
state s\_bk
planner generates subgoal g\_k
frozen executor executes g\_k until boundary
record macro transition:
(s\_bk, g\_k, R\_k^H, s\_b{k+1})
```

### 8.2 高层 reward

主方法使用保守 reward：

```text
R\_k^H = 1\_terminal R\_task - β C\_side - η C\_red
```

不要加入正的 executor-quality shaping。

terminal reward：

* success: +1
* failure: 0 或 -1

side penalty 使用 global side-effect：

* 目标物丢失
* 关键状态被破坏
* 走到明显错误流程
* WebShop 选择硬约束不匹配商品
* WebShop 过早购买错误商品

redundancy penalty：

```python
def redundancy\_penalty(g\_k, previous\_subgoals):
    if g\_k semantically similar to recent subgoal:
        return 0.2
    if planner repeats same completed subgoal:
        return 0.5
    return 0.0
```

早期可以先用字符串 / canonical subgoal type 判断，不必做复杂语义 embedding。

### 8.3 Macro advantage

早期先用固定 macro discount：

```text
δ\_k = R\_k^H + γ\_H V\_H(s\_b{k+1}) - V\_H(s\_bk)
A\_k^H = δ\_k + γ\_H λ\_H A\_{k+1}^H
```

代码：

```python
def compute\_macro\_gae(rewards, values, next\_values, gamma=0.95, lam=0.95):
    K = len(rewards)
    adv = torch.zeros(K)
    gae = 0.0
    for k in reversed(range(K)):
        delta = rewards\[k] + gamma \* next\_values\[k] - values\[k]
        gae = delta + gamma \* lam \* gae
        adv\[k] = gae
    return adv
```

Duration-aware discount 可作为 ablation，不要一开始放主线。

### 8.4 Planner token 更新

只更新 subgoal span：

```python
ratio = exp(new\_subgoal\_logprob - old\_subgoal\_logprob)
loss\_planner = -min(
    ratio \* adv\_H,
    clip(ratio, 1-eps, 1+eps) \* adv\_H
)
loss\_planner = (loss\_planner \* subgoal\_mask).sum() / subgoal\_mask.sum()
```

同时训练 macro critic：

```python
loss\_value = MSE(V\_H(s\_bk), target\_return)
```

### 8.5 Phase 6 验收标准

* final task success > Plan-Execute SFT
* wrong subgoal rate 下降
* redundant subgoal rate 下降
* avg macro steps 合理
* executor forced-subgoal skill 未明显下降
* ALFWorld 六类任务整体 success 提升
* WebShop task score / success 提升

如果 planner 生成大量 executor 没见过的 subgoal，则进入 executor refresh。

\---

## 9\. Phase 7：Executor Refresh / 慢速交替

这一步可选，但很可能有帮助。

### 9.1 Executor refresh

用当前 planner 生成 subgoal，冻结 planner，重新训练 executor 少量步数：

1. 收集 planner-generated subgoals
2. 过滤格式错误和明显不可执行 subgoal
3. 加入 low-level training distribution
4. 训练 executor 1–2 epoch

目的：缓解 planner 分布漂移。

### 9.2 慢速交替

如果 refresh 后稳定，可以交替：

1. planner 更新 3 个 epoch
2. executor 更新 1 个 epoch
3. 评估 validation
4. 如果 success 下降 > 5%，回退

不要一开始 joint training。

\---

## 10\. Phase 8：核心诊断实验

这是论文最重要的实验，不只是工程验证。

### 10.1 任务对设计

例子：

```text
Task A:
clean cup and put it in cabinet

错误子目标 X:
cool cup in fridge

Task B:
cool cup and put it in cabinet

正确子目标 X:
cool cup in fridge
```

WebShop 也应构造类似诊断对：

```text
Task A:
buy a red cotton shirt under $30

错误子目标 X:
select blue polyester shirt

Task B:
buy a blue polyester shirt under $30

正确子目标 X:
select blue polyester shirt
```

### 10.2 期望结果

HSPO 应该做到：

* Task A 中 `P\_high(select X)` 下降
* Task B 中 `P\_high(select X)` 保持或上升
* forced-X executor success 不下降

### 10.3 指标

* `P(X | Task A)`
* `P(X | Task B)`
* `ExecutorSuccess(X | forced X)`
* `SkillRetention(X)`
* `FinalSuccess(Task A)`
* `FinalSuccess(Task B)`

这个实验直接证明：

> 高层能抑制全局错误子目标，但低层不遗忘该子目标的执行技能。

\---

## 11\. Phase 9：主实验与 Ablation

### 11.1 Baselines

* Base Model
* ReAct
* Plan-Execute SFT
* ReAct + PPO
* Plan-Execute + Flat PPO
* HiPER-style baseline
* GiGPO-style baseline
* HSPO

HiPER 原论文在 ALFWorld 和 WebShop 上与 PPO、GRPO、GiGPO 等方法比较，并报告了 ALFWorld 六类任务和 WebShop 指标，这可以作为你实验设计的参考框架。

### 11.2 Ablation

必须做：

* HSPO full
* w/o PRM
* w/o process λ-return
* w/o token mask
* w/o side-effect
* w/o switch supervision
* joint training
* 

  * executor-quality shaping
* duration-aware macro discount

其中最关键是：

* w/o token mask
* w/o process λ-return
* w/o PRM

### 11.3 评估指标

任务层：

* success rate
* average episode length
* invalid action rate
* task score
* WebShop final score / purchase accuracy

子目标层：

* subgoal completion rate
* forced-subgoal success
* switch F1
* premature switch rate
* delayed switch rate
* average segment length

信用解耦层：

* `P\_wrong\_subgoal`
* `SkillRetention`
* `CreditConflictRate`

PRM 层：

* progress ranking accuracy
* done F1
* valid AUC
* side-effect recall

\---

## 12\. 推荐项目时间线

如果单人或小团队执行，建议按 10–12 周规划。

### Week 1：基础设施

任务：

* 跑通 ALFWorld 全量任务接口
* 跑通 WebShop 基础接口
* 实现 wrapper / parser / logger / token mask
* 跑 100 episode 日志

交付物：

* `raw trajectory logger`
* `parse success report`
* `mask unit test`

### Week 2：成功轨迹收集

任务：

* 实现 heuristic planner
* 收集 ALFWorld 六类任务成功轨迹
* 启动 WebShop 高分轨迹收集
* 过滤轨迹

交付物：

* `raw\_success.jsonl`
* `trajectory\_stats.json`

### Week 3：分段与 SFT 数据

任务：

* 规则分段
* LLM 改写 subgoal
* 生成 `sft\_format / sft\_executor / sft\_planner / sft\_all`
* 人工检查 200 条

交付物：

* `segmented\_success.jsonl`
* `sft\_\*.jsonl`
* `segmentation\_quality\_report`

### Week 4：SFT warm-up

任务：

* format SFT
* executor SFT
* planner SFT
* joint SFT

交付物：

* `sft\_checkpoint`
* `format\_valid\_rate`
* `forced-subgoal initial success`

### Week 5：Rule PRM

任务：

* 实现 ALFWorld progress / done / validity / side-effect
* 实现 WebShop progress / done / validity / side-effect
* 构造 negative perturbations
* PRM sanity test

交付物：

* `rule\_prm\_alfworld.py`
* `rule\_prm\_webshop.py`
* `prm\_validation\_set.jsonl`
* `prm\_eval\_report`

### Week 6–7：低层训练

任务：

* 冻结 planner
* 训练 executor
* 训练 switch CE
* forced-subgoal evaluation

交付物：

* `executor\_checkpoint`
* `low\_level\_eval\_report`

### Week 8–9：高层训练

任务：

* 冻结 executor
* 训练 macro planner
* macro critic
* 评估 final success

交付物：

* `planner\_checkpoint`
* `main\_eval\_dev\_report`

### Week 10：诊断实验

任务：

* 构造 Task A / Task B
* 评估 `P(X|A)`, `P(X|B)`, forced-X success
* 加入 WebShop 诊断对

交付物：

* `diagnostic\_report`
* `case\_studies`

### Week 11：Ablation

任务：

* 跑 w/o PRM
* 跑 w/o mask
* 跑 w/o λ-return
* 跑 w/o switch

交付物：

* `ablation\_table`
* `failure\_analysis`

### Week 12：扩展与论文图表

任务：

* ALFWorld 全量正式实验
* WebShop 正式实验
* 整理 curves / tables / cases

交付物：

* `main\_tables`
* `training\_curves`
* `qualitative\_cases`

\---

## 

