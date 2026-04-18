# 面向 Codex 的详细改造实施文档：将当前基于 PRAP 的源码改造成“以容错为导向的 OU 冗余组剪枝/映射框架”

> 适用对象：`https://github.com/a842404957/ft-test` 当前版本，以及你已接入的三级容错仿真代码。  
> 目标：不再以 PRAP 的“压缩/复用”叙事为中心，而是建立一个**以增加 OU 冗余度、提高 Level 1 可修复覆盖率为中心**的新剪枝/映射方法，并接入你现有的三级容错实验流程。  
> 本文档是写给 Codex 的执行说明书，要求按阶段逐步修改，**每阶段都必须通过验收后再进入下一阶段**。

---

## 0. 总目标与改造边界

### 0.1 总目标

把当前工程从：

- shape pruning + value similar translate
- 通过 `map_information` / `multiple_relationship_information` 表达“谁复用谁”
- 在容错阶段把“可复用 OU”解释为“冗余组”

改造成：

- **FT-oriented pruning**（容错导向剪枝）
- **FT-oriented grouping / mapping**（容错导向冗余组构建）
- 直接输出**group_information**，把“冗余组”作为一等公民
- 保持与现有三级容错入口兼容，同时逐步摆脱对 PRAP 文件命名和语义的依赖

### 0.2 非目标

本轮不要做下面这些事：

1. 不要重构模型定义文件（`model.py`）的网络结构。
2. 不要改 CIFAR10 数据流。
3. 不要在第一轮实现里引入过于复杂的聚类库依赖。
4. 不要一开始就推翻现有 `PatternDataLoader` / `FaultToleranceSimulator`，要先做兼容式修改。
5. 不要删除 PRAP 原有函数；先保留，新增 FT 分支。

### 0.3 最终要达到的结果

最终工程至少应支持一个新的 `translate_name`：

```python
ft_group_cluster_translate
```

它应能：

1. 生成新的 mask；
2. 生成新的 group / map / multiplier / coverage 文件；
3. 训练并导出：
   - `model_{model_name}_ft_group_cluster_translate_after_translate_parameters.pth`
4. 被 `run_hierarchical_fault_tolerance.py` 正确加载；
5. 在三级容错实验中：
   - Level 1 使用**显式冗余组**做替换；
   - Level 2 / Level 3 保持可用；
6. 产生一组新的指标，证明方法不是 PRAP 的简单改名版。

---

## 1. 先理解当前源码状态（Codex 必须先读懂再改）

### 1.1 当前训练主入口 `main.py` 的真实职责

当前 `main.py` 并不是一个简洁的“注册式入口”，而是一个**按模型写死的大脚本**。它负责：

1. 训练原始模型；
2. 为每层构造：
   - `mask`
   - `map_information`
   - `multiple_relationship_information`
   - `reuse_ratio_information`
3. 根据 `translate_name` 进入不同分支；
4. 最后统一调用 `pattern_translate(...)` 做第二阶段微调。

所以本次改造的核心不是“改一处参数”，而是要给它新增一条完整的 FT pipeline。

### 1.2 当前 `cut.py` 的真实职责

`cut.py` 目前混合承担了三类工作：

1. 生成 mask：
   - `get_structure_mask`
   - `get_ORC_mask`
   - `get_shape_mask`
2. 生成映射：
   - `pattern_value_identical_translate`
   - `pattern_value_similar_translate`
   - `structure_and_value_identical_translate`
   - `pattern_shape_and_value_similar_translate`
3. 训练：
   - `pattern_translate`

这意味着：

- 你要做 FT-oriented 方法，**最自然的落点仍是 `cut.py`**；
- 但要避免继续在旧函数上“魔改”，应新增一套 FT 函数。

### 1.3 当前三级容错入口的真实依赖关系

`run_hierarchical_fault_tolerance.py` 目前默认还是按 PRAP 的名字加载模型和映射文件，因此你训练出新方法后，仿真入口如果不改，仍然会去找旧文件。  
所以必须同步修改：

- `load_model(...)`
- `run_simulation_with_config(...)`
- CLI 参数

### 1.4 当前 FT-core 的真实瓶颈

当前 FT-core 里：

- `PatternDataLoader` 虽然已经尝试先读通用文件名，再 fallback 到 `shape_and_value_similar_*`；
- 但它还没有把 `group_information.pkl` 当成一等公民；
- `RedundancyGroupParser` 目前主要还是从 `map_information` 反推组；
- `FaultToleranceSimulator` 的 Level 1 恢复策略仍偏“从 map 推出组，再做替换”。

本次改造的策略是：

**先兼容，再转正。**

也就是：

- 第一轮先让新方法也输出 `map_information` 兼容旧解析器；
- 第二轮再让 `group_information` 成为主数据源。

---

## 2. 新方法命名、叙事和文件协议（必须先定下来）

### 2.1 方法命名

统一使用：

```python
translate_name = 'ft_group_cluster_translate'
```

不要再使用下面这些新旧混合名字：

- `weight_pattern_shape_and_value_similar_translate_ft`
- `fault_tolerance_shape_translate`
- `redundancy_oriented_prap`

原因：

- 名字里不要再带 `pattern reusing`；
- 直接突出 `ft` + `group` + `cluster` 三个核心概念；
- 后续写论文时可以扩展成完整名：
  - `OU-Group Regularized Pruning for Fault Tolerance`

### 2.2 新文件协议

Codex 必须统一输出以下文件名：

```text
model_{model_name}_{translate_name}_mask.pkl
model_{model_name}_{translate_name}_map_information.pkl
model_{model_name}_{translate_name}_multiple_relationship_information.pkl
model_{model_name}_{translate_name}_reuse_ratio_information.pkl
model_{model_name}_{translate_name}_group_information.pkl
model_{model_name}_{translate_name}_after_translate_parameters.pth
model_{model_name}_{translate_name}.csv
model_{model_name}_{translate_name}_train_info.csv
```

说明：

- `reuse_ratio_information.pkl` 这个名字虽然带有 reuse 语义，但**第一轮保留**，用于兼容旧代码；
- 实际语义改为：
  - `coverage_ratio_information`
  - 即“进入可修复组的 OU 比例”
- 第二轮可以再新增：

```text
model_{model_name}_{translate_name}_coverage_ratio_information.pkl
```

但不要在第一轮就删除旧名字。

### 2.3 group_information 的统一数据结构

Codex 必须新增一个明确的数据结构，建议如下：

```python
group_information = {
    layer_name: {
        'layer_name': layer_name,
        'group_count': int,
        'ou_count': int,
        'coverage_ratio': float,
        'groups': [
            {
                'group_id': int,
                'prototype': {
                    'out_ch': int,
                    'in_ch_start': int,
                    'channel_span': int,
                    'multiplier': 1.0,
                },
                'members': [
                    {
                        'out_ch': int,
                        'in_ch_start': int,
                        'channel_span': int,
                        'multiplier': float,
                        'similarity': float,
                        'role': 'prototype' | 'member'
                    },
                    ...
                ],
                'mask_signature': str,
                'group_size': int,
                'repair_mode': 'exact' | 'scaled'
            },
            ...
        ]
    },
    ...
}
```

要求：

1. `prototype` 一定要显式保存；
2. `members` 里要包含 prototype 自己；
3. `channel_span` 用于表示一个 OU 横跨多少连续输入通道；
4. `multiplier` 表示“成员如何由 prototype 缩放得到”；
5. `mask_signature` 用于后续快速检查组内 shape 是否一致；
6. `group_size` 必须冗余保存，避免每次重复算。

---

## 3. 先做最小侵入式改造总路线

Codex 必须严格按以下顺序执行：

### Phase A：框架解耦

1. 让 `main.py` 和 `run_hierarchical_fault_tolerance.py` 支持新 `translate_name`；
2. 不改变旧 PRAP 分支行为；
3. 确保新旧方法可以共存。

### Phase B：新增 FT mask + group builder

1. 在 `cut.py` 新增：
   - `ft_group_score_mask(...)`
   - `ft_group_cluster_translate(...)`
   - 若干 helper
2. 先只做“静态分组版”；
3. 先复用现有训练器也可以，但要尽快准备独立训练器。

### Phase C：新增 FT 训练器

1. 新增 `ft_group_translate_train(...)`；
2. 引入新的 group regularization loss；
3. 定期 refresh group。

### Phase D：容错链路转正

1. `PatternDataLoader` 支持 `group_information.pkl`；
2. `RedundancyGroupParser` 优先直接读 group；
3. `FaultToleranceSimulator` 的 Level 1 优先走显式 group。

### Phase E：分析与验收

1. 新增 FT 指标分析脚本；
2. 新增实验验收脚本或命令；
3. 完成完整回归测试。

---

## 4. Phase A：先改 `main.py`，把入口从“PRAP 单一路径”改成“可注册新方法”

> 这一阶段的目标不是做算法，而是把入口解耦。  
> 验收标准：只改框架，不改算法时，旧 PRAP 功能仍然能跑，新 FT 分支的空壳也能进入。

### 4.1 修改 import 区域

在 `main.py` 里现有的：

```python
from cut import pattern_translate, get_structure_mask, get_ORC_mask, get_shape_mask, pattern_value_identical_translate, pattern_value_similar_translate, structure_and_value_identical_translate, pattern_shape_and_value_similar_translate
```

改成扩展式导入：

```python
from cut import (
    pattern_translate,
    get_structure_mask,
    get_ORC_mask,
    get_shape_mask,
    pattern_value_identical_translate,
    pattern_value_similar_translate,
    structure_and_value_identical_translate,
    pattern_shape_and_value_similar_translate,
    ft_group_score_mask,
    ft_group_cluster_translate,
    ft_group_translate_train,
)
```

注意：

- 如果此时新函数还没写，先写占位实现；
- 占位实现必须 `raise NotImplementedError`，不要静默返回。

### 4.2 修改顶层 `translate_name`

把默认值从：

```python
translate_name = 'weight_pattern_shape_and_value_similar_translate'
```

改为：

```python
translate_name = 'ft_group_cluster_translate'
```

同时加注释：

```python
# 新默认方法：面向容错的OU分组剪枝/映射
```
建议在main.py里增加新分支：

```python
if translate_name == 'ft_group_cluster_translate':
    map_file = f'model_{model_name}_{translate_name}_map_information.pkl'
    mult_file = f'model_{model_name}_{translate_name}_multiple_relationship_information.pkl'
    cov_file = f'model_{model_name}_{translate_name}_reuse_ratio_information.pkl'   # 兼容旧接口，后续可改名
    mask_file = f'model_{model_name}_{translate_name}_mask.pkl'
    group_file = f'model_{model_name}_{translate_name}_group_information.pkl'

    if not os.path.exists(map_file):
        checkpoint = torch.load(
            f'model_{model_name}_original_parameter_epoch{translate_epoch[0]}_ckpt.pth'
        )
        model_original.load_state_dict(checkpoint['model'])

        group_information = {}

        for i, layer in enumerate(weight_name):
            mask[layer], group_information[layer] = ft_group_score_mask(
                model=model_original,
                weight_name=layer,
                in_channel=layer_in_channel[i],
                out_channel=layer_out_channel[i],
                kernel_size=kernel_size[i],
                channel_number=channel_number[i],
                pattern_value_number=pattern_value_number[i],
                pattern_shape_number=pattern_shape_number,
                ou_size=OU_size,
                target_group_size=target_group_size[i],
                sim_threshold=similarity_threshold[i],
            )

            map_information[layer], multiple_relationship_information[layer], \
            reuse_ratio_information[layer] = ft_group_cluster_translate(
                model=model_original,
                weight_name=layer,
                in_channel=layer_in_channel[i],
                out_channel=layer_out_channel[i],
                kernel_size=kernel_size[i],
                channel_number=channel_number[i],
                mask=mask[layer],
                min_group_size=min_group_size,
                target_group_size=target_group_size[i],
                sim_threshold=similarity_threshold[i],
                scale_candidates=scale_candidates,
            )

        pkl.dump(mask, open(mask_file, 'wb'))
        pkl.dump(map_information, open(map_file, 'wb'))
        pkl.dump(multiple_relationship_information, open(mult_file, 'wb'))
        pkl.dump(reuse_ratio_information, open(cov_file, 'wb'))
        pkl.dump(group_information, open(group_file, 'wb'))
    else:
        ...
    
    ft_group_translate_train(...)
```
    
这里最重要的是：

文件名改成通用的 {translate_name}_xxx.pkl

额外保存 group_information.pkl

训练函数单独起名，别继续沿用 pattern_translate 作为正式方法

### 4.3 新增 FT 参数区

在全局超参数区域新增：

```python
ft_min_group_size = 2
ft_target_group_size = 4
ft_similarity_threshold = 0.85
ft_exact_similarity_threshold = 0.98
ft_scale_candidates = [0.25, 0.5, 1.0, 2.0, 4.0]
ft_group_refresh_epoch = [150, 160, 170, 180, 190, 195, 200]
ft_balance_lambda = 1e-4
ft_proto_lambda = 1e-3
ft_mask_lambda = 1e-3
ft_sep_lambda = 5e-5
```

说明：

- 第一轮先给固定默认值；
- 第二轮再考虑按模型/按层配置。

### 4.4 不要继续把 `best_keep_ratio` 当成唯一核心变量

当前 `main.py` 中，`best_keep_ratio` 同时承担：

- 剪枝率
- 重用后保留率
- 某层是否参与 translate

这会让 FT 语义很混乱。

Codex 必须做的最小修改：

新增一个平行变量：

```python
ft_layer_enabled = [True] * len(weight_name)
ft_group_target_ratio = [0.75] * len(weight_name)
```

含义：

- `ft_layer_enabled[i]`：该层是否做 FT group 构建；
- `ft_group_target_ratio[i]`：该层希望进入组的 OU 比例目标；

第一轮里：

- `best_keep_ratio` 可以保留给旧 PRAP；
- FT 方法主要使用 `ft_group_target_ratio`。

### 4.5 给每个模型分支都加 FT 方法占位分支

在 `Vgg16` / `Res18` / `Res50` 三个大分支里，都要加：

```python
if translate_name == 'ft_group_cluster_translate':
    ...
```

注意：

- 三个模型分支都要加；
- 不允许只改 `Vgg16`。

### 4.6 FT 分支里先做统一初始化

参考旧逻辑，构造：

```python
value_list = [...]
mask = dict(zip(weight_name, value_list))

layer_map_list = [...]
map_information = dict(zip(weight_name, layer_map_list))

layer_multiple_list = [...]
multiple_relationship_information = dict(zip(weight_name, layer_multiple_list))

layer_reuse_ratio_list = [torch.zeros(1) for _ in range(len(weight_name))]
reuse_ratio_information = dict(zip(weight_name, layer_reuse_ratio_list))

group_information = {layer: None for layer in weight_name}
```

其中：

- conv 层和 fc 层维度必须保持和旧代码兼容；
- `multiple_relationship_information` 对 fc 层也要初始化成二维张量。

### 4.7 FT 分支的文件缓存逻辑

Codex 必须按下面模板写缓存逻辑：

```python
mask_file = f'model_{model_name}_{translate_name}_mask.pkl'
map_file = f'model_{model_name}_{translate_name}_map_information.pkl'
mult_file = f'model_{model_name}_{translate_name}_multiple_relationship_information.pkl'
reuse_file = f'model_{model_name}_{translate_name}_reuse_ratio_information.pkl'
group_file = f'model_{model_name}_{translate_name}_group_information.pkl'
```

如果这些文件不存在：

1. 加载 `translate_epoch[0]` 对应的 checkpoint；
2. 对每层先生成 FT mask；
3. 再生成 FT group translate；
4. 再保存 5 个文件。

如果这些文件存在：

1. 逐个读取；
2. 如果旧缓存缺少 `group_file`，要打印警告并退回 `map_information` 兼容模式；
3. 不要崩溃。

---

## 5. Phase B：在 `cut.py` 新增 FT 方法主体

> 这是本次改造的算法核心。  
> 目标是：在不改变硬件 OU 正确性的前提下，构建**可修复组**，而不是“可复用组”。

### 5.1 新增 helper：统一抽取 OU pattern

Codex 必须先新增一个帮助函数：

```python
def extract_ou_patterns(model, weight_name, in_channel, out_channel, kernel_size, channel_number, mask=None):
    ...
```

返回建议是一个列表，每个元素是：

```python
{
    'out_ch': int,
    'in_ch_start': int,
    'channel_span': int,
    'raw': tensor,
    'masked': tensor,
    'norm1': float,
    'norm2': float,
    'mask_signature': bytes,
}
```

要求：

1. conv / shortcut / fc 都能处理；
2. `in_ch_start` 表示这个 OU 所在输入通道块起点；
3. `channel_span` 就是 `channel_number`；
4. `masked` 必须在有 mask 时使用 `raw * mask`；
5. `mask_signature` 用于快速判断 shape 一致性。

### 5.2 新增 helper：离散幂次缩放搜索

新增：

```python
def search_best_power_of_two_scale(source_tensor, target_tensor, scale_candidates=None):
    ...
```

功能：

- 输入两个 pattern：
  - `source_tensor`
  - `target_tensor`
- 在候选缩放集合中找一个 `alpha`，使：

```python
|| source_tensor - alpha * target_tensor ||_1
```

最小。

返回：

```python
best_scale, best_error, best_similarity
```

要求：

1. 默认候选集合就是全局 `ft_scale_candidates`；
2. 若 `target_tensor` 全零：
   - 只允许把 `source_tensor` 映射到零组；
3. 相似度建议定义为：

```python
similarity = 1 - error / (||source_tensor||_1 + 1e-8)
```

### 5.3 新增 helper：组内原型选择

新增：

```python
def select_group_prototype(pattern_list):
    ...
```

建议规则：

- 在同一个候选组中选 **medoid**：
  - 与其他成员总 L1 距离最小；
- 如果实现复杂，可先选：
  - `norm1` 最大且组内平均相似度最高的成员。

返回 prototype 的 index。

### 5.4 新增 helper：平衡式聚类

新增：

```python
def build_ft_groups_for_block(pattern_list, min_group_size, target_group_size, sim_threshold, exact_threshold, scale_candidates):
    ...
```

这是最关键的函数之一。

输入：同一个 `in_ch block` 内的所有 OU pattern。  
输出：若干 group，每个 group 至少满足：

- 同一个 block 内；
- 同一 `mask_signature`；
- 组内成员能被某个 prototype 以 power-of-two scale 近似。

推荐流程：

#### Step 1：按 mask_signature 分桶

先只让 shape 一致的 pattern 进入同一个候选桶。

#### Step 2：在每个桶内按 norm1 降序

排序目的：

- 大权重 pattern 更适合当 prototype；
- 小权重 pattern 更适合成为成员。

#### Step 3：贪心建组

伪代码：

```python
for each bucket:
    unassigned = sorted_patterns
    while unassigned not empty:
        choose prototype from largest norm pattern
        create group = [prototype]
        for candidate in remaining patterns:
            scale, err, sim = search_best_power_of_two_scale(candidate, prototype)
            if sim >= sim_threshold:
                add to current group
            if len(group) == target_group_size:
                break
        remove group members from unassigned
```

#### Step 4：处理 singleton

如果某组只有 1 个成员：

- 先尝试合并到最近的已有 group；
- 条件：
  - same block
  - same mask_signature
  - similarity >= sim_threshold
- 如果仍失败，就保留 singleton，但记为不可 Level 1 修复。

#### Step 5：标记 repair_mode

- 如果所有 member 的最佳 scale 都是 1：
  - `repair_mode = 'exact'`
- 否则：
  - `repair_mode = 'scaled'`

### 5.5 新增核心函数 `ft_group_score_mask(...)`

新增：

```python
def ft_group_score_mask(model, weight_name, in_channel, out_channel, kernel_size,
                        channel_number, pattern_value_number, pattern_shape_number,
                        OU_size, target_group_size=4, sim_threshold=0.85):
    ...
```

作用：

- 不再只按“出现次数最多的 shape”选 mask；
- 改成优先选能形成更高 coverage 的 shape。

#### 5.5.1 推荐 FTScore

对于每个候选 shape `M`，定义：

```math
FTScore(M) = \lambda_1 C(M) + \lambda_2 S(M) + \lambda_3 G(M) - \lambda_4 E(M)
```

其中：

- `C(M)`：使用该 shape 后，能进入 size>=2 组的比例；
- `S(M)`：组内平均相似度；
- `G(M)`：平均组大小；
- `E(M)`：剪枝误差；

第一轮建议权重：

```python
lambda_1 = 1.0
lambda_2 = 0.5
lambda_3 = 0.25
lambda_4 = 0.25
```

#### 5.5.2 第一轮简化实现

为了降低实现风险，Codex 可以先做简化版：

- 在 `get_shape_mask(...)` 生成的 top-k shape 候选基础上；
- 对每个候选 shape 做一次 block 内 grouping 评估；
- 选 FTScore 最高的那个 shape 赋给该 block。

即：

- 允许先**复用 `get_shape_mask` 的候选生成方式**；
- 但最终选择依据改为 FTScore。

#### 5.5.3 返回值

```python
return pattern_mask, group_seed_info
```

其中 `group_seed_info` 可先保存：

```python
{
    'selected_shape_ids': ...,
    'estimated_coverage': ...,
    'estimated_avg_group_size': ...
}
```

### 5.6 新增核心函数 `ft_group_cluster_translate(...)`

新增：

```python
def ft_group_cluster_translate(model, in_channel, out_channel, weight_name,
                               kernel_size, channel_number, mask,
                               min_group_size=2, target_group_size=4,
                               sim_threshold=0.85, exact_threshold=0.98,
                               scale_candidates=None):
    ...
```

它必须输出：

```python
map_table, multiple_relationship_table, coverage_ratio, layer_group_information
```

#### 5.6.1 它的语义和 PRAP 的不同点

PRAP 的语义是：

- `source -> target` 表示 source 复用 target

现在改成：

- `source -> prototype` 表示 source 属于 prototype 所代表的冗余组

也就是说：

- `map_table` 只是兼容层；
- 真实语义以 `group_information` 为准。

#### 5.6.2 map_table 兼容写法

仍保持：

```python
map_table[in_ch][entry][0] = member_out_ch
map_table[in_ch][entry][1] = prototype_out_ch or -1
```

规则：

1. 如果 member 是 prototype 自己：
   - 不写入 mapping entry；
2. 只为非 prototype member 写 entry；
3. 末尾保留 `-1` 结束标记；
4. conv 和 fc 的写法要与现有 loader / parser 兼容。

#### 5.6.3 multiple_relationship_table 写法

对于每个 member：

```python
member ≈ alpha * prototype
```

则保存：

```python
multiple_relationship_table[member_out][member_in] = alpha
```

对 conv 层保持四维广播赋值，对 fc 层保持二维。

#### 5.6.4 coverage_ratio 定义

建议定义为：

```python
coverage_ratio = repairable_ou_count / total_ou_count
```

其中 `repairable_ou_count` 是所有属于 `size >= min_group_size` 组的 OU 数量。

#### 5.6.5 layer_group_information 组织形式

必须符合第 2.3 节协议。

---

## 6. Phase C：新增 FT 训练器 `ft_group_translate_train(...)`

> 这是让方法从“静态后处理”升级为“真正训练得到的 FT 模型”的关键。

### 6.1 为什么不能一直复用 `pattern_translate(...)`

因为 `pattern_translate(...)` 的设计目标是：

- shape regularization
- value reuse regularization

它的损失组织方式会把你的方法叙事重新拉回“重用/压缩”。  
所以必须复制一份训练器出来，逻辑独立。

### 6.2 新函数签名

新增：

```python
def ft_group_translate_train(model, model_name, translate_name,
                             weight_name, in_channel, out_channel, kernel_size,
                             ft_layer_enabled, mask, group_information,
                             map_information, multiple_relationship_information,
                             ft_mask_lambda, ft_proto_lambda,
                             ft_balance_lambda, ft_sep_lambda,
                             device, optimizer, scheduler,
                             train_loader, test_loader,
                             max_epoches, translate_epoch,
                             group_refresh_epoch):
    ...
```

### 6.3 新训练目标

建议总损失：

```math
L = L_{CE} + \lambda_m L_{mask} + \lambda_p L_{proto} + \lambda_b L_{balance} + \lambda_s L_{sep}
```

#### 6.3.1 Mask loss

```math
L_{mask} = \sum_i \|W_i - M_{g(i)} \odot W_i\|_2^2
```

含义：把每个 OU 拉回共享 shape。

#### 6.3.2 Prototype alignment loss

```math
L_{proto} = \sum_i \|M_{g(i)} \odot W_i - \alpha_i P_{g(i)}\|_2^2
```

含义：把同组成员拉向组原型。

#### 6.3.3 Balance loss

```math
L_{balance} = \sum_g \max(0, s_{min} - |G_g|)^2
```

含义：惩罚 singleton，鼓励组更稳。

#### 6.3.4 Separation loss

可做简化版：

```math
L_{sep} = - \sum_i cos(W_i, P_{g(i)})
```

第一轮先不显式加入“与其他组远离”的项，也可以。

### 6.4 每次 refresh group 的时机

在 `group_refresh_epoch` 中列出的 epoch 到来时：

1. 重新从当前模型权重生成 `mask`（可选，第一轮也可固定）；
2. 重新跑 `ft_group_cluster_translate(...)`；
3. 重新构造 prototype target；
4. 更新 `group_information` / `map_information` / `multiple_relationship_information`；
5. 继续训练。

### 6.5 训练结束时必须保存哪些文件

训练结束后必须保存：

1. `model_{model_name}_{translate_name}_after_translate_parameters.pth`
2. 最新的：
   - mask
   - map_information
   - multiple_relationship_information
   - reuse_ratio_information
   - group_information
3. 训练曲线 CSV

注意：

- 不允许只保存模型，不保存 group 文件；
- 否则仿真阶段会读不到最新冗余组。

---

## 7. Phase D：修改 `PatternDataLoader`，让 FT 数据成为一等公民

> 这一阶段要让仿真器首先理解“group”，而不只是“map”。

### 7.1 新增成员变量

在 `PatternDataLoader.__init__` 里新增：

```python
self.group_information = None
```

### 7.2 在 `load_all_data()` 里新增 group 文件读取

顺序建议改成：

1. `group_information`
2. `map_information`
3. `multiple_relationship_information`
4. `reuse_ratio_information`
5. `mask`

代码逻辑示意：

```python
self.group_information = self._load_pkl_file(
    f'model_{self.model_name}_{self.translate_name}_group_information.pkl'
)
```

### 7.3 新增 getter

新增：

```python
def get_layer_group_info(self, layer_name: str):
    if self.group_information and layer_name in self.group_information:
        return self.group_information[layer_name]
    return None
```

### 7.4 验证逻辑调整

`_validate_data()` 改成：

- `map_information` 或 `group_information` 至少有一个存在即可；
- 若两者都缺失，失败；
- 如果只有 group 而没有 map，也允许继续。

---

## 8. Phase E：修改 `RedundancyGroupParser`，优先直接解析 group_information

### 8.1 修改 `parse_layer(...)`

现有逻辑是：

- 从 `map_table` 推 `pattern_to_ous`
- 再构造组

新逻辑应改成：

```python
layer_group_info = self.data_loader.get_layer_group_info(layer_name)
if layer_group_info is not None:
    return self._parse_layer_from_group_info(layer_name, layer_group_info)
else:
    return self._parse_layer_from_map_info(layer_name)
```

### 8.2 新增 `_parse_layer_from_group_info(...)`

把 `group_information[layer_name]['groups']` 转成当前 `RedundancyGroup` 结构。

关键要求：

1. `group_id` 保持原值；
2. `pattern_id` 可以直接用 prototype 的 `(out_ch, in_ch_start)` 编码；
3. `ou_indices` 必须导入所有 member；
4. `multipliers` 必须与 member 一一对应；
5. prototype 的 multiplier 固定为 1.0。

### 8.3 旧 `_parse_layer_from_map_info(...)` 保留不删

保留原因：

- 兼容旧 PRAP 数据；
- 可以用来做 A/B 对比。

### 8.4 统计信息加两项

在 `_compute_statistics()` 输出时新增：

- `singleton_groups`
- `repairable_ou_ratio`

定义：

```python
repairable_ou_ratio = 所有 size>=2 组中的OU总数 / 全部OU总数
```

---

## 9. Phase F：修改 `FaultToleranceSimulator`，让 Level 1 真正以“冗余组”为核心

### 9.1 不要破坏 Level 2 / Level 3

Level 2 / 3 保持原样，优先只改 Level 1。

### 9.2 新增 Level 1 的候选选择策略

现在 Level 1 不能简单“组里随便找一个健康 OU 就替换”。

Codex 必须实现以下策略：

#### 优先级 1：prototype 优先

如果故障组内 prototype 未故障：

- 直接使用 prototype 恢复。

#### 优先级 2：最高相似度健康成员

如果 prototype 故障：

- 在健康成员里找与故障 OU 原始权重最接近的成员；
- 使用其 multiplier 恢复。

#### 优先级 3：组内加权平均恢复

如果同组多个健康成员都可用：

- 可选实现：按预存 similarity 加权平均；
- 第一轮也可只选 best match。

### 9.3 新增恢复公式

对故障 OU `i`，若选中参考成员 `j`，且两者关系为：

```math
W_i \approx \alpha_i P_g,
\quad
W_j \approx \alpha_j P_g
```

则：

```math
W_i \approx \frac{\alpha_i}{\alpha_j} W_j
```

在输出恢复时同样使用：

```math
y_i \approx \frac{\alpha_i}{\alpha_j} y_j
```

Codex 要把这个公式写成清晰的函数：

```python
def compute_member_to_member_scale(alpha_i, alpha_j, eps=1e-8):
    return alpha_i / (alpha_j + eps)
```

### 9.4 统计信息新增

在 metrics 里增加：

- `level1_prototype_repairs`
- `level1_member_repairs`
- `level1_weighted_repairs`
- `level1_exact_repairs`
- `level1_scaled_repairs`
- `level1_failed_singleton`

这样你后面写论文时可以说清楚：

- Level 1 纠正成功来自哪类组；
- 主要靠 exact 还是 scaled。

---

## 10. Phase G：修改 `run_hierarchical_fault_tolerance.py`

### 10.1 新增 CLI 参数 `--translate`

在 argparse 中新增：

```python
parser.add_argument('--translate', type=str,
                   default='ft_group_cluster_translate',
                   help='转换方法名称')
```

### 10.2 修改 `load_model(...)`

把函数默认值：

```python
def load_model(model_name, translate_name='weight_pattern_shape_and_value_similar_translate', ...)
```

改成：

```python
def load_model(model_name, translate_name='ft_group_cluster_translate', ...)
```

### 10.3 所有调用都传 `args.translate`

包括：

1. `load_model(...)`
2. `run_simulation_with_config(...)`
3. `FaultToleranceSimulator(...)`

### 10.4 compare 模式也必须可切换 translate

不要只改 single 模式。

`compare_strategies(...)` 也要加一个参数：

```python
def compare_strategies(config_file=None, num_samples=1000, translate_name='ft_group_cluster_translate'):
```

---

## 11. Phase H：新增一个 FT 分析脚本，别再复用 `computation_analyse.py`

### 11.1 新文件名

新增：

```text
fault_tolerance_analyse.py
```

### 11.2 指标要求

至少实现以下统计：

1. `group_coverage_ratio`
2. `repairable_ou_ratio`
3. `singleton_ratio`
4. `avg_group_size`
5. `max_group_size`
6. `exact_group_ratio`
7. `scaled_group_ratio`
8. `mean_multiplier_deviation`
9. `layerwise_coverage`
10. `layerwise_group_count`

### 11.3 不要再输出“logical improvement”当主指标

原因：

- FT 方法保留所有 OU 作为冗余组时，不再以跳过计算为目标；
- 继续拿 `logical_improvement` 当核心指标，会把方法叙事再次拉回压缩/复用。

---

## 12. 推荐的具体实现顺序（Codex 真正执行时要按这个顺序）

### Step 1：先加空壳函数

在 `cut.py` 里加：

- `ft_group_score_mask`
- `ft_group_cluster_translate`
- `ft_group_translate_train`
- `extract_ou_patterns`
- `search_best_power_of_two_scale`
- `select_group_prototype`
- `build_ft_groups_for_block`

先让它们可 import。

### Step 2：改 `main.py` 入口，让新方法可进入

先让下面命令至少能运行到“未实现函数”位置：

```bash
python main.py
```

若失败点进入新函数说明入口接通。

### Step 3：先做静态 `ft_group_cluster_translate`

- 不做训练；
- 从 checkpoint 直接生成：
  - mask
  - map
  - multiple
  - group

### Step 4：让 `PatternDataLoader` 读到新文件

执行一个最小测试脚本，确认 loader 能读到 group。

### Step 5：让 `RedundancyGroupParser` 优先走 group

确认 parser 输出的 group 数不为 0。

### Step 6：让 `run_hierarchical_fault_tolerance.py` 支持 `--translate`

确认脚本能加载：

```text
model_{model}_ft_group_cluster_translate_after_translate_parameters.pth
```

### Step 7：接入 Level 1 新策略

确认仿真中 Level 1 的统计会增加。

### Step 8：最后再写 `ft_group_translate_train`

把静态 grouping 升级为训练态 grouping。

---

## 13. 每一阶段的具体验收方法（必须逐条做）

---

### 验收 A：入口解耦成功

#### 操作

1. 把 `translate_name` 改为 `ft_group_cluster_translate`；
2. 运行：

```bash
python main.py
```

#### 预期结果

- 程序成功进入 FT 分支；
- 如果函数尚未实现，应明确报：

```text
NotImplementedError: ft_group_cluster_translate not implemented
```

#### 不通过条件

- 仍进入旧 `weight_pattern_shape_and_value_similar_translate`；
- import 报错；
- 没有进入 FT 分支。

---

### 验收 B：静态 group 文件生成成功

#### 操作

运行 `main.py` 直到 group builder 完成。

#### 必须生成的文件

```text
model_Vgg16_ft_group_cluster_translate_mask.pkl
model_Vgg16_ft_group_cluster_translate_map_information.pkl
model_Vgg16_ft_group_cluster_translate_multiple_relationship_information.pkl
model_Vgg16_ft_group_cluster_translate_reuse_ratio_information.pkl
model_Vgg16_ft_group_cluster_translate_group_information.pkl
```

#### 检查命令

```bash
python - <<'PY'
import pickle as pkl
files = [
    'model_Vgg16_ft_group_cluster_translate_mask.pkl',
    'model_Vgg16_ft_group_cluster_translate_map_information.pkl',
    'model_Vgg16_ft_group_cluster_translate_multiple_relationship_information.pkl',
    'model_Vgg16_ft_group_cluster_translate_reuse_ratio_information.pkl',
    'model_Vgg16_ft_group_cluster_translate_group_information.pkl',
]
for f in files:
    with open(f, 'rb') as fp:
        obj = pkl.load(fp)
    print(f, type(obj))
PY
```

#### 预期结果

- 五个文件都存在；
- 都能被 pickle 正常读取；
- `group_information` 是 dict，且至少包含一个 layer。

---

### 验收 C：group_information 结构正确

#### 检查命令

```bash
python - <<'PY'
import pickle as pkl
with open('model_Vgg16_ft_group_cluster_translate_group_information.pkl', 'rb') as f:
    info = pkl.load(f)
first_layer = list(info.keys())[0]
layer = info[first_layer]
print('layer_name =', first_layer)
print('keys =', layer.keys())
print('group_count =', layer['group_count'])
print('coverage_ratio =', layer['coverage_ratio'])
print('first_group_keys =', layer['groups'][0].keys() if layer['groups'] else None)
PY
```

#### 预期结果

至少出现：

- `group_count`
- `coverage_ratio`
- `groups`
- `prototype`
- `members`
- `group_size`

#### 不通过条件

- `groups` 缺失；
- member 没 multiplier；
- prototype 没保存。

---

### 验收 D：PatternDataLoader 正确加载 group 文件

#### 操作

写最小测试：

```bash
python - <<'PY'
from simulator.Fault_Tolerance.pattern_data_loader import PatternDataLoader
loader = PatternDataLoader(model_name='Vgg16', translate_name='ft_group_cluster_translate', data_dir='./')
ok = loader.load_all_data()
print('load_ok =', ok)
print('has_group =', loader.group_information is not None)
print('layers =', loader.get_all_layer_names()[:3])
PY
```

#### 预期结果

- `load_ok = True`
- `has_group = True`

---

### 验收 E：RedundancyGroupParser 优先走 group 信息

#### 操作

```bash
python - <<'PY'
from simulator.Fault_Tolerance.pattern_data_loader import PatternDataLoader
from simulator.Fault_Tolerance.redundancy_group_parser import RedundancyGroupParser
loader = PatternDataLoader(model_name='Vgg16', translate_name='ft_group_cluster_translate', data_dir='./')
loader.load_all_data()
parser = RedundancyGroupParser(loader)
ok = parser.parse_all_layers()
print('parse_ok =', ok)
print('total_groups =', parser.statistics['total_groups'])
print('avg_group_size =', parser.statistics['avg_group_size'])
print('repairable_ratio =', parser.statistics.get('repairable_ou_ratio'))
PY
```

#### 预期结果

- `parse_ok = True`
- `total_groups > 0`
- `avg_group_size >= 2`（至少在主要层上）

---

### 验收 F：训练器能导出 FT 模型权重

#### 操作

运行完整训练后检查：

```bash
ls model_Vgg16_ft_group_cluster_translate_after_translate_parameters.pth
```

#### 预期结果

文件存在，且：

```bash
python - <<'PY'
import torch
sd = torch.load('model_Vgg16_ft_group_cluster_translate_after_translate_parameters.pth', map_location='cpu')
print(type(sd), len(sd))
PY
```

输出应为 state_dict。

---

### 验收 G：容错脚本能正确加载 FT 模型

#### 操作

```bash
python run_hierarchical_fault_tolerance.py --mode single --model Vgg16 --translate ft_group_cluster_translate --samples 128
```

#### 预期结果

脚本不再去找：

```text
model_Vgg16_weight_pattern_shape_and_value_similar_translate_after_translate_parameters.pth
```

而是找：

```text
model_Vgg16_ft_group_cluster_translate_after_translate_parameters.pth
```

---

### 验收 H：Level 1 真实在用冗余组

#### 操作

运行单次仿真，打开 verbose。

#### 预期日志

应出现类似统计：

```text
level1_corrections > 0
level1_prototype_repairs > 0 或 level1_member_repairs > 0
```

#### 不通过条件

- Level 1 纠正永远为 0；
- 只有 Level 2 / Level 3 在起作用。

---

### 验收 I：核心科学指标方向正确

#### 你至少要比较以下三组结果

1. 原 PRAP 变形版本
2. 新 FT 静态组版本
3. 新 FT 训练版

#### 至少输出下列对比表

| 方法 | baseline acc | faulty acc | ft acc | level1 correction rate | repairable ou ratio | avg group size |
|---|---:|---:|---:|---:|---:|---:|

#### 预期方向

新 FT 方法不一定在“逻辑加速比”上最好，但应该在以下至少一项显著更好：

- `level1 correction rate`
- `repairable_ou_ratio`
- `avg_group_size`
- `singleton_ratio` 更低

---

## 14. 强约束：Codex 修改时绝对不要做的事

1. 不要删除旧 PRAP 函数；
2. 不要把新方法继续写回旧文件名；
3. 不要把 `group_information` 只存在内存里不落盘；
4. 不要在 `PatternDataLoader` 里写死只能读 PRAP 文件；
5. 不要让 `run_hierarchical_fault_tolerance.py` 仍然默默使用旧 `translate_name`；
6. 不要把 Level 1 的恢复实现成“随机拿一个组员替换”；
7. 不要只支持 Vgg16。

---

## 15. 推荐的最小可行实现（MVP）

如果时间紧，Codex 应优先完成以下 MVP：

### MVP-1

- `main.py` 支持 `ft_group_cluster_translate`
- `cut.py` 新增：
  - `ft_group_score_mask`（可先简化）
  - `ft_group_cluster_translate`
- 生成新文件：
  - mask / map / multiple / reuse / group

### MVP-2

- `PatternDataLoader` 读取 group
- `RedundancyGroupParser` 直接解析 group
- `run_hierarchical_fault_tolerance.py` 支持 `--translate`

### MVP-3

- Level 1 使用 prototype-first 策略
- 跑三级容错实验

### MVP-4

- 最后再实现 `ft_group_translate_train`

---

## 16. 新方法的通俗定义（给后续论文写作使用）

你现在要做的方法，不再是：

> 先找谁能复用谁，然后把这些可复用 OU 当成冗余。

而是：

> 先主动把 OU 剪枝并组织成“可互相替代的组”，再把这些组直接作为硬件容错的冗余资源。

这两者的区别非常关键：

- PRAP 是“复用优先，容错附带”；
- 你的新方法是“容错优先，复用只是副现象”。

---

## 17. 新方法的数学定义（建议稿）

### 17.1 OU pattern 表示

记第 `i` 个 OU 的原始权重为：

```math
W_i
```

其共享形状掩码为：

```math
M_{g(i)}
```

剪枝后模式为：

```math
\widetilde{W}_i = M_{g(i)} \odot W_i
```

### 17.2 组原型关系

若 `i` 属于组 `g`，组原型为 `P_g`，则希望：

```math
\widetilde{W}_i \approx \alpha_i P_g
```

其中 `\alpha_i` 取自有限的 power-of-two 候选集合：

```math
\alpha_i \in \{2^k \mid k \in \mathcal{K}\}
```

### 17.3 最佳缩放选择

对候选原型 `P_g`，最优缩放由：

```math
\alpha_i^* = \arg\min_{\alpha \in \mathcal{A}} \|\widetilde{W}_i - \alpha P_g\|_1
```

给出。

### 17.4 组构建目标

希望最大化：

```math
\max \; \sum_g |G_g|
```

同时满足：

```math
\text{sim}(\widetilde{W}_i, \alpha_i P_g) \ge \tau
```

并保证同组成员具有一致的 mask signature。

### 17.5 训练目标

```math
L = L_{CE} + \lambda_m L_{mask} + \lambda_p L_{proto} + \lambda_b L_{balance} + \lambda_s L_{sep}
```

其中：

```math
L_{mask} = \sum_i \|W_i - M_{g(i)} \odot W_i\|_2^2
```

```math
L_{proto} = \sum_i \|M_{g(i)} \odot W_i - \alpha_i P_{g(i)}\|_2^2
```

```math
L_{balance} = \sum_g \max(0, s_{min} - |G_g|)^2
```

---

## 18. 交付要求（Codex 最终必须提交什么）

Codex 最终至少要提交：

1. 修改后的 `main.py`
2. 修改后的 `cut.py`
3. 修改后的 `run_hierarchical_fault_tolerance.py`
4. 修改后的：
   - `pattern_data_loader.py`
   - `redundancy_group_parser.py`
   - `fault_tolerance_simulation.py`
5. 新增 `fault_tolerance_analyse.py`
6. 若需要，新增一个：
   - `ft_method_config.py`
   - 或 `layer_spec.py`
7. 一份简短 changelog，说明：
   - 新增函数
   - 新文件协议
   - 兼容策略

---

## 19. 最终验收结论模板（完成后必须人工填写）

请在完成代码后，人工填写下面表格：

| 检查项 | 结果 | 备注 |
|---|---|---|
| main.py 支持 ft_group_cluster_translate |  |  |
| cut.py 新增 ft_group_score_mask |  |  |
| cut.py 新增 ft_group_cluster_translate |  |  |
| cut.py 新增 ft_group_translate_train |  |  |
| group_information.pkl 成功生成 |  |  |
| PatternDataLoader 成功读取 group_information |  |  |
| RedundancyGroupParser 优先解析 group_information |  |  |
| run_hierarchical_fault_tolerance.py 支持 --translate |  |  |
| Level 1 使用显式冗余组 |  |  |
| FT 模型权重成功导出 |  |  |
| 三级容错实验成功运行 |  |  |
| 新 FT 指标分析脚本可运行 |  |  |

---

## 20. 一句话总结

本次改造的本质不是“把 PRAP 的 reuse map 换个解释”，而是：

> **把 OU 的组织目标从“可跳过计算”彻底改成“可被冗余修复”，再用显式 group 信息驱动三级容错。**

