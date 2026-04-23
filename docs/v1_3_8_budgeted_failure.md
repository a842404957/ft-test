# V1.3.8-b Budgeted Failure Diagnosis

## 背景配置

V1.3.8-b 的 `Res18 build-only` 主要使用：

- `--ft-grouping-mode budgeted`
- `--ft-budget-bucket-mode nonzero_count`
- `--ft-budget-mask-family shape_seed,shared_topk,per_out_topk`
- `--ft-budget-mask-keep-ratios 0.6667,0.4444`
- `--ft-budget-max-scale-error 0.25`

## 已知失败现象

典型层级日志表现为：

- `conv2 coverage=0.0269 singleton=0.9928`
- `conv3 coverage=0.1150 singleton=0.9819`
- `conv4 coverage=0.0803 singleton=0.9764`
- `conv5 coverage=0.2234 singleton=0.9334`
- `conv8 coverage=0.0897 singleton=0.9782`

整体上仍然表现为：

- repairable coverage 很低
- singleton ratio 很高
- 只有极少数超宽层（如 `conv17`）出现明显冗余

## 为什么 `shared_topk_0.4444` 仍然失败

`shared_topk_0.4444` 虽然比 dense mask 更稀疏，但它只是“更少非零”，并不等价于“更多共享 shape”。

当前 budgeted 路径里：

1. 稀疏 mask family 只是在 seed 阶段提供有限候选。
2. 真正 grouping 时仍然先按桶分开，再在桶里做 prototype assignment。
3. 如果一个层的 shape 本身很碎，mask 稀疏一些并不会自动让 bucket 变得可复用。

因此很多层只是把原本 dense 的 singleton，变成 sparse 的 singleton。

## 为什么 `nonzero_count bucket` 不等于真正共享 shape

`nonzero_count` 只保证“保留元素个数相同”，并不保证：

- 非零位置一致
- 方向相似
- 可以被同一个 prototype 用离散 scale 有效逼近

所以它能扩大 bucket，但也会把本来不可替换的 OU 放进同一个候选池，最后大量 OU 仍然因为 assignment error 过大而退化为 singleton。

## 为什么 threshold-based assignment 不是真正 budgeted grouping

V1.3.8-b 的核心仍然是：

- 先选 prototype
- 再根据 `assignment_error <= max_scale_error` 或 relax 后阈值决定是否加入 group

这本质上还是“threshold-based acceptance”，而不是“强制制造冗余”。

只要 assignment error 超阈值，OU 仍然被留成 singleton。  
这意味着 prototype budget 并没有真正转化为 coverage guarantee。

## 为什么停止继续搜索 V1.3.8-b 参数

当前问题已经不是某个单一超参数不合适，而是方法结构本身仍然允许：

- 无限/过碎的 mask 变体
- threshold 拒收
- singleton 作为默认退路

继续搜索：

- `prototype_budget_ratio`
- `bucket_mode`
- `max_scale_error`

只能做局部修修补补，不能从机制上保证 repairable coverage。

## 转向 V1.3.9 的原因

V1.3.9 转向：

- **有限 mask codebook**
- **强制 prototype assignment**
- **projection accuracy 验证**

核心目的不是继续“发现自然相似 OU”，而是：

1. 先用有限 codebook 压缩 shape 自由度，减少 mask fragmentation。
2. 再在 codebook 内做 deterministic prototype selection。
3. 用 forced assignment 把 coverage 目标变成约束，而不是仅仅统计结果。
4. 用 projected model accuracy 检查“主动制造的冗余”是否仍在可接受误差预算内。

因此，V1.3.8-b 在这里固定为失败诊断基线，不再继续做参数搜索。
