# V1.4 Algorithm Notes

V1.3 的目标是把 FT-oriented 主路径收口，而不是继续扩大算法面。下面这些方向适合进入 V1.4，但不建议在 V1.3 阶段继续混入主链路。

## 建议优先级

### 现在就适合预研，但先不要并入主路径

1. `prototype confidence / repair confidence`
   - 价值：可以给 Level 1 和 analysis 增加“修复可信度”指标。
   - 风险：主要是统计和阈值设计，工程风险低。
   - 建议：先做离线分析字段，不先改默认修复决策。

2. `group seed 搜索加速`
   - 价值：当前 FTScore / grouping 的 CPU 开销较高，真实实验会慢。
   - 风险：如果只做候选裁剪、缓存和向量化，行为风险较低。
   - 建议：优先做 profile，再决定是缓存 mask 候选还是减少相似度对比次数。

### 需要先有更强回归再考虑合入

3. `Level 1 多成员加权恢复`
   - 价值：可能优于当前 prototype-first / best-member 策略。
   - 风险：会直接改变修复行为，必须有更强 regression 和真实实验对照。
   - 建议：先作为可选策略，不替换默认行为。

4. `refresh 策略自适应`
   - 价值：可能提升动态 regroup 质量。
   - 风险：会影响训练稳定性和收敛路径。
   - 建议：先把 refresh decision 从固定 acceptance score 扩展成可配置策略，再做搜索。

5. `coverage-accuracy tradeoff 自动搜索`
   - 价值：对论文实验最有帮助，可以系统给出 Pareto 曲线。
   - 风险：计算成本高，容易把工程问题变成调参系统。
   - 建议：等 V1.3 的结果目录、summary 和 remote pipeline 稳定后再做。

## 不建议在 V1.3 继续做的事

- 不要现在就重写 `main.py` 的四段模型脚本。
- 不要在没有回归保护的情况下替换默认 Level 1 修复策略。
- 不要把 PRAP 兼容层一次性全部删除。

## V1.4 进入条件

在下面条件满足后，再开 V1.4 更稳妥：

- V1.3 的 regression 能稳定跑通。
- 至少有 `Res18` 和一个对照模型的真实结果 summary。
- `coverage_ratio_information.pkl` / `reuse_ratio_information.pkl` 的兼容层已经稳定。
