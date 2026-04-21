# ft-test V1.3.3

V1.3.3 是 low-cost FT profiling 版：当前推荐主路径是 `ft_group_cluster_translate`，目标是把低成本 FT 训练做成可量化、可比较、可调参、可复现的研究原型。

## 环境前提

- Python 环境需要至少安装 `torch` 和 `pandas`
- CIFAR-10 数据目录默认是 `./cifar10_data`
- 若做长时间训练，建议在 GPU 机器上执行同一套命令
- 如果当前 shell 不是装好依赖的环境，请把下面的 `python` 替换成 `conda run -n <env> python`
- 在 `aris-gpu` 上，当前建议使用 `conda run -n ming python`

## 推荐工作流

1. 本地 smoke / regression

```bash
python scripts/regression_check.py
```

2. 训练 / 转换

```bash
python main.py --model Vgg16 --translate ft_group_cluster_translate
```

如果只是做快速探索，优先用 build-only：

```bash
python main.py \
  --model Res18 \
  --translate ft_group_cluster_translate \
  --build-only
```

如果要忽略已有缓存、强制重新构建 artifacts：

```bash
python main.py \
  --model Res18 \
  --translate ft_group_cluster_translate \
  --build-only \
  --force-rebuild
```

FT 主路径现在支持：

- `--build-only`：只构建 FT grouping artifacts 和投影后的 `after_translate_parameters.pth`
- `--force-rebuild`：忽略已有 FT artifacts 缓存，强制从 `--base-checkpoint-epoch` 对应 checkpoint 重新构建
- `--ft-cost-preset {none,fast,balanced,full}`：低成本训练 preset
- `--ft-low-cost`：启用低成本 FT 训练 preset；默认会把 FT 训练终点压到 `160`，并保持至少一次 refresh
- `--ft-end-epoch`：控制 FT 训练终点；默认 `200`，`--ft-low-cost` 默认 `160`
- `--ft-reg-interval`：每 N 个 batch 才计算一次 FT 正则；默认 `1`，`--ft-low-cost` 默认 `10`
- `--ft-reg-min-coverage`：只对 coverage 不低于该阈值的层计算 FT 正则；默认 `0.0`，`--ft-low-cost` 默认 `0.1`
- `--ft-reg-min-groups`：只对 repairable group 数不少于该阈值的层计算 FT 正则；默认 `1`，`--ft-low-cost` 默认 `64`
- `--ft-reg-boost-after-refresh`：refresh epoch 和 refresh 后 1 个 epoch 内，把有效正则间隔减半
- `--translate-epochs`：控制 FT 训练期间做 before/after translate 评估的 epoch，默认 `200`
- `--refresh-epochs`：控制动态重分组 refresh 节点，默认 `190,200`
- `--base-checkpoint-epoch`：控制 FT 构建/训练的起始原始 checkpoint，默认 `150`

cost preset 建议：

- `fast`：探索性验证，默认等价于 `--ft-low-cost`
- `balanced`：适合 `Res18` / `WRN` 的初步真实实验
- `full`：保留更多正则和 refresh，适合最终实验前复核

探索性 FT 训练建议先用 `fast`：

```bash
python main.py \
  --model Res18 \
  --translate ft_group_cluster_translate \
  --ft-low-cost
```

`balanced` 示例：

```bash
python main.py \
  --model Res18 \
  --translate ft_group_cluster_translate \
  --ft-cost-preset balanced
```

3. 单次三级容错

```bash
python run_hierarchical_fault_tolerance.py \
  --mode single \
  --model Vgg16 \
  --translate ft_group_cluster_translate \
  --config fault_tolerance_config_low_fault_rate.json \
  --samples 256 \
  --output-dir results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_demo/sim
```

4. 策略对比

```bash
python run_hierarchical_fault_tolerance.py \
  --mode compare \
  --model Vgg16 \
  --translate ft_group_cluster_translate \
  --config fault_tolerance_config_high_fault_rate.json \
  --samples 256 \
  --output-dir results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_demo/sim
```

5. FT 分析

```bash
python fault_tolerance_analyse.py \
  --model Vgg16 \
  --translate ft_group_cluster_translate \
  --output-json results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_demo/analysis/ft_report.json \
  --output-csv results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_demo/analysis/ft_layers.csv
```

6. 结果收集

```bash
python scripts/collect_ft_results.py \
  --model Vgg16 \
  --translate ft_group_cluster_translate \
  --config fault_tolerance_config_high_fault_rate.json \
  --samples 256 \
  --report-dir results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_demo/sim \
  --results-root results/ft_runs \
  --tag vgg16_demo
```

## 本地 smoke 与真实实验

- `python scripts/regression_check.py` 是本地 smoke，用 synthetic case 验证 block-aware 提取、parser、simulator、analyse 和 PRAP fallback。
- `python scripts/regression_check.py` 现在也覆盖 low-cost FT 的 preset、regularization layer report、training profile 和 refresh cache rebuild smoke。
- `python main.py ...` 之后的训练 / single / compare / analyse / collect 是真实实验闭环。
- 如果先做探索性实验，建议先执行 `python main.py --model Res18 --translate ft_group_cluster_translate --build-only`，确认 artifacts、仿真和分析链都通，再决定是否跑完整 FT 微调。
- 若在远端 GPU 上运行，只需要在远端仓库根目录执行同一套命令；`collect` 也应在同一仓库根目录执行。

## 真实实验操作

下面给出一个可直接照搬的 `Vgg16` 闭环命令序列。假设本轮实验 tag 是 `vgg16_real`：

```bash
python main.py --model Vgg16 --translate ft_group_cluster_translate

python run_hierarchical_fault_tolerance.py \
  --mode single \
  --model Vgg16 \
  --translate ft_group_cluster_translate \
  --config fault_tolerance_config_low_fault_rate.json \
  --samples 256 \
  --output-dir results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_real/sim

python run_hierarchical_fault_tolerance.py \
  --mode compare \
  --model Vgg16 \
  --translate ft_group_cluster_translate \
  --config fault_tolerance_config_high_fault_rate.json \
  --samples 256 \
  --output-dir results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_real/sim

python fault_tolerance_analyse.py \
  --model Vgg16 \
  --translate ft_group_cluster_translate \
  --output-json results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_real/analysis/ft_report.json \
  --output-csv results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_real/analysis/ft_layers.csv

python scripts/collect_ft_results.py \
  --model Vgg16 \
  --translate ft_group_cluster_translate \
  --config fault_tolerance_config_high_fault_rate.json \
  --samples 256 \
  --report-dir results/ft_runs/Vgg16/ft_group_cluster_translate/vgg16_real/sim \
  --results-root results/ft_runs \
  --tag vgg16_real
```

如果你是在远端 GPU 上跑，只需要把上面这组命令原样放到远端仓库根目录执行；区别只是执行机器不同，不需要改脚本入口。

## 结果目录约定

推荐把同一轮实验统一收在：

```text
results/ft_runs/<model>/<translate>/<tag>/
```

其中推荐使用：

- `.../sim/` 保存 single / compare 的仿真报告
- `.../analysis/` 保存 `fault_tolerance_analyse.py` 输出
- `.../summary.*` 和 `run_metadata.json` 由 `scripts/collect_ft_results.py` 生成

## Low-cost FT 报告

FT 训练现在会额外生成：

- `model_{model}_{translate}_training_profile.csv`
- `model_{model}_{translate}_regularization_layers.csv`

其中：

- `training_profile.csv` 用来看每个 epoch 的训练代价、正则代价、refresh 代价和 projection 代价
- `regularization_layers.csv` 用来看哪些层真正参与了 FT 正则，哪些层因为 coverage 太低或 repairable group 太少而被跳过

建议优先查看：

- `reg_batch_count` / `effective_reg_interval`
- `active_reg_layers`
- `skipped_low_coverage_layers`
- `skipped_small_group_layers`
- `epoch_time_sec` / `reg_time_sec` / `refresh_time_sec`

## 产物协议

`ft_group_cluster_translate` 主路径会生成以下产物：

- `model_{model_name}_{translate_name}_mask.pkl`
- `model_{model_name}_{translate_name}_map_information.pkl`
- `model_{model_name}_{translate_name}_multiple_relationship_information.pkl`
- `model_{model_name}_{translate_name}_group_information.pkl`
- `model_{model_name}_{translate_name}_coverage_ratio_information.pkl`
- `model_{model_name}_{translate_name}_reuse_ratio_information.pkl`
- `model_{model_name}_{translate_name}_after_translate_parameters.pth`
- `model_{model_name}_{translate_name}_refresh_log.csv`
- `model_{model_name}_{translate_name}_training_profile.csv`
- `model_{model_name}_{translate_name}_regularization_layers.csv`

其中：

- `group_information.pkl` 是 FT 主协议
- `coverage_ratio_information.pkl` 是正式统计命名
- `reuse_ratio_information.pkl` 仅作为兼容别名保留
- `map_information.pkl` / `multiple_relationship_information.pkl` 继续作为旧路径 fallback

## 示例配置

仓库当前提供两个示例配置：

- `fault_tolerance_config_low_fault_rate.json`
- `fault_tolerance_config_high_fault_rate.json`

## 兼容说明

- 旧 PRAP 路径和 `weight_pattern_shape_and_value_similar_translate` 仍保留在代码里，用于兼容旧产物。
- 新文档、脚本和推荐命令统一以 `ft_group_cluster_translate` 为主路径。

## 推荐命令

1. build-only

```bash
python main.py \
  --model Res18 \
  --translate ft_group_cluster_translate \
  --build-only
```

2. fast

```bash
python main.py \
  --model Res18 \
  --translate ft_group_cluster_translate \
  --ft-low-cost
```

3. balanced

```bash
python main.py \
  --model Res18 \
  --translate ft_group_cluster_translate \
  --ft-cost-preset balanced \
  --ft-reg-boost-after-refresh
```

## 相关文档

- 协议说明：[docs/ft_artifact_protocol.md](/Volumes/980PRO/ft-test/docs/ft_artifact_protocol.md)
- V1.4 预案：[docs/v1_4_algorithm_notes.md](/Volumes/980PRO/ft-test/docs/v1_4_algorithm_notes.md)
