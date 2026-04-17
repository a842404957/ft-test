# PRAP-PIM 容错机制框架

基于权重模式冗余的PIM硬件容错框架，将PRAP-PIM的"效率优化"机制转换为"冗余容错"机制。

## 📋 概述

本框架实现了以下核心功能：

1. **冗余组识别** - 从权重模式映射中识别冗余计算组
2. **故障注入** - 可配置的故障注入机制（支持多种故障模型）
3. **多数表决** - 基于多数表决的故障检测和纠正
4. **性能评估** - 完整的可靠性和硬件开销指标
5. **报告生成** - 多格式报告输出（JSON、CSV、Markdown）

## 🏗️ 架构设计

```
simulator/Fault_Tolerance/
├── config.py                      # 全局配置管理
├── pattern_data_loader.py         # 权重模式数据加载
├── redundancy_group_parser.py     # 冗余组解析
├── fault_injector.py              # 故障注入器
├── majority_voter.py              # 多数表决器
├── metrics_collector.py           # 评估指标收集
├── report_generator.py            # 报告生成器
├── fault_tolerance_simulation.py  # 主仿真流程（核心）
└── README.md                      # 本文档
```

### 核心概念转换

**原概念（效率驱动）**：
- 如果OU1, OU4, OU5, OU8使用相同权重模式
- 只激活OU1计算，结果复用到其他OU
- 目标：节省能耗和延迟

**新概念（可靠性驱动）**：
- 如果OU1, OU4, OU5, OU8使用相同权重模式
- 所有OU并行计算（冗余计算组）
- 通过多数表决检测和纠正故障
- 目标：提高系统可靠性

## 🚀 快速开始

### 1. 准备数据

首先运行`main.py`生成权重模式复用数据：

```bash
python main.py
```

这将生成以下.pkl文件：
- `model_Vgg16_shape_and_value_similar_map_information.pkl`
- `model_Vgg16_shape_and_value_multiple_relationship_information.pkl`
- `model_Vgg16_shape_and_value_reuse_ratio_information.pkl`
- `model_Vgg16_pattern_mask.pkl`

### 2. 运行容错仿真

使用提供的运行脚本：

```bash
python run_fault_tolerance_simulation.py \
    --model Vgg16 \
    --fault-rate 0.001 \
    --num-samples 1000
```

### 3. 查看结果

仿真完成后，在`fault_tolerance_results/`目录下查看生成的报告：
- `fault_tolerance_report_YYYYMMDD_HHMMSS.json` - 详细数据
- `fault_tolerance_summary_YYYYMMDD_HHMMSS.csv` - 摘要表格
- `fault_tolerance_report_YYYYMMDD_HHMMSS.md` - 格式化报告

## 🔧 配置说明

### 故障注入配置

```json
{
  "fault_injection": {
    "enabled": true,
    "fault_rate": 0.001,           // 故障率（0.1%）
    "fault_models": ["bit_flip"],  // 故障模型
    "random_seed": 42              // 随机种子
  }
}
```

支持的故障模型：
- `bit_flip` - 单比特翻转
- `output_corruption` - 输出损坏
- `stuck_at_zero` - 固定为0
- `stuck_at_one` - 固定为1
- `random_value` - 随机值
- `amplification` - 放大错误

### 多数表决器配置

```json
{
  "majority_voter": {
    "enabled": true,
    "voting_strategy": "simple_majority",  // 表决策略
    "tie_breaking": "detection_failure",   // 平局处理
    "voter_latency_ns": 10,                // 表决器延迟（纳秒）
    "voter_energy_pj": 50                  // 表决器能耗（皮焦）
  }
}
```

表决策略：
- `simple_majority` - 简单多数表决
- `weighted_majority` - 加权多数表决
- `exact_match` - 精确匹配表决

### 冗余组配置

```json
{
  "redundancy_group": {
    "min_group_size": 2,           // 最小组大小
    "max_group_size": 8,           // 最大组大小
    "grouping_strategy": "pattern_based"  // 分组策略
  }
}
```

## 📊 评估指标

### 可靠性指标

- **Total Faults Injected** - 总故障注入数
- **Faults Corrected** - 成功纠正数
- **Faults Missed** - 未纠正数
- **Fault Correction Rate** - 故障纠正率
- **Detection Failures** - 检测失败数

### 硬件开销指标

- **Total Latency** - 总延迟（包含表决器）
- **Total Energy** - 总能耗（包含表决器）
- **Voter Overhead** - 表决器开销
- **Redundancy Overhead Ratio** - 冗余开销比

### 准确率指标

- **Baseline Accuracy** - 基线准确率（无故障）
- **Faulty Accuracy** - 故障准确率（无容错）
- **FT Accuracy** - 容错准确率（有容错）
- **Accuracy Recovery Rate** - 准确率恢复率

## 🧪 使用示例

### 示例1：基本仿真

```python
from simulator.Fault_Tolerance import FaultToleranceSimulator
from model import Vgg16

# 加载模型
model = Vgg16(num_classes=10)
model.load_state_dict(torch.load('model_Vgg16_..._parameters.pth'))

# 创建仿真器
simulator = FaultToleranceSimulator(
    model=model,
    model_name='Vgg16',
    translate_name='weight_pattern_shape_and_value_similar_translate'
)

# 运行仿真
results = simulator.run_simulation(test_loader, num_samples=1000)
```

### 示例2：自定义配置

```python
# 创建配置文件
config = {
    'fault_injection': {
        'fault_rate': 0.005,  # 0.5%故障率
        'fault_models': ['bit_flip', 'output_corruption']
    },
    'majority_voter': {
        'voting_strategy': 'weighted_majority'
    }
}

# 使用自定义配置
simulator = FaultToleranceSimulator(
    model=model,
    model_name='Vgg16',
    config_file='my_config.json'
)
```

### 示例3：对比不同配置

```python
# 测试不同故障率
fault_rates = [0.0001, 0.001, 0.01]
results = []

for rate in fault_rates:
    simulator.config.set('fault_injection', 'fault_rate', rate)
    result = simulator.run_simulation(test_loader, num_samples=1000)
    results.append(result)

# 生成对比报告
simulator.report_generator.generate_comparison_report(
    results, 
    labels=[f'Fault Rate {r}' for r in fault_rates]
)
```

## 📈 预期结果

在Vgg16模型上，典型的仿真结果：

| 指标 | 值 |
|------|-----|
| 基线准确率 | 92% |
| 故障准确率（0.1%故障率） | 75-80% |
| 容错准确率 | 88-91% |
| 故障纠正率 | 80-90% |
| 延迟开销 | 2-3x |
| 能耗开销 | 2.5-3.5x |

## 🛠️ 命令行参数

```bash
python run_fault_tolerance_simulation.py --help

参数说明:
  --model MODEL          模型名称 (Vgg16, Res18, Res50, WRN)
  --translate TRANSLATE  转换方法名称
  --model-path PATH      模型文件路径
  --num-samples N        测试样本数 (-1表示全部)
  --batch-size N         批大小
  --fault-rate RATE      故障率
  --config-file FILE     配置文件路径
  --data-dir DIR         数据文件目录
  --output-dir DIR       输出目录
  --device DEVICE        计算设备 (auto, cuda, cpu)
```

## 🔍 模块详解

### 1. PatternDataLoader
负责加载和解析权重模式复用数据：
- 读取.pkl文件
- 提供统一的查询接口
- 支持多种模型架构

### 2. RedundancyGroupParser
从映射信息中识别冗余计算组：
- 解析map_information
- 按pattern_id分组OU
- 计算冗余统计

### 3. FaultInjector
实现可配置的故障注入：
- 多种故障模型
- 可控故障率
- 故障统计

### 4. MajorityVoter
实现多数表决机制：
- 多种表决策略
- 平局处理
- 硬件开销建模

### 5. MetricsCollector
收集和计算评估指标：
- 可靠性指标
- 硬件开销
- 准确率统计
- 逐层分析

### 6. ReportGenerator
生成多格式报告：
- JSON（详细数据）
- CSV（表格数据）
- Markdown（可读报告）

## 📝 注意事项

1. **数据准备**：确保先运行`main.py`生成必要的.pkl文件
2. **模型路径**：根据实际情况调整模型文件路径
3. **样本数量**：初次测试建议使用较小的样本数（如1000）
4. **故障率**：建议从0.001开始，逐步增加
5. **设备选择**：GPU可显著加速仿真过程

## 🐛 故障排查

### 问题1：找不到.pkl文件
```
解决：运行 python main.py 生成数据文件
```

### 问题2：模型加载失败
```
解决：检查模型文件路径是否正确
```

### 问题3：CUDA内存不足
```
解决：减小batch_size或num_samples，或使用CPU
```

### 问题4：冗余组数量为0
```
解决：检查translate_name是否与训练时一致
```

## 🔄 扩展开发

### 添加新的故障模型

在`fault_injector.py`中添加：

```python
def _new_fault_model(self, output: torch.Tensor) -> torch.Tensor:
    """新的故障模型"""
    # 实现故障注入逻辑
    return faulty_output
```

### 添加新的表决策略

在`majority_voter.py`中添加：

```python
def _new_voting_strategy(self, outputs: List[torch.Tensor]) -> Tuple:
    """新的表决策略"""
    # 实现表决逻辑
    return corrected_output, correction_made, details
```

## 📚 参考文献

1. PRAP-PIM原论文
2. PIM容错机制相关研究
3. 多数表决算法

## 👥 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

与PRAP-PIM项目保持一致

