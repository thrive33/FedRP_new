# FedRP 动态投影实验运行指南

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 或手动安装
pip install torch torchvision numpy tqdm matplotlib
```

### 2. 快速测试(5分钟)

运行简化版实验,验证代码是否正常工作:

```bash
python quick_test.py
```

这将运行4个算法,每个10轮,5个客户端。预期输出:

```
快速测试结果对比
================================================================================
Algorithm                 Accuracy     Comm Cost       Time (s)
--------------------------------------------------------------------------------
FedAvg                       25.34%      6.15e+07         120.5
FedRP (m=10)                 18.67%      6.15e+05          95.3
FedRP_Linear                 23.45%      1.84e+07         105.8
FedRP_Adaptive               24.12%      2.15e+07         110.2
```

### 3. 完整实验(2-4小时)

运行所有对比实验:

```bash
python resnet18_dynamic.py
```

这将运行:
- 8个算法在IID数据上(30轮,10个客户端)
- 4个算法在Non-IID数据上

**预计时间**: 
- GPU (RTX 3090): ~2小时
- GPU (GTX 1080): ~4小时
- CPU: ~12小时(不推荐)

### 4. 结果可视化

实验完成后,生成图表:

```bash
python visualize_results.py
```

生成的图表:
- `plots/accuracy_comparison_iid.png` - IID数据精度对比
- `plots/accuracy_comparison_noniid.png` - Non-IID数据精度对比
- `plots/dimension_history.png` - 维度变化轨迹
- `plots/comm_accuracy_tradeoff.png` - 通信-精度权衡图

## 详细配置

### 修改实验参数

编辑 `resnet18_dynamic.py` 中的 `Arguments` 类:

```python
class Arguments:
    def __init__(self):
        # === 基础训练参数 ===
        self.batch_size = 64          # 训练批次大小
        self.test_batch_size = 16     # 测试批次大小
        self.epochs = 30              # 通信轮数
        self.lr = 0.1                 # 学习率
        self.client_count = 10        # 客户端数量
        self.E = 1                    # 本地训练轮数
        self.alpha = 1.0              # ADMM惩罚参数
        
        # === 动态投影参数 ===
        # 线性增长策略
        self.rp_dim_min = 10          # 最小投影维度
        self.rp_dim_max = 1000        # 最大投影维度
        self.rp_growth_rate = 5       # 每轮增长量
        
        # 自适应调整策略
        self.adaptive_threshold_high = 0.5   # 高变化阈值
        self.adaptive_threshold_low = 0.1    # 低变化阈值
        self.adaptive_increment = 50         # 维度增量
        self.adaptive_decrement = 20         # 维度减量
```

### 参数调优建议

#### 线性增长率 (rp_growth_rate)

| 值 | 效果 | 适用场景 |
|----|------|---------|
| 1-2 | 缓慢增长,通信成本低,精度可能略低 | 通信受限环境 |
| 5-10 | 平衡增长(推荐) | 一般场景 |
| 20-50 | 快速增长,接近固定大维度 | 追求高精度 |

#### 自适应阈值

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| threshold_low | 0.05-0.15 | 太低:频繁增加维度; 太高:增加太慢 |
| threshold_high | 0.3-0.7 | 太低:频繁减少维度; 太高:很少减少 |
| increment | 30-100 | 维度增加的步长 |
| decrement | 10-30 | 维度减少的步长(通常 < increment) |

#### ADMM惩罚参数 (alpha)

| 值 | 效果 |
|----|------|
| 0.1-0.5 | 弱惩罚,可能收敛慢 |
| 1.0 | 默认值(推荐) |
| 2.0-5.0 | 强惩罚,收敛快但可能震荡 |

### 自定义实验

#### 只运行特定算法

编辑 `resnet18_dynamic.py` 的主函数:

```python
if __name__ == '__main__':
    train_data, test_data = get_datasets()
    
    # 只运行你感兴趣的实验
    results = []
    
    # 例如:只对比FedAvg和两个动态策略
    results.append(run_experiment(FedAvg, train_data, test_data, args))
    
    results.append(run_experiment(
        FedRP_Linear, train_data, test_data, args,
        algorithm_name="FedRP_Linear",
        alpha=args.alpha,
        rp_dim_min=args.rp_dim_min,
        rp_dim_max=args.rp_dim_max,
        growth_rate=args.rp_growth_rate
    ))
    
    results.append(run_experiment(
        FedRP_Adaptive, train_data, test_data, args,
        algorithm_name="FedRP_Adaptive",
        alpha=args.alpha,
        rp_dim_min=args.rp_dim_min,
        rp_dim_max=args.rp_dim_max,
        threshold_high=args.adaptive_threshold_high,
        threshold_low=args.adaptive_threshold_low,
        increment=args.adaptive_increment,
        decrement=args.adaptive_decrement
    ))
```

#### 添加新的动态策略

创建新的策略类:

```python
class FedRP_Exponential(DynamicFedRP):
    """指数增长策略"""
    def __init__(self, Model, device, client_count, optimizer, criterion, 
                 alpha, rp_dim_min, rp_dim_max, base=1.1):
        super().__init__(Model, device, client_count, optimizer, criterion,
                        alpha, rp_dim_min, rp_dim_max)
        self.base = base
    
    def _update_projection_dimension(self, epoch):
        """指数增长: m(t) = min(m_min * base^t, m_max)"""
        new_dim = int(min(self.rp_dim_min * (self.base ** epoch), self.rp_dim_max))
        return new_dim

# 运行实验
results.append(run_experiment(
    FedRP_Exponential, train_data, test_data, args,
    algorithm_name="FedRP_Exponential",
    alpha=args.alpha,
    rp_dim_min=10,
    rp_dim_max=1000,
    base=1.15
))
```

## 实验检查清单

### 实验前

- [ ] GPU可用且有足够内存(建议 ≥8GB)
- [ ] 数据集已下载(首次运行会自动下载CIFAR-100)
- [ ] 磁盘空间充足(≥5GB)
- [ ] 已安装所有依赖包

### 实验中

- [ ] 监控GPU使用率(`nvidia-smi`)
- [ ] 检查日志文件是否正常写入
- [ ] 观察训练精度是否合理(不应为0或100%)
- [ ] 注意内存使用,避免OOM

### 实验后

- [ ] 检查日志文件完整性
- [ ] 运行可视化脚本生成图表
- [ ] 备份结果文件
- [ ] 记录关键发现

## 常见问题

### Q1: CUDA out of memory

**解决方法**:
1. 减小 `batch_size`(如从64改为32)
2. 减少 `client_count`(如从10改为5)
3. 使用CPU运行(慢但稳定)

```python
# 修改为CPU模式
device = torch.device("cpu")
```

### Q2: 训练精度不增长

**可能原因**:
1. 学习率过大或过小 → 调整 `args.lr`
2. 投影维度太小 → 增大 `rp_dim_min`
3. ADMM惩罚参数不当 → 调整 `args.alpha`

### Q3: 实验时间太长

**加速方法**:
1. 减少训练轮数:`args.epochs = 10`
2. 减少客户端数:`args.client_count = 5`
3. 使用更快的GPU
4. 运行 `quick_test.py` 而不是完整实验

### Q4: 如何复现论文结果

```python
# 使用以下配置
args.epochs = 30
args.client_count = 10
args.lr = 0.1
args.batch_size = 64
args.E = 1
args.alpha = 1.0

# 线性策略
args.rp_dim_min = 10
args.rp_dim_max = 1000
args.rp_growth_rate = 5

# 自适应策略
args.adaptive_threshold_low = 0.1
args.adaptive_threshold_high = 0.5
args.adaptive_increment = 50
args.adaptive_decrement = 20
```

### Q5: Non-IID数据效果不好

**调整建议**:
1. 增大Dirichlet参数alpha(降低异构性)
2. 使用更大的投影维度
3. 增加本地训练轮数 `args.E = 2`
4. 使用梯度校正方法(需要额外实现)

## 实验记录模板

建议创建一个实验日志,记录每次运行的配置和结果:

```markdown
## 实验 #1 - 基准对比
- 日期: 2025-11-24
- 配置:
  - epochs: 30
  - client_count: 10
  - rp_dim_min: 10
  - rp_dim_max: 1000
  - growth_rate: 5
- 结果:
  - FedAvg: 45.23% (通信: 1.23e8)
  - FedRP_Linear: 44.78% (通信: 3.45e7, 降低72%)
  - FedRP_Adaptive: 45.01% (通信: 4.21e7, 降低66%)
- 发现:
  - 线性策略通信成本最低
  - 自适应策略精度更接近FedAvg
  - 维度在第15轮后稳定
```

## 性能基准

### 硬件参考

| 硬件配置 | 单轮耗时 | 30轮总时间 |
|---------|---------|-----------|
| RTX 4090 | ~2分钟 | ~1小时 |
| RTX 3090 | ~3分钟 | ~1.5小时 |
| RTX 2080 Ti | ~4分钟 | ~2小时 |
| GTX 1080 | ~6分钟 | ~3小时 |
| CPU (i9-12900K) | ~25分钟 | ~12.5小时 |

### 内存需求

| 配置 | GPU内存 | 系统内存 |
|------|---------|---------|
| 默认(10客户端) | ~6GB | ~8GB |
| 大规模(20客户端) | ~10GB | ~16GB |
| 小规模(5客户端) | ~4GB | ~6GB |

## 下一步

实验完成后:

1. **分析结果**: 查看日志和图表,总结关键发现
2. **撰写论文**: 使用 `THEORY.md` 中的理论框架
3. **进一步改进**: 尝试新的动态策略或结合其他技术
4. **投稿准备**: 整理代码,撰写README,准备开源

## 支持

如遇问题:
1. 检查日志文件 `resnet18_cifar100_dynamic.log`
2. 参考理论文档 `THEORY.md`
3. 查看主README `README_DYNAMIC.md`
4. 提交Issue或联系作者

---

**祝实验顺利!** 🚀
