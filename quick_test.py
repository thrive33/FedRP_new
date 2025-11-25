"""
快速实验脚本 - 用于测试动态FedRP算法
包含较少的训练轮次,方便快速验证
"""

import sys
sys.path.append('.')

from resnet18_dynamic import *

class QuickTestArguments:
    """快速测试的参数配置"""
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 16
        self.epochs = 30              # 减少到10轮用于快速测试
        self.lr = 0.1
        self.client_count = 5         # 减少到5个客户端
        self.E = 1
        self.alpha = 1.0
        self.rp_dim = 10
        
        # 动态投影参数
        self.rp_dim_min = 10
        self.rp_dim_max = 200         # 减少最大维度用于快速测试
        self.rp_growth_rate = 10      # 增加增长率,快速看到效果
        
        # 自适应参数
        self.adaptive_threshold_high = 0.5
        self.adaptive_threshold_low = 0.1
        self.adaptive_increment = 30
        self.adaptive_decrement = 10

if __name__ == '__main__':
    print("="*80)
    print("快速测试: 动态FedRP算法")
    print("="*80)
    print("注意: 本脚本使用较少轮次和客户端,仅用于验证代码正确性")
    print("完整实验请运行: python resnet18_dynamic.py")
    print("="*80)
    
    # 使用快速测试参数
    args = QuickTestArguments()
    
    # 准备数据
    train_data, test_data = get_datasets()
    
    results = []
    
    # 测试1: FedAvg基线
    print("\n>>> 测试 1/4: FedAvg")
    results.append(run_experiment(
        FedAvg, train_data, test_data, args,
        algorithm_name="FedAvg"
    ))
    
    # 测试2: 原始FedRP (固定维度)
    print("\n>>> 测试 2/4: FedRP (m=10)")
    results.append(run_experiment(
        FedRP, train_data, test_data, args,
        algorithm_name="FedRP (m=10)",
        alpha=args.alpha, rp_dim=10
    ))
    
    # 测试3: FedRP_Linear (线性增长)
    print("\n>>> 测试 3/4: FedRP_Linear")
    results.append(run_experiment(
        FedRP_Linear, train_data, test_data, args,
        algorithm_name="FedRP_Linear",
        alpha=args.alpha,
        rp_dim_min=args.rp_dim_min,
        rp_dim_max=args.rp_dim_max,
        growth_rate=args.rp_growth_rate
    ))
    
    # 测试4: FedRP_Adaptive (自适应调整)
    print("\n>>> 测试 4/4: FedRP_Adaptive")
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
    
    # 打印对比结果
    print("\n" + "="*80)
    print("快速测试结果对比")
    print("="*80)
    print(f"{'Algorithm':<25} {'Accuracy':<12} {'Comm Cost':<15} {'Time (s)':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['algorithm']:<25} {r['final_accuracy']:>10.2f}% "
              f"{r['total_comm_cost']:>14.2e} {r['training_time']:>9.1f}")
    
    print("\n" + "="*80)
    print("关键观察:")
    print("="*80)
    
    # 通信成本对比
    fedavg_cost = results[0]['total_comm_cost']
    for r in results[1:]:
        reduction = (1 - r['total_comm_cost'] / fedavg_cost) * 100
        print(f"{r['algorithm']}: 通信成本降低 {reduction:.1f}%")
    
    # 精度对比
    print("\n精度对比 (相对FedAvg):")
    fedavg_acc = results[0]['final_accuracy']
    for r in results[1:]:
        diff = r['final_accuracy'] - fedavg_acc
        print(f"{r['algorithm']}: {diff:+.2f}%")
    
    # 维度历史
    print("\n维度变化历史:")
    for r in results:
        if r['dimension_history']:
            print(f"{r['algorithm']}: {r['dimension_history']}")
    
    print("\n" + "="*80)
    print("快速测试完成!")
    print("建议:")
    print("1. 如果代码运行正常,可以运行完整实验: python resnet18_dynamic.py")
    print("2. 可以在Arguments中调整参数,观察不同配置的效果")
    print("3. 完整实验结果会保存在 resnet18_cifar100_dynamic.log")
    print("="*80)
