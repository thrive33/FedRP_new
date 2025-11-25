"""
实验结果可视化脚本
读取日志文件并生成对比图表
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_log_file(log_file='resnet18_cifar100_dynamic.log'):
    """
    解析日志文件,提取实验结果
    """
    results = defaultdict(lambda: {
        'epochs': [],
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': [],
        'dimensions': []
    })
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 匹配格式: Epoch X, client_count Y, Algo: AlgoName, data_type: IID/Non-IID: ...
                match = re.search(
                    r'Epoch (\d+).*Algo: ([^,]+).*data_type: ([^:]+):.*'
                    r'train_accuracy=([\d.]+).*test_accuracy=([\d.]+).*'
                    r'train_loss=([\d.]+).*test_loss=([\d.]+)',
                    line
                )
                
                if match:
                    epoch = int(match.group(1))
                    algo = match.group(2).strip()
                    data_type = match.group(3).strip()
                    train_acc = float(match.group(4))
                    test_acc = float(match.group(5))
                    train_loss = float(match.group(6))
                    test_loss = float(match.group(7))
                    
                    key = f"{algo} ({data_type})"
                    results[key]['epochs'].append(epoch)
                    results[key]['train_acc'].append(train_acc)
                    results[key]['test_acc'].append(test_acc)
                    results[key]['train_loss'].append(train_loss)
                    results[key]['test_loss'].append(test_loss)
                    
                    # 提取维度信息
                    dim_match = re.search(r'RP Dim: (\d+)', line)
                    if dim_match:
                        results[key]['dimensions'].append(int(dim_match.group(1)))
    
    except FileNotFoundError:
        print(f"警告: 找不到日志文件 {log_file}")
        print("请先运行实验: python resnet18_dynamic.py")
        return None
    
    return dict(results)

def plot_accuracy_comparison(results, data_type='IID', save_path='accuracy_comparison.png'):
    """
    绘制测试精度对比曲线
    """
    plt.figure(figsize=(12, 6))
    
    # 筛选指定数据类型的结果
    filtered_results = {k: v for k, v in results.items() if data_type in k}
    
    for algo_name, data in filtered_results.items():
        if data['test_acc']:
            # 移除括号中的数据类型标注
            label = algo_name.replace(f" ({data_type})", "")
            plt.plot(data['epochs'], data['test_acc'], marker='o', label=label, linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title(f'Test Accuracy Comparison ({data_type} Data)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {save_path}")
    plt.close()

def plot_dimension_history(results, save_path='dimension_history.png'):
    """
    绘制动态算法的维度变化历史
    """
    plt.figure(figsize=(12, 6))
    
    # 筛选有维度历史的算法
    dynamic_algos = {k: v for k, v in results.items() 
                     if v['dimensions'] and ('Linear' in k or 'Adaptive' in k)}
    
    for algo_name, data in dynamic_algos.items():
        if data['dimensions']:
            plt.plot(data['epochs'], data['dimensions'], marker='s', label=algo_name, linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Projection Dimension m(t)', fontsize=12)
    plt.title('Dynamic Projection Dimension Over Time', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {save_path}")
    plt.close()

def plot_communication_accuracy_tradeoff(results, data_type='IID', 
                                        save_path='comm_accuracy_tradeoff.png'):
    """
    绘制通信成本 vs 精度的权衡图
    需要从日志的Summary部分提取数据
    """
    # 这里需要手动提取或从日志Summary部分解析
    # 示例数据(需要根据实际运行结果更新)
    algorithms = []
    accuracies = []
    comm_costs = []
    
    # 从results中提取最终精度
    for algo_name, data in results.items():
        if data_type in algo_name and data['test_acc']:
            algorithms.append(algo_name.replace(f" ({data_type})", ""))
            accuracies.append(data['test_acc'][-1])  # 最后一轮的精度
            
            # 估算通信成本(需要从Summary中获取准确值)
            if 'm=10' in algo_name:
                comm_costs.append(1e6)
            elif 'm=100' in algo_name:
                comm_costs.append(1e7)
            elif 'm=1000' in algo_name:
                comm_costs.append(1e8)
            elif 'Linear' in algo_name:
                comm_costs.append(3.5e7)  # 估算值
            elif 'Adaptive' in algo_name:
                comm_costs.append(4.5e7)  # 估算值
            else:
                comm_costs.append(1.2e8)  # FedAvg等
    
    if not algorithms:
        print("警告: 没有足够的数据绘制通信成本-精度图")
        return
    
    plt.figure(figsize=(10, 6))
    
    # 使用不同颜色和标记
    colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms)))
    
    for i, (algo, acc, cost) in enumerate(zip(algorithms, accuracies, comm_costs)):
        plt.scatter(cost, acc, s=200, alpha=0.7, color=colors[i], label=algo, edgecolors='black')
    
    plt.xlabel('Total Communication Cost (parameters)', fontsize=12)
    plt.ylabel('Final Test Accuracy (%)', fontsize=12)
    plt.title(f'Communication Cost vs Accuracy Tradeoff ({data_type} Data)', 
              fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {save_path}")
    plt.close()

def plot_all_comparisons(results):
    """
    生成所有对比图表
    """
    print("\n生成可视化图表...")
    
    # 1. IID数据精度对比
    plot_accuracy_comparison(results, data_type='IID', 
                            save_path='plots/accuracy_comparison_iid.png')
    
    # 2. Non-IID数据精度对比
    plot_accuracy_comparison(results, data_type='Non-IID', 
                            save_path='plots/accuracy_comparison_noniid.png')
    
    # 3. 维度变化历史
    plot_dimension_history(results, save_path='plots/dimension_history.png')
    
    # 4. 通信成本-精度权衡
    plot_communication_accuracy_tradeoff(results, data_type='IID',
                                        save_path='plots/comm_accuracy_tradeoff.png')
    
    print("\n所有图表已生成完毕!")

def create_summary_table(results):
    """
    创建结果汇总表格
    """
    print("\n" + "="*100)
    print("实验结果汇总表")
    print("="*100)
    print(f"{'Algorithm':<30} {'Data Type':<10} {'Final Acc (%)':<15} {'Max Acc (%)':<15}")
    print("-"*100)
    
    for algo_name, data in sorted(results.items()):
        if data['test_acc']:
            algo_clean = algo_name.split('(')[0].strip()
            data_type = 'IID' if 'IID' in algo_name else 'Non-IID'
            final_acc = data['test_acc'][-1]
            max_acc = max(data['test_acc'])
            
            print(f"{algo_clean:<30} {data_type:<10} {final_acc:>13.2f}  {max_acc:>13.2f}")
    
    print("="*100)

if __name__ == '__main__':
    import os
    
    # 创建plots目录
    os.makedirs('plots', exist_ok=True)
    
    print("="*80)
    print("FedRP 动态投影实验结果可视化")
    print("="*80)
    
    # 解析日志文件
    results = parse_log_file('resnet18_cifar100_dynamic.log')
    
    if results is None or not results:
        print("\n错误: 无法读取实验结果")
        print("请确保已运行实验并生成日志文件")
        print("运行命令: python resnet18_dynamic.py")
    else:
        print(f"\n成功解析 {len(results)} 个实验结果")
        
        # 显示汇总表格
        create_summary_table(results)
        
        # 生成所有图表
        plot_all_comparisons(results)
        
        print("\n" + "="*80)
        print("可视化完成! 图表保存在 plots/ 目录下:")
        print("  - accuracy_comparison_iid.png")
        print("  - accuracy_comparison_noniid.png")
        print("  - dimension_history.png")
        print("  - comm_accuracy_tradeoff.png")
        print("="*80)
