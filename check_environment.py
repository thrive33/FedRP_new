"""
环境检查脚本
验证所有依赖是否正确安装
"""

import sys

def check_package(package_name, import_name=None):
    """检查单个包是否安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} - 已安装")
        return True
    except ImportError:
        print(f"❌ {package_name} - 未安装")
        return False

def check_cuda():
    """检查CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA - 可用 (设备: {torch.cuda.get_device_name(0)})")
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠️  CUDA - 不可用 (将使用CPU,训练速度较慢)")
            return False
    except:
        print("❌ 无法检查CUDA状态")
        return False

def check_data_directory():
    """检查数据目录"""
    import os
    data_dir = './data'
    if os.path.exists(data_dir):
        print(f"✅ 数据目录 - 存在 ({data_dir})")
        return True
    else:
        print(f"ℹ️  数据目录 - 不存在 (首次运行时会自动下载)")
        return True

def main():
    print("="*60)
    print("FedRP 动态投影 - 环境检查")
    print("="*60)
    print()
    
    print("📦 检查Python版本:")
    print(f"   Python {sys.version}")
    if sys.version_info < (3, 7):
        print("❌ Python版本过低,需要 >= 3.7")
        return False
    else:
        print("✅ Python版本符合要求")
    print()
    
    print("📦 检查依赖包:")
    packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('numpy', 'numpy'),
        ('tqdm', 'tqdm'),
        ('matplotlib', 'matplotlib'),
    ]
    
    all_installed = True
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            all_installed = False
    
    if not all_installed:
        print()
        print("⚠️  缺少依赖包,请运行:")
        print("   pip install -r requirements.txt")
        print()
        return False
    
    print()
    print("🔧 检查CUDA:")
    check_cuda()
    
    print()
    print("📁 检查数据目录:")
    check_data_directory()
    
    print()
    print("="*60)
    
    if all_installed:
        print("✅ 环境检查通过!")
        print()
        print("下一步:")
        print("1. 快速测试: python quick_test.py")
        print("2. 完整实验: python resnet18_dynamic.py")
        print("3. 查看文档: README_DYNAMIC.md")
    else:
        print("❌ 环境检查失败,请安装缺失的依赖")
    
    print("="*60)
    
    return all_installed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
