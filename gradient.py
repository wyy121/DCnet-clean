# gradient_check_fixed_v2.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from model_channel_tau import Conv2dEIRNN
import os
import yaml
import argparse
import gc


def convert_legacy_tau_shapes(state_dict):
    """将旧版逐神经元tau参数压缩为新版逐channel tau参数。"""
    converted = {}
    for key, value in state_dict.items():
        if key.endswith(("tau_pyr", "tau_inter")) and value.ndim == 4:
            if value.shape[-2:] != (1, 1):
                converted[key] = value.mean(dim=(-2, -1), keepdim=True)
                continue
        converted[key] = value
    return converted


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_checkpoint(model, checkpoint_path, strict=False):
    """加载检查点"""
    print(f"正在加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_state = convert_legacy_tau_shapes(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"Checkpoint epoch: {epoch}")
    
    # 尝试加载模型
    try:
        if hasattr(model, '_orig_mod'):
            model._orig_mod.load_state_dict(checkpoint_state, strict=strict)
        else:
            model.load_state_dict(checkpoint_state, strict=strict)
        print(f"成功加载模型参数 (strict={strict})")
    except Exception as e:
        print(f"加载模型参数失败: {e}")
        print("尝试不严格加载...")
        try:
            if hasattr(model, '_orig_mod'):
                model._orig_mod.load_state_dict(checkpoint_state, strict=False)
            else:
                model.load_state_dict(checkpoint_state, strict=False)
            print("不严格加载成功")
        except Exception as e2:
            print(f"不严格加载也失败: {e2}")
            print("使用随机初始化的模型")
    
    return epoch, checkpoint.get('history', None)

def create_test_inputs(config, batch_size=2, device='cpu'):
    """创建测试输入"""
    model_config = config.get('model', {})
    
    # 获取输入维度
    input_size = model_config.get('input_size', (224, 224))
    input_dim = model_config.get('input_dim', 3)
    
    # 获取类别数
    num_classes = model_config.get('num_classes', 6)
    
    # 创建测试数据
    cue = torch.randn(batch_size, input_dim, *input_size).to(device)
    mixture = torch.randn(batch_size, input_dim, *input_size).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    print(f"测试数据形状:")
    print(f"  cue: {cue.shape}")
    print(f"  mixture: {mixture.shape}")
    print(f"  labels: {labels.shape}, 标签范围: [{labels.min().item()}, {labels.max().item()}]")
    
    return cue, mixture, labels

def check_gradients(model, cue, mixture, labels, device='cpu'):
    """检查模型各层的梯度"""
    model = model.to(device)
    model.train()
    
    cue = cue.to(device)
    mixture = mixture.to(device)
    labels = labels.to(device)
    
    print(f"\n前向传播...")
    # 前向传播
    outputs = model(cue, mixture)
    print(f"输出形状: {outputs.shape}, 输出范围: [{outputs.min().item():.6f}, {outputs.max().item():.6f}]")
    
    # 检查输出和标签维度
    if outputs.shape[1] != labels.max().item() + 1:
        print(f"⚠️  警告: 输出维度({outputs.shape[1]})与标签范围({labels.max().item()+1})不匹配")
    
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    print(f"损失值: {loss.item():.6f}")
    
    # 清理CUDA缓存（如果使用GPU）
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    print(f"反向传播...")
    # 反向传播
    model.zero_grad()
    loss.backward()
    
    # 收集梯度信息
    grad_info = {
        'mean_grad_norm': [],
        'max_grad_norm': [],
        'zero_grad_ratio': [],
        'layer_names': []
    }
    
    print("\n" + "="*80)
    print("梯度检查报告")
    print("="*80)
    
    modulation_grad_count = 0
    total_grad_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_grad_count += 1
            if param.grad is not None:
                grad = param.grad.data
                grad_norm = grad.norm().item()
                max_grad = grad.abs().max().item()
                zero_ratio = (grad.abs() < 1e-8).float().mean().item()
                
                grad_info['mean_grad_norm'].append(grad_norm)
                grad_info['max_grad_norm'].append(max_grad)
                grad_info['zero_grad_ratio'].append(zero_ratio)
                grad_info['layer_names'].append(name)
                
                # 统计调制模块
                if 'modulation' in name:
                    modulation_grad_count += 1
                    print(f"{name[:50]:50s} | 梯度范数: {grad_norm:8.6f} | "
                          f"最大梯度: {max_grad:8.6f} | 零梯度比例: {zero_ratio:6.2%}")
            elif 'modulation' in name:
                print(f"{name[:50]:50s} | 无梯度")
    
    print(f"\n梯度统计:")
    print(f"总参数层数: {total_grad_count}")
    print(f"有梯度的层数: {len(grad_info['mean_grad_norm'])}")
    print(f"调制模块参数层数: {modulation_grad_count}")
    
    # 整体统计
    if grad_info['mean_grad_norm']:
        print("\n" + "-"*80)
        print(f"梯度统计:")
        print(f"平均梯度范数: {np.mean(grad_info['mean_grad_norm']):.6f}")
        print(f"最大梯度范数: {np.max(grad_info['mean_grad_norm']):.6f}")
        print(f"最小梯度范数: {np.min(grad_info['mean_grad_norm']):.6f}")
        print(f"零梯度比例均值: {np.mean(grad_info['zero_grad_ratio']):.2%}")
    
    return grad_info

def test_forward_pass(model, config, device='cpu'):
    """测试前向传播"""
    print("\n" + "="*80)
    print("前向传播测试")
    print("="*80)
    
    # 创建测试数据
    cue, mixture, labels = create_test_inputs(config, batch_size=2, device=device)
    
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        # 基本前向传播
        outputs = model(cue, mixture)
        print(f"模型输出形状: {outputs.shape}")
        print(f"模型输出范围: [{outputs.min().item():.6f}, {outputs.max().item():.6f}]")
        
        # 检查是否有NaN或Inf
        if torch.isnan(outputs).any():
            print("⚠️  警告：输出包含NaN!")
        if torch.isinf(outputs).any():
            print("⚠️  警告：输出包含Inf!")
        
        # 测试单个调制模块
        print("\n调制模块详细测试:")
        if hasattr(model, 'modulations') and len(model.modulations) > 0:
            for i, mod in enumerate(model.modulations):
                if mod is not None:
                    # 创建测试输入
                    if hasattr(model, 'h_pyr_dims') and i < len(model.h_pyr_dims):
                        in_channels = model.h_pyr_dims[i]
                        if model.modulation_on == "hidden":
                            spatial_size = model.input_sizes[i]
                        else:
                            spatial_size = model.output_sizes[i]
                    else:
                        in_channels = 32
                        spatial_size = (28, 28)
                    
                    test_cue = torch.randn(2, in_channels, *spatial_size).to(device)
                    test_mixture = torch.randn(2, in_channels, *spatial_size).to(device)
                    
                    # 前向传播
                    output = mod(test_cue, test_mixture)
                    
                    # 检查调制效果
                    modulation_effect = output / (test_mixture + 1e-8)
                    
                    print(f"  调制层 {i}:")
                    print(f"    输入cue: {test_cue.shape}, 输入mixture: {test_mixture.shape}")
                    print(f"    输出形状: {output.shape}")
                    print(f"    输出范围: [{output.min().item():.6f}, {output.max().item():.6f}]")
                    print(f"    调制因子范围: [{modulation_effect.min().item():.6f}, {modulation_effect.max().item():.6f}]")
    
    return cue, mixture, labels

def check_model_structure(model):
    """检查模型结构"""
    print("\n" + "="*80)
    print("模型结构检查")
    print("="*80)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"不可训练参数量: {total_params - trainable_params:,}")
    
    # 检查调制模块
    if hasattr(model, 'modulations'):
        print(f"\n调制模块数量: {len(model.modulations)}")
        for i, mod in enumerate(model.modulations):
            if mod is not None:
                mod_params = sum(p.numel() for p in mod.parameters())
                print(f"  调制层 {i}: {type(mod).__name__}, 参数: {mod_params:,}")
                
                # 显示调制层详细信息
                print(f"    输入通道: {mod.in_channels}, 空间大小: {mod.spatial_size}")
                if hasattr(mod, 'hidden_dim'):
                    print(f"    hidden_dim: {mod.hidden_dim}")
    
    # 检查输出层
    if hasattr(model, 'out_layer'):
        out_layer_params = sum(p.numel() for p in model.out_layer.parameters())
        print(f"\n输出层参数: {out_layer_params:,}")
        
    # 获取输出类别数
    if hasattr(model, 'out_layer'):
        for module in model.out_layer.modules():
            if isinstance(module, torch.nn.Linear):
                print(f"输出层线性层: {module.in_features} -> {module.out_features}")

def clear_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA内存已清理，当前占用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型梯度检查')
    parser.add_argument('--checkpoint', type=str, help='检查点路径（可选）')
    parser.add_argument('--config', type=str, default='config/config1.yaml', help='配置文件路径')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', type=int, default=2, help='批大小')
    args = parser.parse_args()
    
    # 清理内存
    clear_memory()
    
    # 设置设备
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用设备: CUDA ({torch.cuda.get_device_name(0)})")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        print(f"使用设备: CPU")
    
    # 加载配置
    config = load_config(args.config)
    print(f"加载配置文件: {args.config}")
    
    # 创建模型
    print("创建模型...")
    model = Conv2dEIRNN(**config['model'])
    
    # 检查模型结构
    check_model_structure(model)
    
    # 如果有checkpoint，尝试加载
    if args.checkpoint and os.path.exists(args.checkpoint):
        epoch, history = load_checkpoint(model, args.checkpoint, strict=False)
    else:
        print("未提供checkpoint或checkpoint不存在，使用随机初始化")
        epoch = 0
    
    # 清理内存
    clear_memory()
    
    # 测试前向传播
    try:
        cue, mixture, labels = test_forward_pass(model, config, device)
    except Exception as e:
        print(f"前向传播测试失败: {e}")
        print("使用更小的输入进行测试...")
        # 使用更小的输入
        cue = torch.randn(1, 3, 64, 64).to(device)
        mixture = torch.randn(1, 3, 64, 64).to(device)
        labels = torch.randint(0, 6, (1,)).to(device)
    
    # 清理内存
    clear_memory()
    
    # 检查梯度
    try:
        grad_info = check_gradients(model, cue, mixture, labels, device)
        
        # 检查是否有梯度问题
        print("\n" + "="*80)
        print("梯度问题检查")
        print("="*80)
        
        if grad_info['mean_grad_norm']:
            max_grad = max(grad_info['max_grad_norm'])
            mean_grad = np.mean(grad_info['mean_grad_norm'])
            
            if max_grad > 100:
                print("⚠️  警告：检测到梯度爆炸（最大梯度 > 100）!")
            elif max_grad > 10:
                print("⚠️  注意：梯度较大（最大梯度 > 10）")
            
            if mean_grad < 1e-6:
                print("⚠️  警告：检测到梯度消失（平均梯度 < 1e-6）!")
            elif mean_grad < 1e-4:
                print("⚠️  注意：梯度较小（平均梯度 < 1e-4）")
            
            if np.mean(grad_info['zero_grad_ratio']) > 0.9:
                print("⚠️  警告：超过90%的参数梯度为零!")
            
            print(f"\n梯度状态总结:")
            print(f"最大梯度值: {max_grad:.6f}")
            print(f"平均梯度范数: {mean_grad:.6f}")
            print(f"平均零梯度比例: {np.mean(grad_info['zero_grad_ratio']):.2%}")
        else:
            print("没有获取到梯度信息")
            
    except Exception as e:
        print(f"梯度检查失败: {e}")
        print("尝试使用更小的批次...")
        try:
            # 使用单个样本
            cue_small = cue[:1] if cue.shape[0] > 1 else cue
            mixture_small = mixture[:1] if mixture.shape[0] > 1 else mixture
            labels_small = labels[:1] if labels.shape[0] > 1 else labels
            
            grad_info = check_gradients(model, cue_small, mixture_small, labels_small, device)
        except Exception as e2:
            print(f"再次失败: {e2}")
    
    print("\n" + "="*80)
    print("检查完成!")
    print("="*80)

if __name__ == "__main__":
    main()
