import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ===== 常数定义 =====
DT_MS = 1.0  # 时间步长 1ms


def ensure_tau_spatial(tau_array):
    """
    统一tau张量形状，兼容旧版(1, C, H, W)和新版(1, C, 1, 1)。
    """
    if tau_array.ndim != 4:
        raise ValueError(f"Unexpected tau shape: {tau_array.shape}")
    return tau_array

def raw_tau_to_ms(raw_tau, dt_ms=DT_MS):
    """
    将raw tau转换为毫秒时间常数
    raw_tau -> sigmoid -> tau_sigmoid = dt/τ_membrane -> τ_membrane = dt/tau_sigmoid
    """
    tau_sigmoid = 1 / (1 + np.exp(-raw_tau))
    # 避免除以0
    tau_sigmoid = np.clip(tau_sigmoid, 1e-6, 1-1e-6)
    tau_membrane_ms = dt_ms / tau_sigmoid
    return tau_membrane_ms

def load_model_and_extract_all_tau(checkpoint_path, model_config):
    """
    从checkpoint加载模型并提取所有位置的tau值
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Using dt = {DT_MS} ms per time step")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 提取模型状态字典
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Checkpoint contains model_state_dict (epoch: {checkpoint.get('epoch', 'unknown')})")
    else:
        state_dict = checkpoint
        print("Checkpoint is direct state_dict")
    
    # 处理DDP/compile前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v
        elif k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict
    
    # 存储所有层的tau值
    all_tau_data = {
        'pyr': [],  # 兴奋性神经元tau
        'inter': [] # 抑制性神经元tau
    }
    
    # 层信息
    num_layers = len(model_config['h_pyr_dim'])
    print(f"\nModel has {num_layers} layers")
    
    for layer_idx in range(num_layers):
        print(f"\nLayer {layer_idx}:")
        
        # 兴奋性神经元的tau
        tau_pyr_key = f'layers.{layer_idx}.tau_pyr'
        if tau_pyr_key in state_dict:
            tau_pyr_raw = ensure_tau_spatial(
                state_dict[tau_pyr_key].cpu().numpy()
            )
            
            # 转换为ms
            tau_pyr_ms = np.array([raw_tau_to_ms(r) for r in tau_pyr_raw.flatten()])
            
            # 存储
            all_tau_data['pyr'].append({
                'values': tau_pyr_ms,
                'shape': tau_pyr_raw.shape,
                'n_neurons': tau_pyr_ms.size
            })
            
            print(f"  Pyr tau: shape {tau_pyr_raw.shape}")
            print(f"    Total neurons: {tau_pyr_ms.size}")
            print(f"    Mean: {tau_pyr_ms.mean():.2f} ms")
            print(f"    Std: {tau_pyr_ms.std():.2f} ms")
            print(f"    Min: {tau_pyr_ms.min():.2f} ms")
            print(f"    Max: {tau_pyr_ms.max():.2f} ms")
            print(f"    Median: {np.median(tau_pyr_ms):.2f} ms")
        else:
            print(f"  Warning: {tau_pyr_key} not found")
            all_tau_data['pyr'].append(None)
        
        # 抑制性神经元的tau
        tau_inter_key = f'layers.{layer_idx}.tau_inter'
        if tau_inter_key in state_dict:
            tau_inter_raw = ensure_tau_spatial(
                state_dict[tau_inter_key].cpu().numpy()
            )
            
            # 转换为ms
            tau_inter_ms = np.array([raw_tau_to_ms(r) for r in tau_inter_raw.flatten()])
            
            # 存储
            all_tau_data['inter'].append({
                'values': tau_inter_ms,
                'shape': tau_inter_raw.shape,
                'n_neurons': tau_inter_ms.size
            })
            
            print(f"  Inter tau: shape {tau_inter_raw.shape}")
            print(f"    Total neurons: {tau_inter_ms.size}")
            print(f"    Mean: {tau_inter_ms.mean():.2f} ms")
            print(f"    Std: {tau_inter_ms.std():.2f} ms")
            print(f"    Min: {tau_inter_ms.min():.2f} ms")
            print(f"    Max: {tau_inter_ms.max():.2f} ms")
            print(f"    Median: {np.median(tau_inter_ms):.2f} ms")
        else:
            print(f"  Warning: {tau_inter_key} not found")
            all_tau_data['inter'].append(None)
    
    return all_tau_data

def plot_tau_distribution_histogram(all_tau_data, model_config, save_dir=None, max_tau=100):
    """
    画所有位置的真实tau分布直方图
    """
    num_layers = len(model_config['h_pyr_dim'])
    
    # ===== 兴奋性神经元 =====
    fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 4))
    if num_layers == 1:
        axes = [axes]
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        
        if all_tau_data['pyr'][layer_idx] is not None:
            tau_values = all_tau_data['pyr'][layer_idx]['values']
            n_neurons = all_tau_data['pyr'][layer_idx]['n_neurons']
            
            # 限制范围以便更好地观察
            tau_values_clipped = tau_values[tau_values <= max_tau]
            n_outliers = n_neurons - len(tau_values_clipped)
            
            # 画直方图
            counts, bins, patches = ax.hist(tau_values_clipped, bins=50, alpha=0.7, 
                                           color='blue', edgecolor='black', linewidth=0.5)
            
            # 添加统计信息
            mean_val = np.mean(tau_values)
            median_val = np.median(tau_values)
            std_val = np.std(tau_values)
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.1f} ms')
            ax.axvline(median_val, color='green', linestyle=':', linewidth=2, 
                      label=f'Median: {median_val:.1f} ms')
            
            # 添加文本信息
            text_str = (f'Total neurons: {n_neurons}\n'
                       f'Mean: {mean_val:.1f} ms\n'
                       f'Median: {median_val:.1f} ms\n'
                       f'Std: {std_val:.1f} ms\n'
                       f'Min: {tau_values.min():.1f} ms\n'
                       f'Max: {tau_values.max():.1f} ms')
            
            if n_outliers > 0:
                text_str += f'\n> {max_tau}ms: {n_outliers} ({n_outliers/n_neurons*100:.1f}%)'
            
            ax.text(0.70, 0.95, text_str,
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=8)
            
            ax.set_xlabel('Time Constant (ms)', fontsize=9)
            ax.set_ylabel('Number of neurons', fontsize=9)
            ax.set_yscale('log')  # 使用对数坐标
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(fontsize=7, loc='upper right')
            ax.set_xlim(0, min(max_tau, tau_values.max()))
        
        ax.set_title(f'Layer {layer_idx+1} - Excitatory\n({n_neurons} neurons, dt={DT_MS}ms)', 
                    fontsize=11, fontweight='bold')
    
    plt.suptitle(f'Excitatory Neurons: Distribution of Time Constants (dt={DT_MS}ms)', 
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, "tau_excitatory_histogram.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Excitatory histogram saved to: {save_path}")
    
    plt.show()
    
    # ===== 抑制性神经元 =====
    fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 4))
    if num_layers == 1:
        axes = [axes]
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        
        if all_tau_data['inter'][layer_idx] is not None:
            tau_values = all_tau_data['inter'][layer_idx]['values']
            n_neurons = all_tau_data['inter'][layer_idx]['n_neurons']
            
            # 限制范围以便更好地观察
            tau_values_clipped = tau_values[tau_values <= max_tau]
            n_outliers = n_neurons - len(tau_values_clipped)
            
            # 画直方图
            counts, bins, patches = ax.hist(tau_values_clipped, bins=50, alpha=0.7, 
                                           color='red', edgecolor='black', linewidth=0.5)
            
            # 添加统计信息
            mean_val = np.mean(tau_values)
            median_val = np.median(tau_values)
            std_val = np.std(tau_values)
            
            ax.axvline(mean_val, color='blue', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.1f} ms')
            ax.axvline(median_val, color='green', linestyle=':', linewidth=2, 
                      label=f'Median: {median_val:.1f} ms')
            
            # 添加文本信息
            text_str = (f'Total neurons: {n_neurons}\n'
                       f'Mean: {mean_val:.1f} ms\n'
                       f'Median: {median_val:.1f} ms\n'
                       f'Std: {std_val:.1f} ms\n'
                       f'Min: {tau_values.min():.1f} ms\n'
                       f'Max: {tau_values.max():.1f} ms')
            
            if n_outliers > 0:
                text_str += f'\n> {max_tau}ms: {n_outliers} ({n_outliers/n_neurons*100:.1f}%)'
            
            ax.text(0.70, 0.95, text_str,
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=8)
            
            ax.set_xlabel('Time Constant (ms)', fontsize=9)
            ax.set_ylabel('Number of neurons', fontsize=9)
            ax.set_yscale('log')  # 使用对数坐标
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(fontsize=7, loc='upper right')
            ax.set_xlim(0, min(max_tau, tau_values.max()))
        
        ax.set_title(f'Layer {layer_idx+1} - Inhibitory\n({n_neurons} neurons, dt={DT_MS}ms)', 
                    fontsize=11, fontweight='bold')
    
    plt.suptitle(f'Inhibitory Neurons: Distribution of Time Constants (dt={DT_MS}ms)', 
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, "tau_inhibitory_histogram.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Inhibitory histogram saved to: {save_path}")
    
    plt.show()

def plot_combined_histogram(all_tau_data, model_config, save_dir=None, max_tau=100):
    """
    将所有层的tau画在一起
    """
    num_layers = len(model_config['h_pyr_dim'])
    
    # 兴奋性神经元
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'cyan', 'green', 'orange']
    for layer_idx in range(num_layers):
        if all_tau_data['pyr'][layer_idx] is not None:
            tau_values = all_tau_data['pyr'][layer_idx]['values']
            tau_values_clipped = tau_values[tau_values <= max_tau]
            
            plt.hist(tau_values_clipped, bins=50, alpha=0.5, color=colors[layer_idx % len(colors)],
                    label=f'Layer {layer_idx+1} (n={len(tau_values)})', edgecolor='black', linewidth=0.5)
    
    plt.xlabel('Time Constant (ms)', fontsize=11)
    plt.ylabel('Number of neurons', fontsize=11)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(fontsize=9)
    plt.title(f'Excitatory Neurons: Distribution of Time Constants by Layer (dt={DT_MS}ms)', fontsize=13)
    plt.xlim(0, max_tau)
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, "tau_excitatory_combined_histogram.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined excitatory histogram saved to: {save_path}")
    
    plt.show()
    
    # 抑制性神经元
    plt.figure(figsize=(10, 6))
    
    for layer_idx in range(num_layers):
        if all_tau_data['inter'][layer_idx] is not None:
            tau_values = all_tau_data['inter'][layer_idx]['values']
            tau_values_clipped = tau_values[tau_values <= max_tau]
            
            plt.hist(tau_values_clipped, bins=50, alpha=0.5, color=colors[layer_idx % len(colors)],
                    label=f'Layer {layer_idx+1} (n={len(tau_values)})', edgecolor='black', linewidth=0.5)
    
    plt.xlabel('Time Constant (ms)', fontsize=11)
    plt.ylabel('Number of neurons', fontsize=11)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(fontsize=9)
    plt.title(f'Inhibitory Neurons: Distribution of Time Constants by Layer (dt={DT_MS}ms)', fontsize=13)
    plt.xlim(0, max_tau)
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, "tau_inhibitory_combined_histogram.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined inhibitory histogram saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # 模型配置
    model_config = {
        'h_pyr_dim': [16, 32, 64, 128],   # 每层兴奋性神经元通道数
        'h_inter_dim': [4, 8, 16, 32]     # 每层抑制性神经元通道数
    }
    
    # checkpoint路径
    checkpoint_path = "/home/wyy/DCnet-update/checkpoints_7/ei/checkpoint.pt"
    
    # 设置显示范围
    MAX_TAU_DISPLAY = 50  # 只显示0-50ms的范围，可以调整
    
    if os.path.exists(checkpoint_path):
        # 提取所有tau值
        all_tau_data = load_model_and_extract_all_tau(checkpoint_path, model_config)
        
        # 获取保存目录
        save_dir = os.path.dirname(checkpoint_path)
        
        # 画每层单独的直方图
        plot_tau_distribution_histogram(all_tau_data, model_config, save_dir, max_tau=MAX_TAU_DISPLAY)
        
        # 画合并的直方图
        plot_combined_histogram(all_tau_data, model_config, save_dir, max_tau=MAX_TAU_DISPLAY)
        
        print(f"\nAll histogram figures saved to: {save_dir}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
