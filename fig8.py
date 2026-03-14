import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ===== 常数定义 =====
DT_MS = 1.0  # 时间步长 1ms
MAX_TAU_MS = 20.0  # 纵轴最大显示范围
CLOSE_THRESHOLD = 0.01  # 判定"非常接近"的阈值（1%）


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

def load_checkpoint_and_extract_tau(checkpoint_path, model_config):
    """
    从checkpoint加载并提取所有层的tau参数
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
    
    # 存储每层的tau
    tau_data = {
        'pyr': [],   # 兴奋性神经元tau
        'inter': []  # 抑制性神经元tau
    }
    
    # 确定层数
    num_layers = len(model_config['h_pyr_dim'])
    print(f"\nModel has {num_layers} layers")
    
    for layer_idx in range(num_layers):
        print(f"\nLayer {layer_idx}:")
        
        # 兴奋性神经元的tau
        tau_pyr_key = f'layers.{layer_idx}.tau_pyr'
        if tau_pyr_key in state_dict:
            tau_pyr_raw = ensure_tau_spatial(
                state_dict[tau_pyr_key].cpu().numpy()
            )  # shape: (1, channels, H, W) or (1, channels, 1, 1)
            
            # 转换为ms
            tau_pyr_ms = np.array([raw_tau_to_ms(r) for r in tau_pyr_raw.flatten()]).reshape(tau_pyr_raw.shape)
            
            # 计算每个通道的平均值
            tau_pyr_channel_mean = np.mean(tau_pyr_ms[0], axis=(1, 2))
            
            # 计算该层的全局平均值
            layer_global_mean = np.mean(tau_pyr_ms)
            
            # 计算每层总神经元个数（通道数 × 空间位置数）
            total_neurons = tau_pyr_raw.shape[1] * tau_pyr_raw.shape[2] * tau_pyr_raw.shape[3]
            n_extreme = max(1, total_neurons // 100)  # 取前1%，至少为1
            
            print(f"    Layer {layer_idx} - Total neurons: {total_neurons}, Taking top/bottom/middle {n_extreme} values (1%)")
            
            # 根据需求计算每个通道的代表值
            tau_pyr_channel_representative = np.zeros_like(tau_pyr_channel_mean)
            channel_types = []  # 记录每个通道的类型：'top', 'bottom', 'middle'
            
            for c in range(tau_pyr_raw.shape[1]):
                # 获取该通道的所有值
                channel_values = tau_pyr_ms[0, c].flatten()
                
                # 判断通道平均值与层平均值的相对差异
                rel_diff = abs(tau_pyr_channel_mean[c] - layer_global_mean) / layer_global_mean
                
                if rel_diff < CLOSE_THRESHOLD:
                    # 非常接近层平均：取中间的n_extreme个值的平均
                    sorted_values = np.sort(channel_values)
                    mid_start = (len(sorted_values) - n_extreme) // 2
                    mid_values = sorted_values[mid_start:mid_start + n_extreme]
                    tau_pyr_channel_representative[c] = np.mean(mid_values)
                    channel_types.append('middle')
                elif tau_pyr_channel_mean[c] > layer_global_mean:
                    # 大于层平均：取最大的n_extreme个值的平均
                    top_values = np.sort(channel_values)[-n_extreme:]
                    tau_pyr_channel_representative[c] = np.mean(top_values)
                    channel_types.append('top')
                else:
                    # 小于层平均：取最小的n_extreme个值的平均
                    bottom_values = np.sort(channel_values)[:n_extreme]
                    tau_pyr_channel_representative[c] = np.mean(bottom_values)
                    channel_types.append('bottom')
            
            tau_data['pyr'].append({
                'channel_representative': tau_pyr_channel_representative,
                'channel_means': tau_pyr_channel_mean,
                'channel_types': channel_types,
                'layer_global_mean': layer_global_mean,
                'all_values_ms': tau_pyr_ms.flatten(),
                'shape': tau_pyr_raw.shape,
                'n_channels': tau_pyr_raw.shape[1],
                'n_extreme': n_extreme,
                'total_neurons': total_neurons
            })
            
            # 统计各类型通道数量
            n_top = sum(1 for t in channel_types if t == 'top')
            n_bottom = sum(1 for t in channel_types if t == 'bottom')
            n_middle = sum(1 for t in channel_types if t == 'middle')
            
            print(f"  Pyr tau: shape {tau_pyr_raw.shape}, {tau_pyr_raw.shape[1]} channels")
            print(f"    Layer global mean: {layer_global_mean:.1f} ms")
            print(f"    Channel types - Top: {n_top}, Bottom: {n_bottom}, Middle: {n_middle}")
            print(f"    Channel reps - min: {tau_pyr_channel_representative.min():.1f} ms, max: {tau_pyr_channel_representative.max():.1f} ms")
            print(f"    Channel reps - mean: {tau_pyr_channel_representative.mean():.1f} ms, std: {tau_pyr_channel_representative.std():.1f} ms")
        else:
            print(f"  Warning: {tau_pyr_key} not found")
            tau_data['pyr'].append(None)
        
        # 抑制性神经元的tau
        tau_inter_key = f'layers.{layer_idx}.tau_inter'
        if tau_inter_key in state_dict:
            tau_inter_raw = ensure_tau_spatial(
                state_dict[tau_inter_key].cpu().numpy()
            )  # shape: (1, channels, H, W) or (1, channels, 1, 1)
            
            # 转换为ms
            tau_inter_ms = np.array([raw_tau_to_ms(r) for r in tau_inter_raw.flatten()]).reshape(tau_inter_raw.shape)
            
            # 计算每个通道的平均值
            tau_inter_channel_mean = np.mean(tau_inter_ms[0], axis=(1, 2))
            
            # 计算该层的全局平均值
            layer_global_mean = np.mean(tau_inter_ms)
            
            # 计算每层总神经元个数
            total_neurons = tau_inter_raw.shape[1] * tau_inter_raw.shape[2] * tau_inter_raw.shape[3]
            n_extreme = max(1, total_neurons // 100)
            
            print(f"    Layer {layer_idx} - Total neurons: {total_neurons}, Taking top/bottom/middle {n_extreme} values (1%)")
            
            # 根据需求计算每个通道的代表值
            tau_inter_channel_representative = np.zeros_like(tau_inter_channel_mean)
            channel_types = []
            
            for c in range(tau_inter_raw.shape[1]):
                # 获取该通道的所有值
                channel_values = tau_inter_ms[0, c].flatten()
                
                # 判断通道平均值与层平均值的相对差异
                rel_diff = abs(tau_inter_channel_mean[c] - layer_global_mean) / layer_global_mean
                
                if rel_diff < CLOSE_THRESHOLD:
                    # 非常接近层平均：取中间的n_extreme个值的平均
                    sorted_values = np.sort(channel_values)
                    mid_start = (len(sorted_values) - n_extreme) // 2
                    mid_values = sorted_values[mid_start:mid_start + n_extreme]
                    tau_inter_channel_representative[c] = np.mean(mid_values)
                    channel_types.append('middle')
                elif tau_inter_channel_mean[c] > layer_global_mean:
                    # 大于层平均：取最大的n_extreme个值的平均
                    top_values = np.sort(channel_values)[-n_extreme:]
                    tau_inter_channel_representative[c] = np.mean(top_values)
                    channel_types.append('top')
                else:
                    # 小于层平均：取最小的n_extreme个值的平均
                    bottom_values = np.sort(channel_values)[:n_extreme]
                    tau_inter_channel_representative[c] = np.mean(bottom_values)
                    channel_types.append('bottom')
            
            tau_data['inter'].append({
                'channel_representative': tau_inter_channel_representative,
                'channel_means': tau_inter_channel_mean,
                'channel_types': channel_types,
                'layer_global_mean': layer_global_mean,
                'all_values_ms': tau_inter_ms.flatten(),
                'shape': tau_inter_raw.shape,
                'n_channels': tau_inter_raw.shape[1],
                'n_extreme': n_extreme,
                'total_neurons': total_neurons
            })
            
            # 统计各类型通道数量
            n_top = sum(1 for t in channel_types if t == 'top')
            n_bottom = sum(1 for t in channel_types if t == 'bottom')
            n_middle = sum(1 for t in channel_types if t == 'middle')
            
            print(f"  Inter tau: shape {tau_inter_raw.shape}, {tau_inter_raw.shape[1]} channels")
            print(f"    Layer global mean: {layer_global_mean:.1f} ms")
            print(f"    Channel types - Top: {n_top}, Bottom: {n_bottom}, Middle: {n_middle}")
            print(f"    Channel reps - min: {tau_inter_channel_representative.min():.1f} ms, max: {tau_inter_channel_representative.max():.1f} ms")
            print(f"    Channel reps - mean: {tau_inter_channel_representative.mean():.1f} ms, std: {tau_inter_channel_representative.std():.1f} ms")
        else:
            print(f"  Warning: {tau_inter_key} not found")
            tau_data['inter'].append(None)
    
    return tau_data

def generate_initial_tau_distribution(model_config, num_samples_per_channel=1000):
    """
    生成初始化时的tau分布（用于对比）
    """
    np.random.seed(42)
    
    init_tau_data = {
        'pyr': [],
        'inter': []
    }
    
    num_layers = len(model_config['h_pyr_dim'])
    
    for layer_idx in range(num_layers):
        # 兴奋性神经元
        n_pyr_channels = model_config['h_pyr_dim'][layer_idx]
        init_channel_raw = np.random.randn(n_pyr_channels)
        init_channel_means_ms = np.array(
            [raw_tau_to_ms(r) for r in init_channel_raw]
        )
        
        init_tau_data['pyr'].append({
            'channel_representative': init_channel_means_ms,  # 初始化用平均值作为代表
            'n_channels': n_pyr_channels
        })
        
        # 抑制性神经元
        n_inter_channels = model_config['h_inter_dim'][layer_idx]
        init_channel_raw = np.random.randn(n_inter_channels)
        init_channel_means_ms = np.array(
            [raw_tau_to_ms(r) for r in init_channel_raw]
        )
        
        init_tau_data['inter'].append({
            'channel_representative': init_channel_means_ms,
            'n_channels': n_inter_channels
        })
    
    return init_tau_data

def plot_tau_by_channel_with_initial(tau_data, init_tau_data, model_config, save_dir=None):
    """
    每个子图显示每个通道的代表值（极端值平均）
    """
    num_layers = len(model_config['h_pyr_dim'])
    
    # ===== 兴奋性神经元 =====
    fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 5))
    if num_layers == 1:
        axes = [axes]
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        
        if tau_data['pyr'][layer_idx] is not None:
            # 训练后的代表值
            tau_reps = tau_data['pyr'][layer_idx]['channel_representative']
            channel_types = tau_data['pyr'][layer_idx]['channel_types']
            n_channels = len(tau_reps)
            n_extreme = tau_data['pyr'][layer_idx]['n_extreme']
            total_neurons = tau_data['pyr'][layer_idx]['total_neurons']
            
            # 初始化的代表值
            init_tau_reps = init_tau_data['pyr'][layer_idx]['channel_representative']
            
            # 按训练后的tau值从高到低排序
            sort_indices = np.argsort(tau_reps)[::-1]
            tau_sorted = tau_reps[sort_indices]
            init_tau_sorted = init_tau_reps[sort_indices]
            
            # 为不同的通道类型设置不同的颜色
            colors = []
            for idx in sort_indices:
                if channel_types[idx] == 'top':
                    colors.append('darkblue')
                elif channel_types[idx] == 'bottom':
                    colors.append('lightblue')
                else:
                    colors.append('mediumblue')
            
            x_pos = np.arange(n_channels)
            bar_width = 0.35
            
            # 画训练后的柱状图（使用不同颜色）
            for i, (x, val, color) in enumerate(zip(x_pos - bar_width/2, tau_sorted, colors)):
                ax.bar(x, val, width=bar_width, 
                      alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
            
            # 画初始化的柱状图（统一灰色）
            ax.bar(x_pos + bar_width/2, init_tau_sorted, width=bar_width,
                  alpha=0.5, color='gray', edgecolor='black', linewidth=0.5,
                  label='Initial (mean)')
            
            # 添加层全局平均线
            layer_mean = tau_data['pyr'][layer_idx]['layer_global_mean']
            ax.axhline(layer_mean, color='red', linestyle='--', linewidth=2,
                      alpha=0.7, label=f'Layer mean: {layer_mean:.1f} ms')
            
            # 统计信息
            mean_rep = np.mean(tau_reps)
            std_rep = np.std(tau_reps)
            
            # 统计各类型通道数量
            n_top = sum(1 for t in channel_types if t == 'top')
            n_bottom = sum(1 for t in channel_types if t == 'bottom')
            n_middle = sum(1 for t in channel_types if t == 'middle')
            
            ax.text(0.70, 0.95, 
                   f'Reps: μ={mean_rep:.1f} ms, σ={std_rep:.1f}\n'
                   f'Layer μ={layer_mean:.1f} ms\n'
                   f'Top:{n_top} Bot:{n_bottom} Mid:{n_middle}\n'
                   f'1% = {n_extreme} neurons\ndt = {DT_MS} ms',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=8)
            
            # x轴标签
            if n_channels <= 16:
                step = 2
            elif n_channels <= 32:
                step = 4
            elif n_channels <= 64:
                step = 8
            else:
                step = 16
            
            ax.set_xticks(x_pos[::step])
            ax.set_xticklabels([f'{i+1}' for i in range(0, n_channels, step)], fontsize=8, rotation=45)
            ax.set_xlabel('Channel Index (sorted by trained τ)', fontsize=9)
            
            # 添加图例说明颜色
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='darkblue', alpha=0.7, label='Top 1%'),
                Patch(facecolor='mediumblue', alpha=0.7, label='Middle 1%'),
                Patch(facecolor='lightblue', alpha=0.7, label='Bottom 1%'),
                Patch(facecolor='gray', alpha=0.5, label='Initial')
            ]
            ax.legend(handles=legend_elements, fontsize=7, loc='upper right')
        
        ax.set_title(f'Layer {layer_idx+1} - Excitatory\n({total_neurons} neurons, 1%={n_extreme})', 
                    fontsize=11, fontweight='bold')
        ax.set_ylabel('Membrane Time Constant (ms)', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, MAX_TAU_MS)  # 限制纵轴范围到20ms
    
    plt.suptitle(f'Excitatory Neurons: Channel Representative Values\n(Top/Bottom/Middle 1% by channel mean vs layer mean, dt = {DT_MS} ms)', 
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, "tau_excitatory_extreme.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Excitatory plot saved to: {save_path}")
    
    plt.show()
    
    # ===== 抑制性神经元 =====
    fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 5))
    if num_layers == 1:
        axes = [axes]
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        
        if tau_data['inter'][layer_idx] is not None:
            tau_reps = tau_data['inter'][layer_idx]['channel_representative']
            channel_types = tau_data['inter'][layer_idx]['channel_types']
            n_channels = len(tau_reps)
            n_extreme = tau_data['inter'][layer_idx]['n_extreme']
            total_neurons = tau_data['inter'][layer_idx]['total_neurons']
            
            init_tau_reps = init_tau_data['inter'][layer_idx]['channel_representative']
            
            sort_indices = np.argsort(tau_reps)[::-1]
            tau_sorted = tau_reps[sort_indices]
            init_tau_sorted = init_tau_reps[sort_indices]
            
            # 为不同的通道类型设置不同的颜色
            colors = []
            for idx in sort_indices:
                if channel_types[idx] == 'top':
                    colors.append('darkred')
                elif channel_types[idx] == 'bottom':
                    colors.append('lightcoral')
                else:
                    colors.append('red')
            
            x_pos = np.arange(n_channels)
            bar_width = 0.35
            
            # 画训练后的柱状图
            for i, (x, val, color) in enumerate(zip(x_pos - bar_width/2, tau_sorted, colors)):
                ax.bar(x, val, width=bar_width,
                      alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
            
            # 画初始化的柱状图
            ax.bar(x_pos + bar_width/2, init_tau_sorted, width=bar_width,
                  alpha=0.5, color='gray', edgecolor='black', linewidth=0.5,
                  label='Initial (mean)')
            
            layer_mean = tau_data['inter'][layer_idx]['layer_global_mean']
            ax.axhline(layer_mean, color='red', linestyle='--', linewidth=2,
                      alpha=0.7, label=f'Layer mean: {layer_mean:.1f} ms')
            
            mean_rep = np.mean(tau_reps)
            std_rep = np.std(tau_reps)
            
            n_top = sum(1 for t in channel_types if t == 'top')
            n_bottom = sum(1 for t in channel_types if t == 'bottom')
            n_middle = sum(1 for t in channel_types if t == 'middle')
            
            ax.text(0.70, 0.95, 
                   f'Reps: μ={mean_rep:.1f} ms, σ={std_rep:.1f}\n'
                   f'Layer μ={layer_mean:.1f} ms\n'
                   f'Top:{n_top} Bot:{n_bottom} Mid:{n_middle}\n'
                   f'1% = {n_extreme} neurons\ndt = {DT_MS} ms',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=8)
            
            if n_channels <= 16:
                step = 2
            elif n_channels <= 32:
                step = 4
            else:
                step = 8
            
            ax.set_xticks(x_pos[::step])
            ax.set_xticklabels([f'{i+1}' for i in range(0, n_channels, step)], fontsize=8, rotation=45)
            ax.set_xlabel('Channel Index (sorted by trained τ)', fontsize=9)
            
            # 添加图例说明颜色
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='darkred', alpha=0.7, label='Top 1%'),
                Patch(facecolor='red', alpha=0.7, label='Middle 1%'),
                Patch(facecolor='lightcoral', alpha=0.7, label='Bottom 1%'),
                Patch(facecolor='gray', alpha=0.5, label='Initial')
            ]
            ax.legend(handles=legend_elements, fontsize=7, loc='upper right')
        
        ax.set_title(f'Layer {layer_idx+1} - Inhibitory\n({total_neurons} neurons, 1%={n_extreme})', 
                    fontsize=11, fontweight='bold')
        ax.set_ylabel('Membrane Time Constant (ms)', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, MAX_TAU_MS)  # 限制纵轴范围到20ms
    
    plt.suptitle(f'Inhibitory Neurons: Channel Representative Values\n(Top/Bottom/Middle 1% by channel mean vs layer mean, dt = {DT_MS} ms)', 
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, "tau_inhibitory_extreme.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Inhibitory plot saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    model_config = {
        'h_pyr_dim': [16, 32, 64, 128],
        'h_inter_dim': [4, 8, 16, 32]
    }
    
    checkpoint_path = "/home/wyy/DCnet-update/checkpoints_7/ei/checkpoint.pt"
    
    if os.path.exists(checkpoint_path):
        tau_data = load_checkpoint_and_extract_tau(checkpoint_path, model_config)
        
        print("\nGenerating initial tau distribution...")
        init_tau_data = generate_initial_tau_distribution(model_config)
        
        save_dir = os.path.dirname(checkpoint_path)
        
        plot_tau_by_channel_with_initial(tau_data, init_tau_data, model_config, save_dir)
        
        print(f"\nAll figures saved to: {save_dir}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
