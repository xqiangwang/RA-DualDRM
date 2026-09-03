#!/usr/bin/env python3
"""
傅立叶特征映射在强间断裂纹位移场上的局限性分析

本脚本通过系统对比 MLP 与 FourierFeatureMLP，揭示傅立叶特征映射
在表达不连续裂纹位移场时的根本性缺陷：

1. 谱偏置（Spectral Bias）：傅立叶特征天然偏好低频、光滑函数
2. Gibbs 现象：用全局光滑的正弦基拟合局部间断，产生振荡/过冲
3. 能量泄漏：间断特征的能量扩散到整个频谱，浪费网络容量
4. 局部性与全局性的冲突：裂纹是局部特征，傅立叶基是全局支撑

输出：定量指标 + 可视化对比图，用于论文论证。
"""

import argparse
import csv
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# 几何与参考场（与主实验一致）
# ============================================================================

class CrackGeometry:
    def __init__(self, a=0.5, eps_H=0.05, ell=0.1):
        self.a = a
        self.eps_H = eps_H
        self.ell = ell

    def signed_distance_to_crack(self, x, y):
        a = self.a
        dc = torch.zeros_like(x)
        mask_mid = torch.abs(x) <= a
        dc = torch.where(mask_mid, torch.abs(y), dc)
        mask_left = x < -a
        dc = torch.where(mask_left, torch.sqrt((x + a) ** 2 + y ** 2), dc)
        mask_right = x > a
        dc = torch.where(mask_right, torch.sqrt((x - a) ** 2 + y ** 2), dc)
        return dc

    def phase_field(self, x, y):
        dc = self.signed_distance_to_crack(x, y)
        return torch.exp(-(dc ** 2) / (self.ell ** 2))

    def phase_field_and_grad(self, x, y):
        a = self.a
        ell2 = self.ell ** 2
        eps = 1e-10

        dc = torch.zeros_like(x)
        dc_x = torch.zeros_like(x)
        dc_y = torch.zeros_like(x)

        mask_mid = torch.abs(x) <= a
        dc = torch.where(mask_mid, torch.abs(y), dc)
        dc_y = torch.where(mask_mid, torch.sign(y), dc_y)

        mask_left = x < -a
        dx_l = x + a
        d_left = torch.sqrt(dx_l ** 2 + y ** 2)
        dc = torch.where(mask_left, d_left, dc)
        dc_x = torch.where(mask_left, dx_l / (d_left + eps), dc_x)
        dc_y = torch.where(mask_left, y / (d_left + eps), dc_y)

        mask_right = x > a
        dx_r = x - a
        d_right = torch.sqrt(dx_r ** 2 + y ** 2)
        dc = torch.where(mask_right, d_right, dc)
        dc_x = torch.where(mask_right, dx_r / (d_right + eps), dc_x)
        dc_y = torch.where(mask_right, y / (d_right + eps), dc_y)

        alpha = torch.exp(-(dc ** 2) / ell2)
        factor = -2.0 * dc / ell2
        alpha_x = alpha * factor * dc_x
        alpha_y = alpha * factor * dc_y
        return alpha, alpha_x, alpha_y

    def heaviside_smooth(self, x, y):
        return torch.tanh(y / self.eps_H)


class ReferenceField:
    def __init__(self, geom, mode='mode_I'):
        self.geom = geom
        self.mode = mode
        if mode == 'mode_I':
            self.eps0, self.w0 = 0.02, 0.2
            self.gamma0, self.s0 = 0.0, 0.0
        elif mode == 'mode_II':
            self.eps0, self.w0 = 0.0, 0.0
            self.gamma0, self.s0 = 0.02, 0.2
        elif mode == 'mixed':
            self.eps0, self.w0 = 0.02, 0.15
            self.gamma0, self.s0 = 0.02, 0.15
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __call__(self, x, y):
        H = self.geom.heaviside_smooth(x, y)
        alpha = self.geom.phase_field(x, y)
        ux = self.gamma0 * y + 0.5 * self.s0 * H * alpha
        uy = self.eps0 * y + 0.5 * self.w0 * H * alpha
        return torch.cat([ux, uy], dim=-1)


# ============================================================================
# 数据生成
# ============================================================================

def generate_training_data(geom, N_uniform=20000, N_crack=10000, device='cpu'):
    x_uni = torch.rand(N_uniform, device=device) * 2.0 - 1.0
    y_uni = torch.rand(N_uniform, device=device) * 2.0 - 1.0
    x_crack = torch.rand(N_crack, device=device) * 1.4 - 0.7
    y_crack = torch.randn(N_crack, device=device) * 0.08
    y_crack = torch.clamp(y_crack, -1.0, 1.0)
    x = torch.cat([x_uni, x_crack]).unsqueeze(-1)
    y = torch.cat([y_uni, y_crack]).unsqueeze(-1)
    perm = torch.randperm(x.shape[0])
    return x[perm], y[perm]


def generate_test_grid(nx=400, ny=400, device='cpu'):
    x = torch.linspace(-1, 1, nx, device=device)
    y = torch.linspace(-1, 1, ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    return X, Y


# ============================================================================
# 网络架构
# ============================================================================

class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, hidden_dim=64, num_layers=4,
                 activation='tanh'):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers + [out_dim]
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.layers.append(nn.Tanh() if activation == 'tanh' else nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FourierFeatureMLP(nn.Module):
    def __init__(self, out_dim=2, hidden_dim=64, num_layers=4,
                 num_frequencies=6, activation='tanh'):
        super().__init__()
        self.num_frequencies = num_frequencies
        in_dim = 4 * num_frequencies
        self.mlp = MLP(in_dim=in_dim, out_dim=out_dim,
                       hidden_dim=hidden_dim, num_layers=num_layers,
                       activation=activation)

    def fourier_embed(self, x, y):
        feats = []
        for k in range(1, self.num_frequencies + 1):
            freq = 2.0 * np.pi * k
            feats.extend([
                torch.sin(freq * x), torch.sin(freq * y),
                torch.cos(freq * x), torch.cos(freq * y),
            ])
        return torch.cat(feats, dim=-1)

    def forward(self, x, y):
        return self.mlp(self.fourier_embed(x, y))


# ============================================================================
# 训练
# ============================================================================

def train_model(model, x_train, y_train, ref_field, geom,
                epochs=5000, lr=1e-3, batch_size=4096, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    u_ref_train = ref_field(x_train, y_train)
    N = x_train.shape[0]
    n_batches = (N + batch_size - 1) // batch_size

    loss_history = []
    t0 = time.time()

    for epoch in range(epochs):
        perm = torch.randperm(N)
        x_s = x_train[perm]
        y_s = y_train[perm]
        u_s = u_ref_train[perm]

        epoch_loss = 0.0
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, N)
            xb = x_s[start:end]
            yb = y_s[start:end]
            ub = u_s[start:end]

            if isinstance(model, MLP):
                u_pred = model(torch.cat([xb, yb], dim=-1))
            elif isinstance(model, FourierFeatureMLP):
                u_pred = model(xb, yb)

            loss = torch.mean((u_pred - ub) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * (end - start)

        epoch_loss /= N
        loss_history.append(epoch_loss)
        scheduler.step()

        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:5d}/{epochs} | Loss: {epoch_loss:.6e}")

    return {'loss_history': loss_history, 'train_time': time.time() - t0}


# ============================================================================
# 评估
# ============================================================================

def compute_metrics(model, X, Y, ref_field, geom, alpha_c=0.3, delta=0.05):
    device = X.device
    model.eval()
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    u_ref = ref_field(x_flat, y_flat)

    with torch.no_grad():
        if isinstance(model, MLP):
            u_pred = model(torch.cat([x_flat, y_flat], dim=-1))
        elif isinstance(model, FourierFeatureMLP):
            u_pred = model(x_flat, y_flat)

    diff = u_pred - u_ref
    E_L2 = torch.sqrt(torch.mean(diff ** 2)) / torch.sqrt(torch.mean(u_ref ** 2))

    alpha = geom.phase_field(x_flat, y_flat)
    mask_crack = alpha.squeeze() > alpha_c
    if mask_crack.sum() > 0:
        E_crack = (torch.sqrt(torch.mean(diff[mask_crack] ** 2)) /
                   torch.sqrt(torch.mean(u_ref[mask_crack] ** 2)))
    else:
        E_crack = torch.tensor(float('nan'))

    mask_smooth = alpha.squeeze() < 0.1
    if mask_smooth.sum() > 0:
        E_smooth = (torch.sqrt(torch.mean(diff[mask_smooth] ** 2)) /
                    torch.sqrt(torch.mean(u_ref[mask_smooth] ** 2)))
    else:
        E_smooth = torch.tensor(float('nan'))

    # Jump error
    a = geom.a
    n_jump = 200
    x_jump = torch.linspace(-a, a, n_jump, device=device).unsqueeze(-1)
    y_plus = torch.full_like(x_jump, delta)
    y_minus = torch.full_like(x_jump, -delta)

    u_ref_plus = ref_field(x_jump, y_plus)
    u_ref_minus = ref_field(x_jump, y_minus)
    jump_ref = u_ref_plus - u_ref_minus

    with torch.no_grad():
        if isinstance(model, MLP):
            cp = torch.cat([x_jump, y_plus], dim=-1)
            cm = torch.cat([x_jump, y_minus], dim=-1)
            u_pred_plus = model(cp)
            u_pred_minus = model(cm)
        elif isinstance(model, FourierFeatureMLP):
            u_pred_plus = model(x_jump, y_plus)
            u_pred_minus = model(x_jump, y_minus)

    jump_pred = u_pred_plus - u_pred_minus
    E_jump = (torch.sqrt(torch.mean((jump_pred - jump_ref) ** 2)) /
              torch.sqrt(torch.mean(jump_ref ** 2)))

    return {
        'E_L2': E_L2.item(),
        'E_crack': E_crack.item(),
        'E_smooth': E_smooth.item(),
        'E_jump': E_jump.item(),
    }


# ============================================================================
# 可视化：论文级分析图
# ============================================================================

def _setup_paper_fonts():
    """设置论文字体：Times New Roman"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11


def _save_figure(fig, outdir, basename):
    """同时保存 PNG 和 PDF"""
    png_path = os.path.join(outdir, f'{basename}.png')
    pdf_path = os.path.join(outdir, f'{basename}.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  Saved: {basename}.png + .pdf")


def _label_subplots(axes, labels=None, fontsize=14, fontweight='normal', x=-0.15, y=1.05):
    """给子图添加 (a)(b)(c)... 编号"""
    if labels is None:
        labels = [f'({chr(ord("a") + i)})' for i in range(len(axes))]
    for ax, label in zip(axes, labels):
        ax.text(x, y, label, transform=ax.transAxes,
                fontsize=fontsize, fontweight=fontweight, va='top', ha='right')


def plot_fig1_reference_and_loss(X, Y, u_ref, all_losses, outdir):
    """
    图1: 左=参考位移场 uy, 右=MLP vs Fourier 训练损失曲线
    """
    _setup_paper_fonts()
    nx, ny = X.shape
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    u_ref_np = u_ref.cpu().numpy().reshape(nx, ny, 2)

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    # 左：参考场 uy
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.contourf(X_np, Y_np, u_ref_np[:, :, 1], levels=50, cmap='RdBu_r')
    plt.colorbar(im, ax=ax0, shrink=0.8)
    ax0.set_aspect('equal')
    ax0.set_xlabel('$x$')
    ax0.set_ylabel('$y$')

    # 右：loss 曲线
    ax1 = fig.add_subplot(gs[0, 1])
    colors = {'MLP': '#1f77b4', 'Fourier': '#ff7f0e'}
    for name, losses in all_losses.items():
        ax1.semilogy(losses, label=name, linewidth=2, color=colors.get(name, 'gray'))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss (log scale)')
    ax1.legend(fontsize=11, frameon=True)
    ax1.grid(True, which='both', linestyle='--', alpha=0.4)

    _label_subplots([ax0, ax1], ['(a)', '(b)'])
    plt.tight_layout()
    _save_figure(fig, outdir, 'fig1_reference_and_loss')
    plt.close()

    # 保存数据
    np.savez(os.path.join(outdir, 'fig1_data.npz'),
             X=X_np, Y=Y_np, uy_ref=u_ref_np[:, :, 1],
             loss_mlp=np.array(all_losses['MLP']),
             loss_fourier=np.array(all_losses['Fourier']))


def plot_fig2_prediction_and_error(X, Y, u_ref, u_pred_dict, outdir):
    """
    图2: 上排=MLP预测、Fourier预测(uy), 下排=对应误差场
    位移场用 RdBu_r, 误差场用 viridis
    """
    _setup_paper_fonts()
    nx, ny = X.shape
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    u_ref_np = u_ref.cpu().numpy().reshape(nx, ny, 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 提取 uy 预测和误差
    pred_uy = {}
    err_uy = {}
    for name, u_pred in u_pred_dict.items():
        u_pred_np = u_pred.cpu().numpy().reshape(nx, ny, 2)
        pred_uy[name] = u_pred_np[:, :, 1]
        err_uy[name] = np.abs(u_pred_np[:, :, 1] - u_ref_np[:, :, 1])

    # 上排：预测场
    for ax, name in zip(axes[0], ['MLP', 'Fourier']):
        im = ax.contourf(X_np, Y_np, pred_uy[name], levels=50, cmap='RdBu_r')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

    # 下排：误差场 (viridis)
    vmax_err = max(np.max(err_uy['MLP']), np.max(err_uy['Fourier']))
    for ax, name in zip(axes[1], ['MLP', 'Fourier']):
        im = ax.contourf(X_np, Y_np, err_uy[name], levels=50, cmap='viridis',
                         vmin=0, vmax=vmax_err)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

    _label_subplots(axes.flat, ['(a)', '(b)', '(c)', '(d)'])
    plt.tight_layout()
    _save_figure(fig, outdir, 'fig2_prediction_and_error')
    plt.close()

    # 保存数据
    np.savez(os.path.join(outdir, 'fig2_data.npz'),
             X=X_np, Y=Y_np, uy_ref=u_ref_np[:, :, 1],
             uy_mlp=pred_uy['MLP'], uy_fourier=pred_uy['Fourier'],
             err_mlp=err_uy['MLP'], err_fourier=err_uy['Fourier'])


def plot_fig3_cross_section_and_error(X, Y, u_ref, u_pred_dict, model_dict, ref_field, outdir):
    """
    图3: 左=沿x=0的uy剖面曲线, 右=对应误差
    """
    _setup_paper_fonts()
    y_vals = torch.linspace(-0.3, 0.3, 500, device=X.device)
    x_zeros = torch.zeros_like(y_vals).unsqueeze(-1)
    y_sec = y_vals.unsqueeze(-1)

    u_ref_sec = ref_field(x_zeros, y_sec)

    pred_sections = {}
    with torch.no_grad():
        for name, model in model_dict.items():
            model.eval()
            if isinstance(model, MLP):
                pred_sections[name] = model(torch.cat([x_zeros, y_sec], dim=-1))
            elif isinstance(model, FourierFeatureMLP):
                pred_sections[name] = model(x_zeros, y_sec)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    y_np = y_vals.cpu().numpy()
    u_ref_np = u_ref_sec[:, 1].cpu().numpy()
    colors = {'MLP': '#1f77b4', 'Fourier': '#ff7f0e'}

    # 左：位移剖面
    ax0 = axes[0]
    ax0.plot(y_np, u_ref_np, 'k-', linewidth=2.5, label='Reference', zorder=10)
    for name, u_pred in pred_sections.items():
        u_pred_np = u_pred[:, 1].cpu().numpy()
        ax0.plot(y_np, u_pred_np, '--', linewidth=1.8,
                 color=colors.get(name, 'gray'), label=name, alpha=0.9)
    ax0.axvline(x=0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax0.set_xlabel('$y$ (cross-crack direction)')
    ax0.set_ylabel('$u_y$')
    ax0.legend(loc='best', fontsize=10, frameon=True)
    ax0.grid(True, linestyle='--', alpha=0.4)
    ax0.set_xlim(-0.3, 0.3)

    # 右：误差剖面
    ax1 = axes[1]
    for name, u_pred in pred_sections.items():
        err_np = np.abs(u_pred[:, 1].cpu().numpy() - u_ref_np)
        ax1.plot(y_np, err_np, linewidth=2,
                 color=colors.get(name, 'gray'), label=f'{name} error', alpha=0.9)
    ax1.axvline(x=0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('$y$ (cross-crack direction)')
    ax1.set_ylabel('$|u_y^{{\\mathrm{{pred}}}} - u_y^{{\\mathrm{{ref}}}}|$')
    ax1.legend(loc='best', fontsize=10, frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_xlim(-0.3, 0.3)

    _label_subplots([ax0, ax1], ['(a)', '(b)'])
    plt.tight_layout()
    _save_figure(fig, outdir, 'fig3_cross_section_and_error')
    plt.close()

    # 保存数据
    np.savez(os.path.join(outdir, 'fig3_data.npz'),
             y=y_np, uy_ref=u_ref_np,
             uy_mlp=pred_sections['MLP'][:, 1].cpu().numpy(),
             uy_fourier=pred_sections['Fourier'][:, 1].cpu().numpy())


def plot_fig4_spectral_analysis(u_ref_test, u_pred_dict, nx, ny, outdir):
    """
    图4: 频谱分析
    """
    _setup_paper_fonts()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    u_ref_np = u_ref_test.cpu().numpy().reshape(nx, ny, 2)
    uy_ref = u_ref_np[:, :, 1]
    fft_ref = np.fft.fft2(uy_ref)
    fft_ref_shift = np.fft.fftshift(fft_ref)
    psd_ref = np.abs(fft_ref_shift) ** 2

    im0 = axes[0].imshow(np.log10(psd_ref + 1e-10), cmap='hot', aspect='auto')
    axes[0].set_title('Reference $u_y$')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    psd_data = {'reference': psd_ref}
    for ax, (name, u_pred) in zip(axes[1:], u_pred_dict.items()):
        u_pred_np = u_pred.cpu().numpy().reshape(nx, ny, 2)
        uy_pred = u_pred_np[:, :, 1]
        fft_pred = np.fft.fft2(uy_pred)
        fft_pred_shift = np.fft.fftshift(fft_pred)
        psd_pred = np.abs(fft_pred_shift) ** 2
        psd_data[name] = psd_pred

        im = ax.imshow(np.log10(psd_pred + 1e-10), cmap='hot', aspect='auto')
        ax.set_title(f'{name}')
        plt.colorbar(im, ax=ax, shrink=0.8)

    for ax in axes:
        ax.set_xlabel('Frequency $k_x$')
        ax.set_ylabel('Frequency $k_y$')

    _label_subplots(axes, ['(a)', '(b)', '(c)'])
    plt.tight_layout()
    _save_figure(fig, outdir, 'fig4_spectral_analysis')
    plt.close()

    # 保存数据
    np.savez(os.path.join(outdir, 'fig4_data.npz'), **psd_data)


def plot_fig5_fourier_basis(outdir, num_frequencies=6):
    """
    图5: Fourier 基函数可视化
    """
    _setup_paper_fonts()
    x = np.linspace(-1, 1, 400)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    for k in range(1, num_frequencies + 1):
        ax = axes.flat[k - 1]
        freq = 2.0 * np.pi * k
        sin_x = np.sin(freq * x)

        ax.plot(x, sin_x, 'b-', linewidth=1.5, label=r'$\sin(2\pi k x)$')
        ax.fill_between(x, -1.2, 1.2, where=(np.abs(x) < 0.5),
                        alpha=0.2, color='red', label='Crack region')
        ax.axvline(x=-0.5, color='red', linestyle=':', alpha=0.5)
        ax.axvline(x=0.5, color='red', linestyle=':', alpha=0.5)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel('$x$')
        ax.legend(fontsize=8, frameon=True)
        ax.grid(True, alpha=0.3)

    labels = [f'({chr(ord("a") + i)})' for i in range(6)]
    _label_subplots(axes.flat, labels, x=-0.12, y=1.08)
    plt.tight_layout()
    _save_figure(fig, outdir, 'fig5_fourier_basis')
    plt.close()

    # 保存数据
    np.savez(os.path.join(outdir, 'fig5_data.npz'),
             x=x, frequencies=np.arange(1, num_frequencies + 1))


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='傅立叶特征映射局限性分析')
    parser.add_argument('--case', type=str, default='mode_I',
                        choices=['mode_I', 'mode_II', 'mixed'])
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', type=str, default='results')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = os.path.join(args.outdir, args.case)
    os.makedirs(outdir, exist_ok=True)

    device = args.device
    print(f"\n{'='*70}")
    print("傅立叶特征映射在强间断裂纹位移场上的局限性分析")
    print(f"{'='*70}")
    print(f"Case: {args.case} | Epochs: {args.epochs} | Device: {device}")

    geom = CrackGeometry(a=0.5, eps_H=0.05, ell=0.1)
    ref_field = ReferenceField(geom, mode=args.case)

    print("\nGenerating data...")
    x_train, y_train = generate_training_data(geom, 20000, 10000, device)
    X_test, Y_test = generate_test_grid(400, 400, device)
    print(f"  Train: {x_train.shape[0]} | Test grid: {X_test.shape}")

    with torch.no_grad():
        u_ref_test = ref_field(X_test.reshape(-1, 1), Y_test.reshape(-1, 1))

    models = {
        'MLP': MLP(in_dim=2, out_dim=2, hidden_dim=64, num_layers=4, activation='tanh'),
        'Fourier': FourierFeatureMLP(out_dim=2, hidden_dim=64, num_layers=4,
                                      num_frequencies=6, activation='tanh'),
    }

    print("\n" + "=" * 60)
    print("Model parameter counts:")
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name:15s}: {n_params:,}")
    print("=" * 60)

    all_losses = {}
    metrics_list = []
    u_pred_dict = {}

    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")

        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())

        train_info = train_model(
            model, x_train, y_train, ref_field, geom,
            epochs=args.epochs, lr=1e-3, batch_size=4096, device=device
        )

        metrics = compute_metrics(model, X_test, Y_test, ref_field, geom)
        metrics.update({
            'case_name': args.case,
            'model_name': model_name,
            'num_parameters': n_params,
            'train_time_sec': train_info['train_time'],
            'final_loss': train_info['loss_history'][-1],
        })

        all_losses[model_name] = train_info['loss_history']
        metrics_list.append(metrics)

        print(f"  Parameters:  {n_params:,}")
        print(f"  Train time:  {metrics['train_time_sec']:.2f} s")
        print(f"  Final loss:  {metrics['final_loss']:.6e}")
        print(f"  E_L2:        {metrics['E_L2']:.6e}")
        print(f"  E_smooth:    {metrics['E_smooth']:.6e}")
        print(f"  E_crack:     {metrics['E_crack']:.6e}")
        print(f"  E_jump:      {metrics['E_jump']:.6e}")

        with torch.no_grad():
            if isinstance(model, MLP):
                u_pred = model(torch.cat([X_test.reshape(-1, 1), Y_test.reshape(-1, 1)], dim=-1))
            elif isinstance(model, FourierFeatureMLP):
                u_pred = model(X_test.reshape(-1, 1), Y_test.reshape(-1, 1))

        u_pred_dict[model_name] = u_pred

    # ============ 论文级可视化 ============
    print("\n" + "=" * 60)
    print("Generating analysis plots for paper...")
    print("=" * 60)

    # 按用户要求的5张图布局
    plot_fig1_reference_and_loss(X_test, Y_test, u_ref_test, all_losses, outdir)
    plot_fig2_prediction_and_error(X_test, Y_test, u_ref_test, u_pred_dict, outdir)
    plot_fig3_cross_section_and_error(X_test, Y_test, u_ref_test, u_pred_dict,
                                       models, ref_field, outdir)
    plot_fig4_spectral_analysis(u_ref_test, u_pred_dict,
                                 X_test.shape[0], X_test.shape[1], outdir)
    plot_fig5_fourier_basis(outdir)

    # ============ 保存指标 ============
    csv_path = os.path.join(outdir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'case_name', 'model_name', 'num_parameters', 'train_time_sec',
            'final_loss', 'E_L2', 'E_smooth', 'E_crack', 'E_jump'
        ])
        writer.writeheader()
        writer.writerows(metrics_list)
    print(f"\nMetrics saved to {csv_path}")

    # ============ 终端总结 ============
    print("\n" + "=" * 70)
    print("FOURIER LIMITATION ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<15} {'Params':>8} {'E_L2':>10} {'E_smooth':>10} {'E_crack':>10} {'E_jump':>10}")
    print("-" * 70)
    for m in metrics_list:
        print(f"{m['model_name']:<15} {m['num_parameters']:>8,} "
              f"{m['E_L2']:>10.4e} {m['E_smooth']:>10.4e} "
              f"{m['E_crack']:>10.4e} {m['E_jump']:>10.4e}")
    print("=" * 70)

    # 计算 Fourier 相对于 MLP 的劣化倍数
    mlp_metrics = {m['model_name']: m for m in metrics_list}
    if 'MLP' in mlp_metrics and 'Fourier' in mlp_metrics:
        mlp = mlp_metrics['MLP']
        fou = mlp_metrics['Fourier']
        print("\n--- Fourier Degradation Factor (Fourier / MLP) ---")
        print(f"  E_L2:     {fou['E_L2'] / mlp['E_L2']:.2f}x worse")
        print(f"  E_smooth: {fou['E_smooth'] / mlp['E_smooth']:.2f}x worse")
        print(f"  E_crack:  {fou['E_crack'] / mlp['E_crack']:.2f}x worse")
        print(f"  E_jump:   {fou['E_jump'] / mlp['E_jump']:.2f}x worse")
        print("=" * 70)


if __name__ == '__main__':
    main()
