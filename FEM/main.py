import ufl
import numpy as np
import ufl.constant

from dolfinx import mesh, fem, io, nls, default_scalar_type
import dolfinx.nls.petsc  # 显式导入，确保 nls.petsc 子模块可用
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_vector, create_matrix
from dolfinx.io import gmshio, XDMFFile

def read_msh_any(filename, comm, gdim=2):
    """读 msh 网格：优先 gmshio；失败（节点编号不连续/缺物理名）时用 gmsh API + mesh.create_mesh 构造"""
    try:
        domain, _, _ = gmshio.read_from_msh(filename, comm, gdim=gdim)
        return domain
    except Exception as e:
        if comm.rank == 0:
            print(f"[mesh] gmshio failed ({e}); using gmsh API fallback", flush=True)
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(filename)
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    tag2row = {int(t): i for i, t in enumerate(node_tags)}
    x = np.asarray(coords, dtype=float).reshape(-1, 3)[:, :gdim]
    tris = []
    for ent in gmsh.model.getEntities(2):
        etypes, etags, ntags = gmsh.model.mesh.getElements(ent[0], ent[1])
        for et, nt in zip(etypes, ntags):
            if et == 2:  # 三角形 (2-node line=1, 3-node triangle=2)
                tris.append(np.asarray(nt, dtype=np.int64))
    gmsh.finalize()
    if not tris:
        raise RuntimeError(f"no triangle elements in {filename}")
    tri = np.vstack(tris)
    tri_row = np.array([[tag2row[t] for t in row] for row in tri], dtype=np.int64)
    # 可能有多余的孤立节点，裁掉不在任何单元里的节点
    used = np.unique(tri_row)
    remap = {old: new for new, old in enumerate(used)}
    tri_row = np.array([[remap[t] for t in row] for row in tri_row], dtype=np.int64)
    x = x[used]
    cell = ufl.Cell("triangle")
    from ufl.element import VectorElement
    mesh_el = VectorElement("Lagrange", cell, 1)
    domain = mesh.create_mesh(comm, tri_row, x, ufl.Mesh(mesh_el))
    return domain
from mpi4py import MPI
from petsc4py import PETSc
from pathlib import Path
import time
from utils import plot_scalar_field, plot_displacement_field, plot_force_disp, distance_points_to_segment
import argparse

import matplotlib.pyplot as plt
# 原始 Miehe 格式（各向异性应力 sigma = g(d)*sigma+ + sigma-，位移方程非线性，Newton 求解）:
#mpirun -np 90 python main.py --case shear --model miehe --formulation original --out_file shear_miehe_orig --l_c 1e-3 --mesh_size 700
#mpirun -np 90 python main.py --case shear --model amor --formulation original --out_file shear_amor_orig --l_c 1e-3 --mesh_size 700
# Ambati hybrid 格式（全应力 sigma0 乘 (1-p)^2，位移方程线性）:
#mpirun -np 90 python main.py --case shear --model miehe --formulation hybrid --out_file shear_miehe_struc --l_c 1e-3 --mesh_size 700
#mpirun -np 90 python main.py --case shear --model amor --formulation hybrid --out_file shear_amor_struc --l_c 1e-3 --mesh_size 700
parser = argparse.ArgumentParser(description='2D shear benchmark test')
parser.add_argument('--case', type=str, default="tension", help='Case to run tension or shear')
parser.add_argument('--model', type=str, default="miehe", help='Model to use (energy decomposition): miehe / amor / star / none')
parser.add_argument('--formulation', type=str, default="original", help='Stress formulation: original (Miehe anisotropic stress) or hybrid (Ambati)')
parser.add_argument('--pff_model', type=str, default="", help='Phase-field model: AT2 (default) or AT1. AT1 is default for case shear_drm')
parser.add_argument('--mesh_size', type=int, default=100, help='Mesh size (built-in rectangle)')
parser.add_argument('--mesh_file', type=str, default="", help='Optional .msh file to read instead of built-in rectangle mesh')
parser.add_argument('--num_steps', type=int, default=-1, help='Override number of load steps (-1 = use case default, 0 = init only)')
parser.add_argument('--out_file', type=str, default="tension-verification", help='Output file')
parser.add_argument('--job_id', type=int, default=0, help='Job id')
parser.add_argument('--l_c', type = float , default=0.15, help='l_c value')
parser.add_argument('--newton_max_it', type=int, default=50, help='Newton max iterations for displacement solve')
parser.add_argument('--newton_rtol', type=float, default=1e-8, help='Newton relative tolerance')
parser.add_argument('--newton_crit', type=str, default="residual", help='Newton convergence criterion: residual or incremental')
parser.add_argument('--s_smooth', type=float, default=1e-6, help='Bracket smoothing width (strain units); scale with typical strain level, e.g. 1e-4 for O(0.1) strains')
parser.add_argument('--delta_T1', type=float, default=0.0, help='Override load increment before t_transition')
parser.add_argument('--delta_T2', type=float, default=0.0, help='Override load increment after t_transition')
parser.add_argument('--t_transition', type=int, default=-1, help='Override step index where delta_T switches')
parser.add_argument('--plot_every', type=int, default=0, help='Plot/XDMF output every N steps (0 = auto)')
parser.add_argument('--pc_u', type=str, default="", help='Displacement preconditioner: hypre / gamg / ilu / jacobi')
parser.add_argument('--ksp_u', type=str, default="", help='Displacement KSP: gmres / cg / preonly')
parser.add_argument('--error_tol', type=float, default=1e-4, help='Staggered iteration tolerance (absolute L2 increment norm)')
parser.add_argument('--k_res', type=float, default=0.0, help='Residual stiffness (0 = use default 1e-4)')
parser.add_argument('--seed', type=str, default="history", help='Initial crack seeding: history (narrow band, default) or gamma (AT2 optimal profile exp(-d/l0))')
parser.add_argument('--nu', type=float, default=0.0, help='Poisson ratio override (0 = case default)')
args = parser.parse_args()

start_time = time.time()
sim_case = args.case
model = args.model
formulation = args.formulation
mesh_size = args.mesh_size
out_file_arg = args.out_file
job_id = args.job_id
l_c = args.l_c

ksp = PETSc.KSP.Type.GMRES
pc = PETSc.PC.Type.HYPRE

out_file = f"./results/{out_file_arg}"
Path(out_file).mkdir(parents=True, exist_ok=True)

print(f"2D {sim_case} benchmark test, model = {model}, formulation = {formulation}, out_file = {out_file}")

results_folder = Path(out_file)
results_folder.mkdir(exist_ok=True, parents=True)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

t_transition = 500
if sim_case == "tension":
    # use rectangular mesh for faster computation; msh file is available at ./mesh/tension_mesh_no_notch_001.msh
    # mesh_address = "./mesh/tension_mesh_no_notch_001.msh"
    # domain, _, _ = gmshio.read_from_msh(mesh_address, MPI.COMM_WORLD, gdim=2)
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-0.5, -0.5]), np.array([0.5, 0.5])], [mesh_size, mesh_size], cell_type=mesh.CellType.quadrilateral)
    delta_T1 = fem.Constant(domain, 1e-5)
    delta_T2 = fem.Constant(domain, 1e-6)
    num_steps = 2000
elif sim_case == "shear":
    if args.mesh_file:
        # 局部加密网格（裂纹扩展区加密，三角形单元），如 mesh_refine_test 的 meshed_geom2.msh
        domain, _, _ = gmshio.read_from_msh(args.mesh_file, MPI.COMM_WORLD, gdim=2)
    else:
        domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-0.5, -0.5]), np.array([0.5, 0.5])], [mesh_size, mesh_size], cell_type=mesh.CellType.quadrilateral)

    delta_T1 = fem.Constant(domain, 1e-5)
    delta_T2 = fem.Constant(domain, 1e-5)
    num_steps = 2000
elif sim_case == "shear_drm" or sim_case == "tensile_drm":
    # Deep Ritz 无量纲算例使用通过命令行提供的局部加密网格。
    if not args.mesh_file:
        parser.error(f"--mesh_file is required for case '{sim_case}'")
    mesh_address = args.mesh_file
    if sim_case == "tensile_drm":
        # DRM 拉伸加载 0.025×3 到 0.075, 再 0.006×25 到 0.25 (29 步); SENT 起裂 ~0.1
        # FEM 采用固定参数版加载: 弹性段 0.005×20 到 0.1, 起裂/扩展 0.001×150 到 0.25
        dT1_val, dT2_val, n_steps, t_trans = 5e-3, 1e-3, 170, 20
    else:
        # DRM 原加载制度为 0.03×8 + 0.01×26 (34 步到 0.5)；FEM 用固定参数版:
        # 弹性段 0.005×50 到 0.25, 起裂/扩展 0.0005×1100 到 0.8
        dT1_val, dT2_val, n_steps, t_trans = 5e-3, 5e-4, 1150, 50
    domain = read_msh_any(mesh_address, MPI.COMM_WORLD, gdim=2)
    delta_T1 = fem.Constant(domain, dT1_val)
    delta_T2 = fem.Constant(domain, dT2_val)
    num_steps = n_steps
    t_transition = t_trans
elif sim_case == "lpanel":
    # DRM L型板: 矩形去掉右下象限, 凹角(0,0)自然起裂, 顶边向上拉
    # 加载: DRM 为 0.05×10 到 0.5 + 0.01 到 0.8 + 0.005 到 1.2 (81步);
    # FEM 简化: 0.01×50 到 0.5, 0.0025×280 到 1.2 (330步), 起裂~0.5 前切细步长
    mesh_address = args.mesh_file or "mesh/lpanel_meshed_geom2.msh"
    domain = read_msh_any(mesh_address, MPI.COMM_WORLD, gdim=2)
    delta_T1 = fem.Constant(domain, 1e-2)
    delta_T2 = fem.Constant(domain, 2.5e-3)
    num_steps = 330
    t_transition = 50
elif sim_case == "internal":
    if not args.mesh_file:
        parser.error("--mesh_file is required for case 'internal'")
    mesh_address = args.mesh_file
    domain, _, _ = gmshio.read_from_msh(mesh_address, MPI.COMM_WORLD, gdim=2)
    delta_T1 = fem.Constant(domain, 5e-7)
    delta_T2 = fem.Constant(domain, 5e-7)
    num_steps = 6000

if args.num_steps >= 0:
    num_steps = args.num_steps
if args.delta_T1 > 0:
    delta_T1 = fem.Constant(domain, args.delta_T1)
if args.delta_T2 > 0:
    delta_T2 = fem.Constant(domain, args.delta_T2)
if args.t_transition >= 0:
    t_transition = args.t_transition

if sim_case == "tension" or sim_case == "shear":
    G_c_ = fem.Constant(domain, 2.7)
    l_0_ = fem.Constant(domain, 4e-3)
    E = fem.Constant(domain, 210.0e3)
    nu = fem.Constant(domain, 0.3)
    top_bound = 0.5
    left_bound = -0.5
    right_bound = 0.5
    bottom_bound = -0.5
if sim_case == "shear_drm" or sim_case == "tensile_drm":
    # DRM 无量纲参数: E=1, nu=0.3, l0=0.01, Gc = w1*l0 = 1.0*0.01
    G_c_ = fem.Constant(domain, 0.01)
    l_0_ = fem.Constant(domain, 0.01)
    E = fem.Constant(domain, 1.0)
    nu = fem.Constant(domain, args.nu if args.nu > 0 else 0.3)
    top_bound = 0.5
    left_bound = -0.5
    right_bound = 0.5
    bottom_bound = -0.5
if sim_case == "lpanel":
    # L型板: nu=0.18 (DRM), 其余同无量纲
    G_c_ = fem.Constant(domain, 0.01)
    l_0_ = fem.Constant(domain, 0.01)
    E = fem.Constant(domain, 1.0)
    nu = fem.Constant(domain, args.nu if args.nu > 0 else 0.18)
    top_bound = 0.5
    left_bound = -0.5
    right_bound = 0.5
    bottom_bound = -0.5
if sim_case == "internal":
    G_c_ = fem.Constant(domain, 3e-3)
    l_0_ = fem.Constant(domain, l_c)
    E = fem.Constant(domain, 30.0e3)
    nu = fem.Constant(domain, 0.333)
    top_bound = 50.0
    left_bound = 0.0
    right_bound = 50.0
    bottom_bound = 0.0

mu = E/(2*(1+nu))
lmbda = E*nu/((1+nu)*(1-2*nu))
n = fem.Constant(domain, 3.0)
Kn = lmbda + 2 * mu / n
gamma_star = fem.Constant(domain, 5.0)

# 相场模型: AT2 或 AT1 (shear_drm 默认 AT1，与深度里兹代码一致)
pff_model = args.pff_model if args.pff_model else ("AT1" if sim_case == "shear_drm" else "AT2")
if rank == 0:
    print(f"pff_model = {pff_model}")

t_ = fem.Constant(domain, 0.0)

out_file_name = XDMFFile(domain.comm, f"{out_file}/p_unit22.xdmf", 'w')
out_file_name.write_mesh(domain)
out_file_name_u = XDMFFile(domain.comm, f"{out_file}/u_unit22.xdmf", 'w')
out_file_name_u.write_mesh(domain)

# Defining function spaces for displacement, phase and history field.
V = fem.functionspace(domain, ("Lagrange", 1,))
W = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
VV = fem.functionspace(domain, ("DG", 0,))

u, v = ufl.TrialFunction(W), ufl.TestFunction(W)
p, q = ufl.TrialFunction(V), ufl.TestFunction(V)

u_new, u_old = fem.Function(W), fem.Function(W)
p_new, H_old, p_old = fem.Function(V), fem.Function(VV), fem.Function(V)
H_init_ = fem.Function(V)

tdim = domain.topology.dim
fdim = tdim - 1

############################################ defining the boundary conditions ########################################
def top_boundary(x):
    return np.isclose(x[1], top_bound)

def left_boundary(x):
    return np.isclose(x[0], left_bound)

def right_boundary(x):
    return np.isclose(x[0], right_bound)

def bottom_boundary(x):
    if sim_case == "lpanel":
        return np.isclose(x[1], bottom_bound) & (x[0] <= 0.0)  # L型板底边只有左半
    return np.isclose(x[1], bottom_bound)

def load_boundary(x):
    # L型板加载段：内台阶边(y=0)右端 x∈[0.44,0.5]（经典 L-panel benchmark 加载点，与 DRM 一致）
    return np.isclose(x[1], 0.0) & (x[0] >= 0.44)

top_facet = mesh.locate_entities_boundary(domain, fdim, top_boundary)
top_marker = 1
top_marked_facets = np.full_like(top_facet, top_marker)

bot_facet = mesh.locate_entities_boundary(domain, fdim, bottom_boundary)
bot_marker = 2
bot_marked_facets = np.full_like(bot_facet, bot_marker)

right_facet = mesh.locate_entities_boundary(domain, fdim, right_boundary)
right_marker = 3
right_marked_facets = np.full_like(right_facet, right_marker)

left_facet = mesh.locate_entities_boundary(domain, fdim, left_boundary)
left_marker = 4
left_marked_facets = np.full_like(left_facet, left_marker)

marked_facets = np.hstack([top_facet, bot_facet, right_facet, left_facet])
marked_values = np.hstack([np.full_like(top_facet, 1), np.full_like(bot_facet, 2), np.full_like(right_facet, 3), np.full_like(left_facet, 4)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

top_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, top_facet)
top_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, top_facet)

bot_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, bot_facet)
bot_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, bot_facet)

right_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, right_facet)
right_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, right_facet)

left_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, left_facet)
left_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, left_facet)

u_bc_top = fem.Constant(domain, default_scalar_type(0.0))
u_bc_bot_ = fem.Constant(domain, default_scalar_type(0.0))
u_bc_right = fem.Constant(domain, default_scalar_type(0.0))

bc_bot_y = fem.dirichletbc(default_scalar_type(0.0), bot_y_dofs, W.sub(1))
bc_bot_x = fem.dirichletbc(default_scalar_type(0.0), bot_x_dofs, W.sub(0))
bc_top_y = fem.dirichletbc(u_bc_top, top_y_dofs, W.sub(1))
bc_left_x = fem.dirichletbc(default_scalar_type(0.0), left_x_dofs, W.sub(0))
bc_left_y = fem.dirichletbc(default_scalar_type(0.0), left_y_dofs, W.sub(1))
bc_right_x = fem.dirichletbc(default_scalar_type(0.0), right_x_dofs, W.sub(0))

if sim_case == "lpanel":
    # L型板: 底边(左半)固定; 加载段(内台阶右端) v=λ 向上, u_x 自由; 顶边自由
    load_facet = mesh.locate_entities_boundary(domain, fdim, load_boundary)
    load_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, load_facet)
    bc_load_y = fem.dirichletbc(u_bc_top, load_y_dofs, W.sub(1))
    bc = [bc_bot_y, bc_bot_x, bc_load_y]
elif sim_case == "tension" or sim_case == "tensile_drm":
    bc = [bc_bot_y, bc_bot_x, bc_top_y]
elif sim_case == "shear" or sim_case == "shear_drm":
    bc_top_x = fem.dirichletbc(u_bc_top, top_x_dofs, W.sub(0))
    bc_top_y = fem.dirichletbc(default_scalar_type(0.0), top_y_dofs, W.sub(1))
    bc = [bc_bot_y, bc_bot_x, bc_top_y, bc_top_x]
elif sim_case == "internal":
    bc_bot_y = fem.dirichletbc(u_bc_bot_, bot_y_dofs, W.sub(1))
    bc = [bc_bot_y, bc_top_y]

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2})

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lmbda*ufl.tr(epsilon(u))*ufl.Identity(2) + 2.0*mu*epsilon(u)

# 扭结光滑化: |x| -> sqrt(x^2+s^2)。Amor/star 分解的 tr(eps) 扭结在剪切主导区(tr≈0)会让
# 半光滑 Newton 活性集振荡不收敛；s 应取特征应变的 ~1e-3 倍（物理剪切算例应变~1e-3 用 1e-6，
# DRM 无量纲算例应变~0.1 用 1e-4），对物理结果影响可忽略
s_smooth = args.s_smooth
def bracket_pos(u):
    return 0.5*(u + ufl.sqrt(u**2 + s_smooth**2))

def bracket_neg(u):
    return 0.5*(u - ufl.sqrt(u**2 + s_smooth**2))

# Spectral decomposition of a symmetric 2x2 strain tensor (Miehe split)
def spectral_split(eps):
    A = ufl.variable(eps)
    I1 = ufl.tr(A)
    delta = (A[0, 0] - A[1, 1])**2 + 4 * A[0, 1] * A[1, 0] + 3.0e-16 ** 2
    eigval_1 = (I1 - ufl.sqrt(delta)) / 2
    eigval_2 = (I1 + ufl.sqrt(delta)) / 2
    eigvec_1 = ufl.diff(eigval_1, A).T
    eigvec_2 = ufl.diff(eigval_2, A).T
    eps_p = 0.5 * (eigval_1 + abs(eigval_1)) * eigvec_1 + 0.5 * (eigval_2 + abs(eigval_2)) * eigvec_2
    eps_n = 0.5 * (eigval_1 - abs(eigval_1)) * eigvec_1 + 0.5 * (eigval_2 - abs(eigval_2)) * eigvec_2
    return eps_p, eps_n

def strain_dev(eps):
    return eps - (1/3) * ufl.tr(eps) * ufl.Identity(2)

# Different energy decompositions, psi = psi_pos + psi_neg, as functions of the strain tensor
def psi_pos_m(eps):
    eps_p, _ = spectral_split(eps)
    return 0.5*lmbda*(bracket_pos(ufl.tr(eps))**2) + mu*(ufl.inner(eps_p, eps_p))

def psi_neg_m(eps):
    _, eps_n = spectral_split(eps)
    return 0.5*lmbda*(bracket_neg(ufl.tr(eps))**2) + mu*(ufl.inner(eps_n, eps_n))

def psi_pos_a(eps):
    return 0.5 * Kn * bracket_pos(ufl.tr(eps))**2 + mu * ufl.inner(strain_dev(eps), strain_dev(eps))

def psi_neg_a(eps):
    return 0.5 * Kn * bracket_neg(ufl.tr(eps))**2

# Amor 分解的 3D 平面应变自洽版本（与深度里兹代码一致）：偏量能量补 e33^2 = (tr/3)^2 项
def psi_pos_a3(eps):
    tre = ufl.tr(eps)
    return 0.5 * Kn * bracket_pos(tre)**2 + mu * (ufl.inner(strain_dev(eps), strain_dev(eps)) + (tre/3.0)**2)

def psi_neg_a3(eps):
    return psi_neg_a(eps)

def psi_pos_s(eps):
    return mu * ufl.inner(strain_dev(eps), strain_dev(eps)) + 0.5 * Kn * (bracket_pos(ufl.tr(eps))**2 - gamma_star * bracket_neg(ufl.tr(eps))**2)

def psi_neg_s(eps):
    return (1 + gamma_star) * 0.5 * Kn * bracket_neg(ufl.tr(eps))**2

def psi_total(eps):
    return 0.5*lmbda*(ufl.tr(eps)**2) + mu*ufl.inner(eps, eps)

eps_var = ufl.variable(epsilon(u_new))
if model == "miehe":
    psi_pos = psi_pos_m(eps_var)
    psi_neg = psi_neg_m(eps_var)
elif model == "amor":
    psi_pos = psi_pos_a(eps_var)
    psi_neg = psi_neg_a(eps_var)
elif model == "amor3d":
    psi_pos = psi_pos_a3(eps_var)
    psi_neg = psi_neg_a3(eps_var)
elif model == "star":
    psi_pos = psi_pos_s(eps_var)
    psi_neg = psi_neg_s(eps_var)
elif model == "none":
    # 不做拉压分解：用总弹性能驱动历史场
    psi_pos = psi_total(eps_var)
    psi_neg = 0.0 * ufl.inner(eps_var, eps_var)  # 可微的零，便于统一求导

# 本构应力 sigma± = d(psi±)/d(eps)，用 UFL 自动微分保证能量与应力严格一致
sigma_pos = ufl.diff(psi_pos, eps_var)
sigma_neg = ufl.diff(psi_neg, eps_var)

def H(u_new, H_old):
    return ufl.conditional(ufl.gt(psi_pos, H_old), psi_pos, H_old)
############################################# defining the initial cracks ########################################

def H_init(dist_list, l_0, G_c, pff_model):
    distances = np.array(dist_list)
    distances = np.min(distances, axis=0)
    mask0 = distances <= l_0.value/2
    H = np.zeros_like(distances)
    phi_c = 0.999
    if pff_model == "AT1":
        H0 = 3.0 * G_c.value / (16.0 * l_0.value * (1.0 - phi_c))
    else:
        H0 = (phi_c/(1-phi_c)) * G_c.value / (2.0 * l_0.value)
    H[mask0] = H0 * (1-(2*distances[mask0]/l_0.value))
    return H

def H_init_gamma(dist_list, l_0, G_c):
    """AT2 Gamma-最优剖面播种：p(d)=exp(-d/l0) 是 AT2 的 1D 最优裂纹剖面，
    反推 H(d) = Gc/(2l0) * p/(1-p)，相场方程解出的 p 自动为 exp 剖面，
    断裂能 = Gc*L 精确（Manav/文献标准做法）。逐节点向量化计算。
    截断：中心 p<=0.999（H 上限），带外 p<1e-4 置 H=0。"""
    distances = np.min(np.array(dist_list), axis=0)
    p_seed = np.exp(-distances / l_0.value)
    p_seed = np.clip(p_seed, 0.0, 0.999)  # 中心截断
    H = (G_c.value / (2.0 * l_0.value)) * p_seed / np.maximum(1.0 - p_seed, 1e-12)
    H[p_seed < 1e-4] = 0.0  # 带外
    return H

if sim_case == "internal":
    A_ = [[12.5, 17.5], [22.5, 22.5], [32.5, 27.5]]
    B_ = [[17.5, 22.5], [27.5, 27.5], [37.5, 32.5]]
    points = domain.geometry.x[:, :2]
    dist_list = []

    for idx in range(len(A_)):
        distances = distance_points_to_segment(points, A_[idx][0], A_[idx][1], B_[idx][0], B_[idx][1])
        dist_list.append(distances)

    H_init_.x.array[:] = H_init(dist_list, l_0_, G_c_, pff_model)
    H_old.interpolate(H_init_)

elif sim_case == "tension" or sim_case == "shear" or sim_case == "shear_drm" or sim_case == "tensile_drm":
    A_ = [[-0.5, 0.0]]
    B_ = [[0.0, 0.0]]
    points = domain.geometry.x[:, :2]
    dist_list = []

    for idx in range(len(A_)):
        distances = distance_points_to_segment(points, A_[idx][0], A_[idx][1], B_[idx][0], B_[idx][1])
        dist_list.append(distances)

    if pff_model == "AT1":
        # 照搬 DRM 逻辑：p = (1-d/(2*l0))^2 直接作为不可逆下界（等价 DRM 的罚函数 α≥hist_alpha）
        # 不再反算 H，p 从正确的 AT1 最优剖面出发，硬约束 p≥p_seed 保证不愈合
        distances = np.min(np.array(dist_list), axis=0)
        p_seed = np.where(distances < 2.0*l_0_.value, (1.0 - distances/(2.0*l_0_.value))**2, 0.0)
        p_new.x.array[:] = p_seed
        p_new.x.scatter_forward()
        p_old.x.array[:] = p_seed
        p_old.x.scatter_forward()
        # 将 p_seed 存入 H_init_ 作为不可逆下界（复用已有 Function，不再播种 H）
        H_init_.x.array[:] = p_seed
        if rank == 0:
            print(f"[init] AT1 seed crack: p_max={p_seed.max():.3f}, band=2*l0={2*l_0_.value:.3f}, E_fr≈{3*G_c_.value/8 * (1/l_0_.value) * l_0_.value:.4f}", flush=True)
    else:
        if args.seed == "gamma":
            # AT2 Gamma-最优剖面播种（DRM 原版做法）：直接播种损伤场 p=exp(-d/l0)，
            # 不可逆性由 p>=p_lb 投影保证（p_lb 每步随 p 推进，等价 DRM 罚函数）。
            # 注意：不反算 H（反算的 H 会让相场方程解偏离 exp 剖面，断裂能偏高 ~19%）。
            distances = np.min(np.array(dist_list), axis=0)
            p_seed = np.exp(-distances / l_0_.value)
            p_seed[p_seed < 1e-6] = 0.0  # 带外截断
            p_new.x.array[:] = p_seed
            p_new.x.scatter_forward()
            p_old.x.array[:] = p_seed
            p_old.x.scatter_forward()
            H_init_.x.array[:] = p_seed  # H_init_ 暂存为不可逆下界 p_lb
            H_old.x.array[:] = 0.0       # H 不播种，从零开始
            H_old.x.scatter_forward()
        else:
            H_init_.x.array[:] = H_init(dist_list, l_0_, G_c_, pff_model)
            H_old.interpolate(H_init_)

#################################### problem definition ############################################
T = fem.Constant(domain, default_scalar_type((0, 0)))

k_res = fem.Constant(domain, default_scalar_type(args.k_res if args.k_res > 0 else 1e-4))  # 残存刚度：1e-4 为默认折中（性能）；1e-6 为文献常用值（KSP 慢 5-8 倍但物理最纯）；1e-3 会偏转裂纹路径
if formulation == "hybrid":
    # Ambati hybrid 格式: 位移方程用未分解全应力 sigma0 乘 g(d) = (1-p)^2（线性问题，Newton 一步收敛）
    stress_eff = ((1.0 - p_new)**2) * sigma(u_new)
else:
    # 原始 Miehe 格式: sigma = g(d)*sigma+ + sigma-（位移方程对 u 非线性，Newton 迭代求解）
    stress_eff = ((1.0 - p_new)**2 + k_res) * sigma_pos + sigma_neg

F_u = ufl.inner(stress_eff, epsilon(v)) * dx + ufl.dot(T, v) * ds
problem_u = fem.petsc.NonlinearProblem(F_u, u_new, bcs=bc)
solver_u = nls.petsc.NewtonSolver(domain.comm, problem_u)
solver_u.rtol = args.newton_rtol
solver_u.atol = 1e-10
solver_u.max_it = args.newton_max_it
solver_u.convergence_criterion = args.newton_crit
solver_u.error_on_nonconvergence = False
if formulation != "hybrid":
    # 各向异性应力在 tr(eps)≈0（剪切主导区）处分段线性扭结，半光滑 Newton 无阻尼时活性集来回翻转难收敛；
    # 加阻尼可稳定收敛（实测: 无阻尼振荡 >50 步，0.95 阻尼约 9 步收敛）
    solver_u.relaxation_parameter = 0.95
ksp_u = solver_u.krylov_solver
ksp_u.setType(ksp)
ksp_u.getPC().setType(pc)
if args.pc_u:
    pc_u_map = {"hypre": PETSc.PC.Type.HYPRE, "gamg": PETSc.PC.Type.GAMG,
                "ilu": PETSc.PC.Type.ILU, "jacobi": PETSc.PC.Type.JACOBI}
    ksp_u.getPC().setType(pc_u_map[args.pc_u])
if args.ksp_u:
    ksp_u_map = {"gmres": PETSc.KSP.Type.GMRES, "cg": PETSc.KSP.Type.CG, "preonly": PETSc.KSP.Type.PREONLY}
    ksp_u.setType(ksp_u_map[args.ksp_u])

if pff_model == "AT1":
    # AT1: w(p)=p, c_w=8/3。平衡方程: 2*psi+(1-p) = (3Gc/8)(1/l0 - 2*l0*lap(p))
    # DRM 式驱动：用当前弹性能 psi_pos 直接驱动（不用历史场 H）；
    # 不可逆性由 p >= p_lb 投影保证（p_lb 每加载步更新，初始为最优剖面种子），
    # 与 DRM 罚函数 ReLU(-(alpha-alpha_prev)) 机制等价。eps 仅防止纯 Laplace 奇异。
    eps_reg = 1e-10 * 3.0 * G_c_ / (8.0 * l_0_)
    drive = psi_pos
    a_phi = fem.form(((2.0 * drive + eps_reg) * p * q + (0.75 * G_c_ * l_0_) * ufl.dot(ufl.grad(p), ufl.grad(q))) * dx)
    L_phi = fem.form((2.0 * drive - 3.0 * G_c_ / (8.0 * l_0_)) * q * dx)
else:
    # AT2: (2l0H/Gc + 1) p q + l0^2 grad(p).grad(q) = (2l0H/Gc) q
    E_phi = (((l_0_**2) * ufl.dot(ufl.grad(p), ufl.grad(q))) + ((2*l_0_/G_c_) * H(u_new, H_old) +1 ) * p * q )* dx - (2*l_0_/G_c_) * H(u_new, H_old) * q * dx
    a_phi = fem.form(ufl.lhs(E_phi))
    L_phi = fem.form(ufl.rhs(E_phi))
A_phi = create_matrix(a_phi)
b_phi = create_vector(L_phi)

solver_phi = PETSc.KSP().create(domain.comm)
solver_phi.setOperators(A_phi)
solver_phi.setType(ksp)
solver_phi.getPC().setType(pc)

u_l2_error = fem.form(ufl.dot(u_new - u_old, u_new - u_old)*dx)
p_l2_error = fem.form(ufl.dot(p_new - p_old, p_new - p_old)*dx)

############################ new rxn force calculation ###############################################
B_bot = []
# 非线性残差 F_u 直接给出内力，v_reac 取边界单位位移方向即得反力
v_reac = fem.Function(W)
virtual_work_form = fem.form(ufl.action(F_u, v_reac))

if sim_case == "lpanel":
    bot_dofs = fem.locate_dofs_geometrical(W, load_boundary)  # 反力在加载段上测
else:
    bot_dofs = fem.locate_dofs_geometrical(W, bottom_boundary)
u_bc_bot = fem.Function(W)
bc_bot_rxn = fem.dirichletbc(u_bc_bot, bot_dofs)
bc_rxn = [bc_bot_rxn]

def one(x):
    values = np.zeros((1, x.shape[1]))
    values[0] = 1.0
    return values

if sim_case == "tension" or sim_case == "internal" or sim_case == "tensile_drm" or sim_case == "lpanel":
    u_bc_bot.sub(1).interpolate(one)
else:
    u_bc_bot.sub(0).interpolate(one)

################################# main simulation loop ###############################################
# 弹性能输出，与所选格式自洽
if formulation == "hybrid":
    # hybrid: g(d) * 0.5*sigma0:eps
    psi_lin = 0.5 * ufl.inner(sigma(u_new), epsilon(u_new))
    E_el_form = fem.form(((1.0 - p_new)**2) * psi_lin * dx)
else:
    # original: g(d)*psi+ + psi-
    E_el_form = fem.form((((1.0 - p_new)**2 + k_res) * psi_pos + psi_neg) * dx)

# 断裂能输出
if pff_model == "AT1":
    # AT1: (3Gc/8) * (p/l0 + l0*|grad p|^2)
    E_fr_form = fem.form((3.0 * G_c_ / 8.0) * (p_new/l_0_ + l_0_ * ufl.dot(ufl.grad(p_new), ufl.grad(p_new))) * dx)
else:
    # AT2: Gc/2 * ( p^2/l0 + l0*|grad p|^2 )
    E_fr_form = fem.form((G_c_/2.0) * ((p_new**2)/l_0_ + l_0_ * ufl.dot(ufl.grad(p_new), ufl.grad(p_new))) * dx)

# num_steps==0: 仅初始化裂缝+输出，不做加载
if num_steps == 0:
    if rank == 0:
        plot_scalar_field(p_new, domain, f"{out_file}/p_0.png", cmap="jet", vmin=0.0, vmax=1.0, title="Initial crack (relaxed)")
        print(f"[init] Initial crack saved to {out_file}/p_0.png, exiting.", flush=True)
    out_file_name.close()
    out_file_name_u.close()
    import sys; sys.exit(0)

E_hist = []  # rank0 用
plot_every = args.plot_every if args.plot_every > 0 else (1 if (sim_case in ("shear_drm", "tensile_drm", "lpanel") or num_steps <= 160) else 100)

delta_T = delta_T1
error_tol = fem.Constant(domain, args.error_tol)  # 基准容差；实际判据按步长缩放（相对判据，见主循环）
error_total = fem.Constant(domain, 1.0)
H_expr = fem.Expression(ufl.conditional(ufl.gt(psi_pos, H_old), psi_pos, H_old), VV.element.interpolation_points())
t_keeper = 0
io_time_total = 0.0  # I/O（画图+写XDMF）总耗时，与求解时间分开统计
for i in range(num_steps+1):
    step_t0 = time.time()
    if rank == 0:
        print(f"Step = {i}/{num_steps}", flush=True)
    if i > t_transition:
        delta_T = delta_T2
    t_.value += delta_T.value
    t_keeper += delta_T.value
    u_bc_top.value = t_.value
    if sim_case=="internal":
        u_bc_bot_.value = -t_.value
    error_tol.value = args.error_tol  # 绝对容差（与路径正确的上午版一致）
    error_total.value = 1.0
    flag = 1
    staggered_iter = 0
    while flag:
        staggered_iter +=1
        if error_total.value < error_tol.value:
            flag = 0
            break
        n_newton, converged_u = solver_u.solve(u_new)
        u_new.x.scatter_forward()
        if rank == 0 and not converged_u:
            print(f"  [warn] step {i}, staggered {staggered_iter}: Newton not converged ({n_newton} its)", flush=True)
        if rank == 0 and staggered_iter <= 2 and i < 5:
            its = ksp_u.getIterationNumber()
            print(f"  [ksp] step {i} iter{staggered_iter}: u_ksp_its={its}", flush=True)
        
        A_phi.zeroEntries()
        assemble_matrix(A_phi, a_phi, bcs = [])
        A_phi.assemble()
        with b_phi.localForm() as loc:
            loc.set(0)
        assemble_vector(b_phi, L_phi)
        b_phi.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver_phi.solve(b_phi, p_new.x.petsc_vec)
        p_new.x.scatter_forward()
        if pff_model == "AT1" or (args.seed == "gamma" and pff_model == "AT2"):
            # 负 p 截断 + 上界 1 + 不可逆下界投影（H_init_ 存 p_lb，gamma 播种时使用）
            p_new.x.array[p_new.x.array < 0.0] = 0.0
            if args.seed == "gamma":
                np.maximum(p_new.x.array, H_init_.x.array, out=p_new.x.array)
            p_new.x.array[p_new.x.array > 1.0] = 1.0
            p_new.x.scatter_forward()
            if args.seed == "gamma":
                # 下界随交错迭代单调推进（迭代级 running max，保证交错收敛），等价 DRM 每步 α≥α_prev
                np.maximum(H_init_.x.array, p_new.x.array, out=H_init_.x.array)

        error_total.value = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(u_l2_error), op=MPI.SUM)) + np.sqrt(domain.comm.allreduce(fem.assemble_scalar(p_l2_error), op=MPI.SUM))
        if rank == 0:
            print(f"staggered_iter = {staggered_iter}, error total = {error_total.value}, newton_its = {n_newton}", flush=True)
        if staggered_iter >= 500:
            # 保险：裂纹跳变步交错可能长期不收敛，接受当前状态继续（增量已较小），避免死循环
            if rank == 0:
                print(f"  [warn] step {i}: staggered loop hit max_iter=500 (error={error_total.value:.3e}), accepting state", flush=True)
            break
        p_old.x.array[:] = p_new.x.array
        u_old.x.array[:] = u_new.x.array
        H_old.interpolate(H_expr)
    ################################################################################
    v_reac.x.petsc_vec.set(0.0)
    v_reac.x.scatter_forward()
    fem.set_bc(v_reac.x.petsc_vec, [bc_bot_rxn])
    R_bot_y= domain.comm.gather(fem.assemble_scalar(virtual_work_form), root=0)
    E_el = domain.comm.allreduce(fem.assemble_scalar(E_el_form), op=MPI.SUM)
    E_fr = domain.comm.allreduce(fem.assemble_scalar(E_fr_form), op=MPI.SUM)
    if rank == 0:
        E_hist.append([t_keeper, E_el, E_fr])
        E_hist_ = np.array(E_hist)
        disp = E_hist_[:, 0]
        Eel  = E_hist_[:, 1]
        Efr  = E_hist_[:, 2]

        plt.figure()
        plt.plot(disp, Eel, label="Elastic energy")
        plt.plot(disp, Efr, label="Fracture energy")
        plt.plot(disp, Eel + Efr, label="Total (Eel+Efr)", linestyle="--")
        plt.xlabel("disp")
        plt.ylabel("energy")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{out_file}/energy_vs_disp.png", bbox_inches="tight", dpi=200)
        np.savetxt(f"{out_file}/energy_vs_disp.txt", E_hist_, header="disp E_elastic E_fracture")
        plt.close()
    if domain.comm.rank == 0:
        B_bot.append([np.sum(R_bot_y), t_keeper])
    if i%plot_every == 0:
        io_t0 = time.time()
        out_file_name.write_function(p_new, t_keeper)
        out_file_name_u.write_function(u_new, t_keeper)
        plot_scalar_field(p_new, domain, f"{out_file}/p_{i}.png", cmap="jet", vmin=0.0, vmax=1.0, title=f"Phase-field at step {i}")
        plot_displacement_field(u_new, domain, f"{out_file}/u_{i}.png", cmap="jet", title=f"Displacement magnitude at step {i}")
        if rank == 0:
            plot_force_disp(B_bot, "bot_rxn", out_file)
        io_time_total += time.time() - io_t0
    if rank == 0:
        print(f"Step = {i}, iter = {staggered_iter}, error = {error_total.value:.2e}, wall = {time.time()-step_t0:.2f}s (io_total = {io_time_total:.1f}s)", flush=True)
################################################################################
out_file_name.close()
out_file_name_u.close()
end_time = time.time()
if rank == 0:
    print(f"Total simulation time: {end_time - start_time} seconds.")
    print(f"Total I/O (plot+XDMF) time: {io_time_total:.1f} seconds; pure solve: {end_time - start_time - io_time_total:.1f} seconds.")
    print(f"rank = {rank} done.")
