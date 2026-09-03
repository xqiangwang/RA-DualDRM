import numpy as np
import matplotlib.pyplot as plt

from dolfinx import plot
import pyvista as pv
pv.start_xvfb()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from dolfinx import plot

def plot_scalar_field(u, domain, filename, cmap="jet", vmin=0.0, vmax=1.0, title=None):
    """
    用 matplotlib tripcolor 输出标量场图片（支持 pdf/png 等任意格式）。
    用 dolfinx.plot.vtk_mesh 得到 VTK 拓扑 + 坐标（与函数空间 dof 对齐）。
    """
    V = u.ufl_function_space()
    mesh = domain

    topology, cell_types, geometry = plot.vtk_mesh(V)

    tdim = mesh.topology.dim
    num_cells_local = mesh.topology.index_map(tdim).size_local
    num_dofs_local = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    top = topology.copy()
    ptr = 0
    for _ in range(num_cells_local):
        n = int(top[ptr])
        dofs_local = top[ptr+1:ptr+1+n]
        dofs_global = V.dofmap.index_map.local_to_global(dofs_local.copy())
        top[ptr+1:ptr+1+n] = dofs_global
        ptr += 1 + n

    root = 0
    global_topology = mesh.comm.gather(top[:ptr], root=root)
    global_geometry = mesh.comm.gather(geometry[:V.dofmap.index_map.size_local, :], root=root)
    global_vals = mesh.comm.gather(u.x.array[:num_dofs_local], root=root)

    if mesh.comm.rank != root:
        return

    root_geom = np.vstack(global_geometry)
    root_vals = np.concatenate(global_vals)
    root_top = np.concatenate(global_topology).astype(np.int64)

    triangles = []
    ptr = 0
    while ptr < len(root_top):
        n = int(root_top[ptr])
        vs = root_top[ptr+1:ptr+1+n]
        if n == 3:
            triangles.append([vs[0], vs[1], vs[2]])
        elif n == 4:
            triangles.append([vs[0], vs[1], vs[2]])
            triangles.append([vs[0], vs[2], vs[3]])
        else:
            raise RuntimeError(f"Unsupported cell with {n} vertices")
        ptr += 1 + n

    triangles = np.array(triangles, dtype=np.int64)

    x = root_geom[:, 0]
    y = root_geom[:, 1]

    tri = mtri.Triangulation(x, y, triangles=triangles)

    fig, ax = plt.subplots()
    tpc = ax.tripcolor(tri, root_vals, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(filename, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_displacement_field(u, domain, filename, cmap="jet", title=None):
    """
    绘制位移矢量场的大小 |u| = sqrt(u_x^2 + u_y^2)。
    """
    mesh = domain
    W = u.ufl_function_space()
    topology, cell_types, geometry = plot.vtk_mesh(W)

    tdim = mesh.topology.dim
    num_cells_local = mesh.topology.index_map(tdim).size_local
    num_dofs_local = W.dofmap.index_map.size_local * W.dofmap.index_map_bs

    top = topology.copy()
    ptr = 0
    for _ in range(num_cells_local):
        n = int(top[ptr])
        dofs_local = top[ptr+1:ptr+1+n]
        dofs_global = W.dofmap.index_map.local_to_global(dofs_local.copy())
        top[ptr+1:ptr+1+n] = dofs_global
        ptr += 1 + n

    root = 0
    global_topology = mesh.comm.gather(top[:ptr], root=root)
    global_geometry = mesh.comm.gather(geometry[:W.dofmap.index_map.size_local, :], root=root)
    global_vals = mesh.comm.gather(u.x.array[:num_dofs_local], root=root)

    if mesh.comm.rank != root:
        return

    root_geom = np.vstack(global_geometry)
    u_all = np.concatenate(global_vals)
    # W 是 Lagrange 1, (2,) 矢量空间，x.array 按 [ux0, uy0, ux1, uy1, ...] 排列
    ux = u_all[0::2]
    uy = u_all[1::2]
    umag = np.sqrt(ux**2 + uy**2)

    root_top = np.concatenate(global_topology).astype(np.int64)
    triangles = []
    ptr = 0
    while ptr < len(root_top):
        n = int(root_top[ptr])
        vs = root_top[ptr+1:ptr+1+n]
        if n == 3:
            triangles.append([vs[0], vs[1], vs[2]])
        elif n == 4:
            triangles.append([vs[0], vs[1], vs[2]])
            triangles.append([vs[0], vs[2], vs[3]])
        else:
            raise RuntimeError(f"Unsupported cell with {n} vertices")
        ptr += 1 + n

    triangles = np.array(triangles, dtype=np.int64)

    x = root_geom[:, 0]
    y = root_geom[:, 1]
    tri = mtri.Triangulation(x, y, triangles=triangles)

    fig, ax = plt.subplots()
    tpc = ax.tripcolor(tri, umag, shading="gouraud", cmap=cmap)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(filename, bbox_inches="tight", dpi=200)
    plt.close(fig)


# 保留旧名称兼容
def plot_scalar_pdf_vector(u, domain, filename, cmap="jet", vmin=0.0, vmax=1.0):
    plot_scalar_field(u, domain, filename, cmap=cmap, vmin=vmin, vmax=vmax)


def plot_force_disp(B, name, out_file):
    plt.figure()
    B_ = np.array(B)
    plt.plot(B_[:, 1], np.abs(B_[:, 0]*1e-3))
    plt.xlabel("disp")
    plt.ylabel("reaction (scaled)")
    plt.grid(True)
    plt.savefig(f"{out_file}/force_disp_{name}.png", bbox_inches="tight", dpi=200)
    np.savetxt(f"{out_file}/force_{name}.txt", B_)
    plt.close()

def distance_points_to_segment(points, x1, y1, x2, y2):
    points = np.array(points)
    AB = np.array([x2 - x1, y2 - y1])
    AB_AB = np.dot(AB, AB)
    distances = []
    for point in points:
        px, py = point
        AP = np.array([px - x1, py - y1])
        AP_AB = np.dot(AP, AB)
        t = AP_AB / AB_AB
        if t < 0:
            closest_point = np.array([x1, y1])
        elif t > 1:
            closest_point = np.array([x2, y2])
        else:
            closest_point = np.array([x1, y1]) + t * AB
        
        distance = np.linalg.norm(np.array([px, py]) - closest_point)
        distances.append(distance)
    return np.array(distances)
