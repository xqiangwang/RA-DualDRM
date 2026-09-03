"""
Phase-field fracture with physics-informed deep learning.

This dual-network version separates the displacement and phase fields, and
augments the phase-network input with previous-step crack-state features.
The first two coordinate entries are Fourier-encoded while the extra history
features are passed through unchanged. Residual connections can be enabled
optionally for the displacement and phase networks.
"""

import atexit
import copy
import json
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gmshparser
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.tri as mtri
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Tee:
    def __init__(self, file_path):
        self.file = open(file_path, "a", encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        atexit.register(self.close)

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        pass


@dataclass
class GmshTriMesh:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float).reshape(-1)
        self.y = np.asarray(self.y, dtype=float).reshape(-1)
        self.t = np.asarray(self.t, dtype=int)
        if self.x.shape != self.y.shape:
            raise ValueError("x and y must have the same shape.")
        if self.t.ndim != 2 or self.t.shape[1] != 3:
            raise ValueError("t must have shape (n_triangles, 3).")

    @classmethod
    def from_file(cls, filename):
        import meshio

        mesh = meshio.read(str(filename))
        points = np.asarray(mesh.points, dtype=float)

        if "triangle" in mesh.cells_dict:
            triangles = np.asarray(mesh.cells_dict["triangle"], dtype=int)
        else:
            triangles = None
            for cell_block in mesh.cells:
                if cell_block.type == "triangle":
                    triangles = np.asarray(cell_block.data, dtype=int)
                    break
        if triangles is None:
            raise ValueError("Only triangle meshes are supported.")
        return cls(points[:, 0], points[:, 1], triangles)


class TriMeshRefiner:
    def __init__(self, mesh: GmshTriMesh):
        self.mesh = mesh

    def refine(self, element_list: Iterable[int], f: Optional[np.ndarray] = None):
        tri_ids = self._resolve_elements(element_list)
        marked_edges = self._marked_edges(tri_ids)

        x_new = self.mesh.x.tolist()
        y_new = self.mesh.y.tolist()
        midpoint_cache: Dict[Tuple[int, int], int] = {}
        values, value_storage, squeeze = self._prepare_values(f)
        new_triangles = []

        for tri in self.mesh.t:
            a, b, c = map(int, tri)
            mids: Dict[Tuple[int, int], int] = {}
            for edge in self._edges(a, b, c):
                if edge in marked_edges:
                    mids[edge] = self._midpoint(edge, x_new, y_new, midpoint_cache, values, value_storage)
            for child in self._split_triangle(a, b, c, mids):
                new_triangles.append(self._make_ccw(child, x_new, y_new))

        x_arr = np.asarray(x_new, dtype=float)
        y_arr = np.asarray(y_new, dtype=float)
        t_arr = np.asarray(new_triangles, dtype=int)
        if value_storage is None:
            return x_arr, y_arr, t_arr

        f_new = np.asarray(value_storage, dtype=float)
        if squeeze:
            f_new = f_new[:, 0]
        return x_arr, y_arr, t_arr, f_new

    def _resolve_elements(self, element_list: Iterable[int]) -> np.ndarray:
        values = np.asarray(list(element_list), dtype=int).reshape(-1)
        if values.size == 0:
            raise ValueError("element_list cannot be empty.")
        tri_ids = values - 1
        if np.any(tri_ids < 0) or np.any(tri_ids >= self.mesh.t.shape[0]):
            raise ValueError("element_list must be a 1-based triangle index list.")
        return np.unique(tri_ids)

    def _prepare_values(self, f: Optional[np.ndarray]):
        if f is None:
            return None, None, False
        values = np.asarray(f, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
            squeeze = True
        elif values.ndim == 2:
            squeeze = False
        else:
            raise ValueError("f must have shape (n_nodes,) or (n_nodes, n_fields).")
        if values.shape[0] != self.mesh.x.size:
            raise ValueError("f length must match the number of mesh nodes.")
        return values, values.tolist(), squeeze

    def _marked_edges(self, tri_ids: np.ndarray) -> set:
        marked = set()
        for tri_id in tri_ids:
            a, b, c = map(int, self.mesh.t[int(tri_id)])
            marked.update(self._edges(a, b, c))
        return marked

    @staticmethod
    def _edges(a: int, b: int, c: int):
        return (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a))))

    def _midpoint(self, edge, x_new, y_new, midpoint_cache, values, value_storage):
        if edge in midpoint_cache:
            return midpoint_cache[edge]
        i, j = edge
        idx = len(x_new)
        x_new.append(0.5 * (x_new[i] + x_new[j]))
        y_new.append(0.5 * (y_new[i] + y_new[j]))
        if values is not None and value_storage is not None:
            value_storage.append((0.5 * (values[i] + values[j])).tolist())
        midpoint_cache[edge] = idx
        return idx

    def _split_triangle(self, a: int, b: int, c: int, mids: Dict[Tuple[int, int], int]):
        ab = mids.get(tuple(sorted((a, b))))
        bc = mids.get(tuple(sorted((b, c))))
        ca = mids.get(tuple(sorted((c, a))))
        n = sum(mid is not None for mid in (ab, bc, ca))
        if n == 0:
            return [(a, b, c)]
        if n == 1:
            if ab is not None:
                return [(a, ab, c), (ab, b, c)]
            if bc is not None:
                return [(b, bc, a), (bc, c, a)]
            return [(c, ca, b), (ca, a, b)]
        if n == 2:
            if ab is not None and bc is not None:
                return [(b, bc, ab), (a, ab, c), (ab, bc, c)]
            if bc is not None and ca is not None:
                return [(c, ca, bc), (b, bc, a), (bc, ca, a)]
            return [(a, ab, ca), (c, ca, b), (ca, ab, b)]
        return [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]

    @staticmethod
    def _make_ccw(tri: Sequence[int], x: Sequence[float], y: Sequence[float]):
        i, j, k = tri
        area2 = (x[j] - x[i]) * (y[k] - y[i]) - (x[k] - x[i]) * (y[j] - y[i])
        if area2 < 0.0:
            return (i, k, j)
        return tuple(tri)


class TransitionTriMeshRefiner(TriMeshRefiner):
    def __init__(self, mesh: GmshTriMesh, transition_layers: int = 2):
        super().__init__(mesh)
        self.transition_layers = max(0, int(transition_layers))

    def refine(self, element_list: Iterable[int], f: Optional[np.ndarray] = None):
        tri_ids = self._resolve_elements(element_list)
        expanded = self._expand_by_edge_neighbors(tri_ids, self.transition_layers)
        return super().refine(expanded + 1, f=f)

    def _expand_by_edge_neighbors(self, tri_ids: np.ndarray, layers: int) -> np.ndarray:
        if layers <= 0 or tri_ids.size == 0:
            return np.unique(tri_ids)
        adjacency = self._build_edge_adjacency()
        active = set(int(tri_id) for tri_id in tri_ids)
        frontier = set(active)
        for _ in range(layers):
            next_frontier = set()
            for tri_id in frontier:
                next_frontier.update(adjacency[tri_id])
            next_frontier.difference_update(active)
            if not next_frontier:
                break
            active.update(next_frontier)
            frontier = next_frontier
        return np.asarray(sorted(active), dtype=int)

    def _build_edge_adjacency(self) -> List[set[int]]:
        n_triangles = int(self.mesh.t.shape[0])
        adjacency: List[set[int]] = [set() for _ in range(n_triangles)]
        edge_to_triangles: Dict[Tuple[int, int], List[int]] = {}
        for tri_id, tri in enumerate(self.mesh.t):
            a, b, c = map(int, tri)
            for edge in self._edges(a, b, c):
                edge_to_triangles.setdefault(edge, []).append(tri_id)
        for tri_ids in edge_to_triangles.values():
            if len(tri_ids) < 2:
                continue
            for tri_id in tri_ids:
                adjacency[tri_id].update(neighbor for neighbor in tri_ids if neighbor != tri_id)
        return adjacency


class TargetSizeTransitionTriMeshRefiner(TransitionTriMeshRefiner):
    def __init__(self, mesh: GmshTriMesh, target_size: float, transition_layers: int = 2):
        super().__init__(mesh, transition_layers=transition_layers)
        self.target_size = float(target_size)

    def refine(self, element_list: Iterable[int], f: Optional[np.ndarray] = None):
        tri_ids = self._resolve_elements(element_list)
        expanded = self._expand_by_edge_neighbors(tri_ids, self.transition_layers)
        max_edge_lengths = self.triangle_max_edge_lengths(self.mesh)
        oversized = expanded[max_edge_lengths[expanded] > self.target_size]
        if oversized.size == 0:
            if f is None:
                return self.mesh.x.copy(), self.mesh.y.copy(), self.mesh.t.copy()
            return self.mesh.x.copy(), self.mesh.y.copy(), self.mesh.t.copy(), np.asarray(f, dtype=float).copy()
        return TriMeshRefiner.refine(self, oversized + 1, f=f)

    @staticmethod
    def triangle_max_edge_lengths(mesh: GmshTriMesh) -> np.ndarray:
        coords = np.column_stack([mesh.x, mesh.y])
        tri_pts = coords[mesh.t]
        edge01 = np.linalg.norm(tri_pts[:, 1] - tri_pts[:, 0], axis=1)
        edge12 = np.linalg.norm(tri_pts[:, 2] - tri_pts[:, 1], axis=1)
        edge20 = np.linalg.norm(tri_pts[:, 0] - tri_pts[:, 2], axis=1)
        return np.maximum(np.maximum(edge01, edge12), edge20)


class TriMeshIO:
    @staticmethod
    def plot_field(mesh, f: np.ndarray, filename: str = None, title: str = "Field", cmap: str = "jet"):
        fig, ax = plt.subplots(figsize=(8, 6))
        triang = mtri.Triangulation(mesh.x, mesh.y, mesh.t)
        tcf = ax.tripcolor(triang, f, cmap=cmap, shading="gouraud")
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(tcf, ax=ax, label="alpha")
        fig.tight_layout()
        if filename:
            fig.savefig(filename, dpi=300)
        plt.close(fig)

    @staticmethod
    def write_msh(mesh, filename):
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as file:
            file.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
            file.write("$Nodes\n")
            file.write(f"{mesh.x.size}\n")
            for node_id, (xi, yi) in enumerate(zip(mesh.x, mesh.y), start=1):
                file.write(f"{node_id} {float(xi):.16e} {float(yi):.16e} 0.0\n")
            file.write("$EndNodes\n")
            file.write("$Elements\n")
            file.write(f"{mesh.t.shape[0]}\n")
            for elem_id, tri in enumerate(mesh.t, start=1):
                n1, n2, n3 = (int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1)
                file.write(f"{elem_id} 2 0 {n1} {n2} {n3}\n")
            file.write("$EndElements\n")

    @staticmethod
    def plot(mesh, filename, title="Triangle Mesh", show_element_ids=True, show_node_ids=False):
        fig, ax = plt.subplots(figsize=(10, 5))
        triang = mtri.Triangulation(mesh.x, mesh.y, mesh.t)
        ax.triplot(triang, color="black", linewidth=0.8)
        if show_element_ids:
            for i, tri in enumerate(mesh.t, start=1):
                center_x = mesh.x[tri].mean()
                center_y = mesh.y[tri].mean()
                ax.text(center_x, center_y, str(i), color="blue", fontsize=7, ha="center", va="center")
        if show_node_ids:
            for i, (xi, yi) in enumerate(zip(mesh.x, mesh.y), start=1):
                ax.text(xi, yi, str(i), color="darkgreen", fontsize=7, ha="left", va="bottom")
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.margins(0.05)
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        plt.close(fig)


# =============================================================================
# 1. Configuration
# =============================================================================

print("[Info] Setting up NotchedHole_pinLoading configuration...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Info] Using device: {device}")
torch.set_float32_matmul_precision("high")
print("[Info] Torch float32 matmul precision set to high.")
print(f"[Info] Torch version: {torch.__version__}")
print("[Info] Setting up model and training parameters...")

network_dict = {
    "disp_net": {
        "model_type": "MLP",
        "hidden_layers": 6,
        "neurons": 400,
        "seed": 1,
        "activation": "TrainableReLU",
        "init_coeff": 3.0,
        "use_fourier": False,
        "use_residual": True,
        "residual_frequency": 2,
    },
    "phase_net": {
        "model_type": "MLP",
        "hidden_layers": 3,
        "neurons": 200,
        "seed": 1,
        "activation": "TrainableReLU",
        "init_coeff": 3.0,
        "use_fourier": True,
        "mapping_size": 64,
        "ff_scale": 6,
        "ff_include_input": True,
        "fourier_trainable": False,
        "fourier_input_dimension": 2,
        "use_residual": False,
        "residual_frequency": 2,
    },
}

feature_dict = {
    "phase_feature_mode": "alpha",
    "damage_threshold": 0.10,
    "distance_chunk_size": 2048,
    "history_alpha_cut": 0.80,
}

optimizer_dict = {
    "weight_decay": 1e-5,
    "n_epochs_RPROP": 10000,
    "n_epochs_LBFGS": 0,
    "optim_rel_tol_pretrain": 1e-6,
    "optim_rel_tol": 5e-7,
}

training_dict = {"save_model_every_n": 500}
adaptive_mesh_dict = {
    "adaptive_refine": True,
    "refine_every_n": 1000,
    "alpha_threshold": 0.5,
    "target_size_ratio": 0.2,
    "transition_layers": 6,
    "pretrain_max_refine_calls": 1,
    "main_max_refine_calls": 8,
    "plot_refine_results": False,
    "compile_dynamic": True,
    "suppress_io_during_timing": True,
    "log_training_progress": False,
    "save_intermediate_models": False,
    "save_refined_meshes": True,
    "phase_monitor_every_n": 2500,
    "phase_monitor_start_disp": 0.8,
}
training_dict.update(adaptive_mesh_dict)
training_dict.update({"alpha_threshold": 0.6, "transition_layers": 8})

numr_dict = {"alpha_constraint": "nonsmooth", "gradient_type": "numerical"}
PFF_model_dict = {"PFF_model": "AT2", "se_split": "miehe", "tol_ir": 5e-3}
# Geometry scaling: L = 120 mm (plate height)
# Nondimensional coordinates: x_tilde = x/120, y_tilde = y/120
L_CHAR = 120.0  # characteristic length in mm
# Validation setting: use a deliberately enlarged nondimensional phase-field
# length before returning to the benchmark value 0.25 / 120.
L0_TILDE = 0.01
L0_PHYS = L0_TILDE * L_CHAR

mat_prop_dict = {"mat_E": 1.0, "mat_nu": 0.221, "w1": 1.0, "l0": L0_TILDE}

domain_extrema = torch.tensor([[0.0, 65.0 / L_CHAR], [0.0, 1.0]])
crack_dict = {
    "x_init": [0.0],
    "y_init": [65.0 / L_CHAR],
    "L_crack": [0.0],
    "angle_crack": [0.0],
}

loading_angle = torch.tensor([np.pi / 2])
# Displacement conversion for physical comparison:
# U_tilde = U / L * sqrt(E*l/Gc).  With the validation l0_tilde = 0.01,
# l = 1.2 mm and the conversion differs from the benchmark l = 0.25 mm.
disp = np.concatenate(
    (
        # np.linspace(0.0, 0.10, 11),
        np.arange(0.11, 0.2501, 0.01),
        np.arange(0.255, 0.4301, 0.005),
    ),
    axis=0,
)
disp = np.unique(np.round(disp, 12))
disp = disp[disp > 0.0]

coarse_mesh_file = "mesh/tip-refined_mesh2/notched_tip_focused.msh"
fine_mesh_file = "mesh/tip-refined_mesh2/notched_tip_focused.msh"

PATH_ROOT = Path.cwd()
model_path = None
trainedModel_path = None
intermediateModel_path = None
refinePlot_path = None
writer = None


def setup_run_artifacts():
    date = time.strftime("%Y%m%d_%H%M%S")
    model_path = PATH_ROOT / "result" / f"NotchedHole_refine_{date}"
    model_path.mkdir(parents=True, exist_ok=True)
    trained_model_path = model_path / "best_models"
    trained_model_path.mkdir(parents=True, exist_ok=True)
    intermediate_model_path = model_path / "intermediate_models"
    intermediate_model_path.mkdir(parents=True, exist_ok=True)
    refine_plot_path = model_path / "refine_plots"
    refine_plot_path.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    shutil.copy2(Path(__file__).resolve(), model_path / Path(__file__).name)
    for script_name in ("plot_result.py", "plot_dual_fourier_augments.py"):
        plot_script = script_dir / script_name
        if plot_script.exists():
            shutil.copy2(plot_script, model_path / plot_script.name)

    log_file = model_path / "run_log.txt"
    Tee(log_file)

    with open(model_path / "model_settings.txt", "w", encoding="utf-8") as file:
        file.write("--- Displacement Network Settings ---")
        for key, value in network_dict["disp_net"].items():
            file.write(f"\n{key}: {value}")
        file.write("\n--- Phase Network Settings ---")
        for key, value in network_dict["phase_net"].items():
            file.write(f"\n{key}: {value}")
        file.write("\n--- Optimizer Settings ---")
        for key, value in optimizer_dict.items():
            file.write(f"\n{key}: {value}")
        file.write("\n--- Training Settings ---")
        for key, value in training_dict.items():
            file.write(f"\n{key}: {value}")
        file.write("\n--- PFF Model Settings ---")
        for key, value in PFF_model_dict.items():
            file.write(f"\n{key}: {value}")
        file.write("\n--- Numerical Settings ---")
        for key, value in numr_dict.items():
            file.write(f"\n{key}: {value}")
        file.write("\n--- Feature Settings ---")
        for key, value in feature_dict.items():
            file.write(f"\n{key}: {value}")
        file.write("\n--- Material Properties ---")
        for key, value in mat_prop_dict.items():
            file.write(f"\n{key}: {value}")

    writer = SummaryWriter(model_path / "TBruns")
    return model_path, trained_model_path, intermediate_model_path, refine_plot_path, writer


def synchronize_if_cuda(run_device=None):
    dev = torch.device(device if run_device is None else run_device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def reset_peak_memory_stats(run_device=None):
    dev = torch.device(device if run_device is None else run_device)
    if dev.type == "cuda":
        synchronize_if_cuda(dev)
        torch.cuda.reset_peak_memory_stats(dev)


def get_memory_stats(run_device=None):
    dev = torch.device(device if run_device is None else run_device)
    if dev.type != "cuda":
        return {
            "device": str(dev),
            "cuda_available": False,
            "allocated_mib": None,
            "reserved_mib": None,
            "max_allocated_mib": None,
            "max_reserved_mib": None,
        }

    synchronize_if_cuda(dev)
    mib = 1024.0 ** 2
    return {
        "device": str(dev),
        "cuda_available": True,
        "allocated_mib": float(torch.cuda.memory_allocated(dev) / mib),
        "reserved_mib": float(torch.cuda.memory_reserved(dev) / mib),
        "max_allocated_mib": float(torch.cuda.max_memory_allocated(dev) / mib),
        "max_reserved_mib": float(torch.cuda.max_memory_reserved(dev) / mib),
    }


def count_model_parameters(module):
    raw_module = getattr(module, "_orig_mod", module)
    return {
        "total": int(sum(param.numel() for param in raw_module.parameters())),
        "trainable": int(sum(param.numel() for param in raw_module.parameters() if param.requires_grad)),
    }


def summarize_model_parameters(field_comp):
    disp_counts = count_model_parameters(field_comp.disp_net)
    phase_counts = count_model_parameters(field_comp.phase_net)
    return {
        "disp_net": disp_counts,
        "phase_net": phase_counts,
        "total": {
            "total": int(disp_counts["total"] + phase_counts["total"]),
            "trainable": int(disp_counts["trainable"] + phase_counts["trainable"]),
        },
    }


def save_parameter_summary(result_dir, field_comp):
    summary = summarize_model_parameters(field_comp)
    summary_dir = result_dir / "training_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "parameter_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    with open(result_dir / "model_settings.txt", "a", encoding="utf-8") as file:
        file.write("\n--- Model Parameter Counts ---")
        file.write(f"\ndisp_net total: {summary['disp_net']['total']}")
        file.write(f"\ndisp_net trainable: {summary['disp_net']['trainable']}")
        file.write(f"\nphase_net total: {summary['phase_net']['total']}")
        file.write(f"\nphase_net trainable: {summary['phase_net']['trainable']}")
        file.write(f"\nall total: {summary['total']['total']}")
        file.write(f"\nall trainable: {summary['total']['trainable']}\n")
    return summary


# =============================================================================
# 2. Material & Physics Models
# =============================================================================

class MaterialProperties:
    def __init__(self, mat_E, mat_nu, w1, l0):
        self.mat_E = mat_E
        self.mat_nu = mat_nu
        self.w1 = w1
        self.l0 = l0
        self.mat_lmbda = self.mat_E * self.mat_nu / (1 + self.mat_nu) / (1 - 2 * self.mat_nu)
        self.mat_mu = self.mat_E / (1 + self.mat_nu) / 2.0

    def __call__(self):
        return self.mat_lmbda, self.mat_mu, self.w1, self.l0


class PFFModel:
    def __init__(self, PFF_model="AT1", se_split="volumetric", tol_ir=5e-3):
        self.PFF_model = PFF_model
        self.se_split = se_split
        self.tol_ir = tol_ir

        if self.se_split not in ("volumetric", "miehe"):
            warnings.warn("Prescribed strain energy split is not recognized. No strain energy split will be applied.")

        if self.PFF_model not in ["AT1", "AT2"]:
            raise ValueError("PFF_model must be AT1 or AT2")

    def Edegrade(self, alpha):
        return (1 - alpha) ** 2, 2 * (alpha - 1)

    def damageFun(self, alpha):
        if self.PFF_model == "AT1":
            return alpha, 1.0, 8.0 / 3.0
        if self.PFF_model == "AT2":
            return alpha**2, 2 * alpha, 2.0
        raise ValueError("Unsupported PFF model")

    def irrPenalty(self):
        if self.PFF_model == "AT1":
            return 27 / 64 / self.tol_ir**2
        if self.PFF_model == "AT2":
            return 1.0 / self.tol_ir**2 - 1.0
        raise ValueError("Unsupported PFF model")


# =============================================================================
# 3. Neural Network
# =============================================================================

class SteepTanh(nn.Module):
    def __init__(self, coeff):
        super().__init__()
        self.coeff = coeff

    def forward(self, x):
        return nn.Tanh()(self.coeff * x)


class SteepReLU(nn.Module):
    def __init__(self, coeff):
        super().__init__()
        self.coeff = coeff

    def forward(self, x):
        return nn.ReLU()(self.coeff * x)


class TrainableTanh(nn.Module):
    def __init__(self, init_coeff):
        super().__init__()
        self.coeff = nn.Parameter(torch.tensor(init_coeff))

    def forward(self, x):
        return nn.Tanh()(self.coeff * x)


class TrainableReLU(nn.Module):
    def __init__(self, init_coeff):
        super().__init__()
        self.coeff = nn.Parameter(torch.tensor(init_coeff))

    def forward(self, x):
        return nn.ReLU()(self.coeff * x)


class FourierFeatureEmbedding(nn.Module):
    def __init__(
        self,
        input_dimension,
        mapping_size,
        scale=1.0,
        include_input=True,
        trainable=False,
        fourier_input_dimension=None,
    ):
        super().__init__()
        self.input_dimension = input_dimension
        self.mapping_size = mapping_size
        self.scale = scale
        self.include_input = include_input
        self.fourier_input_dimension = input_dimension if fourier_input_dimension is None else fourier_input_dimension

        if self.fourier_input_dimension <= 0 or self.fourier_input_dimension > self.input_dimension:
            raise ValueError("fourier_input_dimension must be in [1, input_dimension]")

        B = torch.randn(mapping_size, self.fourier_input_dimension) * scale
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    @property
    def output_dimension(self):
        raw_passthrough_dimension = self.input_dimension - self.fourier_input_dimension
        encoded_dimension = 2 * self.mapping_size + (self.fourier_input_dimension if self.include_input else 0)
        return encoded_dimension + raw_passthrough_dimension

    def forward(self, x):
        x_fourier = x[:, : self.fourier_input_dimension]
        x_passthrough = x[:, self.fourier_input_dimension :]
        x_proj = 2.0 * np.pi * x_fourier @ self.B.T
        encoded = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        if self.include_input:
            encoded = torch.cat([x_fourier, encoded], dim=-1)
        if x_passthrough.shape[1] > 0:
            encoded = torch.cat([encoded, x_passthrough], dim=-1)
        return encoded


class NeuralNet(nn.Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        n_hidden_layers,
        neurons,
        activation,
        init_coeff=1.0,
        use_fourier=False,
        mapping_size=128,
        ff_scale=1.0,
        ff_include_input=True,
        fourier_trainable=False,
        fourier_input_dimension=None,
        use_residual=False,
        residual_frequency=2,
    ):
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.n_hidden_layers = n_hidden_layers
        self.name_activation = activation
        self.init_coeff = init_coeff
        self.use_fourier = use_fourier
        self.use_residual = use_residual
        self.residual_frequency = residual_frequency

        if self.use_fourier:
            self.input_encoding = FourierFeatureEmbedding(
                input_dimension=input_dimension,
                mapping_size=mapping_size,
                scale=ff_scale,
                include_input=ff_include_input,
                trainable=fourier_trainable,
                fourier_input_dimension=fourier_input_dimension,
            )
            encoded_input_dimension = self.input_encoding.output_dimension
        else:
            self.input_encoding = nn.Identity()
            encoded_input_dimension = self.input_dimension

        self.input_layer = nn.Linear(encoded_input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.activations, self.trainable_activation = self._get_activations(activation, init_coeff, n_hidden_layers)

        if self.use_residual and self.residual_frequency < 2:
            raise ValueError("residual_frequency must be >= 2 when use_residual=True")

    def _get_activations(self, activation, init_coeff, n_hidden_layers):
        if activation == "SteepTanh":
            return SteepTanh(init_coeff), False
        if activation == "SteepReLU":
            return SteepReLU(init_coeff), False
        if activation == "TrainableTanh":
            return nn.ModuleList([TrainableTanh(init_coeff) for _ in range(n_hidden_layers)]), True
        if activation == "TrainableReLU":
            return nn.ModuleList([TrainableReLU(init_coeff) for _ in range(n_hidden_layers)]), True
        warnings.warn("Defaulting to Tanh.")
        return nn.Tanh(), False

    def _apply_activation(self, idx, x):
        if self.trainable_activation:
            return self.activations[idx](x)
        return self.activations(x)

    def forward(self, x):
        x = self.input_encoding(x)
        x = self._apply_activation(0, self.input_layer(x))

        residual_anchor = x
        for j, layer in enumerate(self.hidden_layers, start=1):
            x = self._apply_activation(j, layer(x))
            if self.use_residual and j % self.residual_frequency == 0:
                x = x + residual_anchor
                residual_anchor = x
        return self.output_layer(x)


def init_xavier(model):
    activation = model.name_activation
    init_coeff = model.init_coeff

    def init_weights(module):
        if isinstance(module, nn.Linear) and module.weight.requires_grad and module.bias.requires_grad:
            if activation in ["TrainableReLU", "SteepReLU"]:
                gain = nn.init.calculate_gain("leaky_relu", np.sqrt(init_coeff**2 - 1.0))
                nn.init.xavier_uniform_(module.weight, gain=gain)
                module.bias.data.fill_(0)
            if activation in ["TrainableTanh", "SteepTanh"]:
                gain = nn.init.calculate_gain("tanh") / init_coeff
                nn.init.xavier_uniform_(module.weight, gain=gain)
                module.bias.data.fill_(0)

    model.apply(init_weights)


def build_network(config, input_dimension, output_dimension):
    torch.manual_seed(config["seed"])
    network = NeuralNet(
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        n_hidden_layers=config["hidden_layers"],
        neurons=config["neurons"],
        activation=config["activation"],
        init_coeff=config["init_coeff"],
        use_fourier=config.get("use_fourier", False),
        mapping_size=config.get("mapping_size", 128),
        ff_scale=config.get("ff_scale", 1.0),
        ff_include_input=config.get("ff_include_input", True),
        fourier_trainable=config.get("fourier_trainable", False),
        fourier_input_dimension=config.get("fourier_input_dimension"),
        use_residual=config.get("use_residual", False),
        residual_frequency=config.get("residual_frequency", 2),
    )
    init_xavier(network)
    return network


def get_feature_dimensions(feature_dict):
    disp_input_dimension = 2
    mode = feature_dict.get("phase_feature_mode", "distance")
    if mode == "alpha":
        phase_input_dimension = 3
    elif mode == "gradient":
        phase_input_dimension = 4
    elif mode == "distance":
        phase_input_dimension = 5
    elif mode == "both":
        phase_input_dimension = 6
    else:
        raise ValueError(f"Unsupported phase_feature_mode: {mode}")
    return disp_input_dimension, phase_input_dimension


def _clean_state_dict_keys(state_dict):
    return {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}


def save_dual_checkpoint(field_comp, path):
    checkpoint = {
        "disp_net": _clean_state_dict_keys(field_comp.disp_net.state_dict()),
        "phase_net": _clean_state_dict_keys(field_comp.phase_net.state_dict()),
    }
    torch.save(checkpoint, path)


def load_dual_checkpoint(field_comp, path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    if "disp_net" not in checkpoint or "phase_net" not in checkpoint:
        raise ValueError(f"Checkpoint format not supported: {path}")
    field_comp.disp_net.load_state_dict(_clean_state_dict_keys(checkpoint["disp_net"]))
    field_comp.phase_net.load_state_dict(_clean_state_dict_keys(checkpoint["phase_net"]))


def plot_fields(triang, u_val, v_val, alpha_val, step_idx, disp_val, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    tpc = axes[0].tripcolor(triang, u_val, shading="gouraud", cmap="viridis")
    plt.colorbar(tpc, ax=axes[0])
    axes[0].set_title(r"$u_x$")
    axes[0].set_aspect("equal")
    axes[0].axis("off")

    tpc = axes[1].tripcolor(triang, v_val, shading="gouraud", cmap="viridis")
    plt.colorbar(tpc, ax=axes[1])
    axes[1].set_title(r"$u_y$")
    axes[1].set_aspect("equal")
    axes[1].axis("off")

    tpc = axes[2].tripcolor(triang, alpha_val, shading="gouraud", cmap="coolwarm", vmin=0.0, vmax=1.0)
    plt.colorbar(tpc, ax=axes[2])
    axes[2].set_title(r"$\alpha$")
    axes[2].set_aspect("equal")
    axes[2].axis("off")

    plt.suptitle(f"Step {step_idx}: displacement = {disp_val:.4f}")
    plt.tight_layout()
    plt.savefig(output_dir / f"fields_step_{step_idx:03d}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_phase_monitor(field_comp, inp, T_conn, step_idx, epoch_idx, disp_val, output_dir):
    if T_conn is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        _, _, alpha = field_comp.fieldCalculation(inp)

    x_np = inp[:, -2].detach().cpu().numpy()
    y_np = inp[:, -1].detach().cpu().numpy()
    t_np = T_conn.detach().cpu().numpy()
    alpha_np = alpha.detach().cpu().numpy().ravel()
    triang = mtri.Triangulation(x_np, y_np, t_np)

    fig, ax = plt.subplots(figsize=(6, 5))
    tpc = ax.tripcolor(triang, alpha_np, shading="gouraud", cmap="coolwarm", vmin=0.0, vmax=1.0)
    plt.colorbar(tpc, ax=ax)
    ax.set_title(f"Step {step_idx}, epoch {epoch_idx}, disp = {disp_val:.4f}")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / f"alpha_step_{step_idx:03d}_epoch_{epoch_idx:05d}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        output_dir / f"alpha_step_{step_idx:03d}_epoch_{epoch_idx:05d}.npz",
        x=x_np,
        y=y_np,
        triangles=t_np,
        alpha=alpha_np,
        displacement=float(disp_val),
        step=int(step_idx),
        epoch=int(epoch_idx),
    )


def plot_energy_curve(disps, energies, output_dir):
    energies = np.asarray(energies, dtype=float)
    if energies.size == 0:
        return
    E_el, E_d, E_h = energies[:, 0], energies[:, 1], energies[:, 2]
    E_tot = E_el + E_d + E_h

    plt.figure(figsize=(8, 6))
    plt.plot(disps, E_el, "b--", label=r"$\mathcal{E}_{el}$")
    plt.plot(disps, E_d, "r--", label=r"$\mathcal{E}_{d}$")
    plt.plot(disps, E_h, "g--", label=r"$\mathcal{E}_{hist}$")
    plt.plot(disps, E_tot, "k-", linewidth=2, label=r"$\mathcal{E}_{total}$")
    plt.xlabel("Displacement")
    plt.ylabel("Energy")
    plt.title("Energy Evolution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "energy_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()


def write_recorded_energy_history(result_dir, disp_rows, energy_rows):
    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    disp_array = np.asarray(disp_rows, dtype=float)
    energy_array = np.asarray(energy_rows, dtype=float)
    np.save(plot_dir / "displacement_history.npy", disp_array)
    np.save(plot_dir / "energy_history.npy", energy_array)
    if energy_array.size:
        total_energy = np.sum(energy_array[:, :3], axis=1)
        log_total_energy = np.log10(np.maximum(total_energy, np.finfo(float).tiny))
        loss_array = np.column_stack((disp_array, energy_array[:, :3], total_energy, log_total_energy))
        np.save(plot_dir / "energy_loss_history.npy", loss_array)
        plot_energy_curve(disp_array, energy_array, plot_dir)


def save_current_step_result(
    field_comp,
    inp,
    T_conn,
    area_T,
    alpha_prev,
    matprop,
    pffmodel,
    step_idx,
    disp_val,
    result_dir,
):
    plot_dir = result_dir / "plots"
    data_dir = result_dir / "step_fields"
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    was_training = field_comp.training
    field_comp.eval()

    inp_eval = inp
    if T_conn is None:
        inp_eval = inp.detach().clone().requires_grad_(True)

    field_comp.set_phase_state(inp_eval, alpha_prev, T_conn, area_T, matprop.l0)
    mesh_cache = build_mesh_cache(inp_eval, area_T, T_conn)
    with torch.enable_grad():
        u, v, alpha = field_comp.fieldCalculation(inp_eval)
        E_el, E_d, E_h = compute_energy(
            inp_eval, u, v, alpha, alpha_prev, matprop, pffmodel, area_T, T_conn, mesh_cache
        )

    x_np = inp_eval[:, -2].detach().cpu().numpy()
    y_np = inp_eval[:, -1].detach().cpu().numpy()
    t_np = T_conn.detach().cpu().numpy() if T_conn is not None else None
    area_np = area_T.detach().cpu().numpy()
    u_np = u.detach().cpu().numpy().ravel()
    v_np = v.detach().cpu().numpy().ravel()
    alpha_np = alpha.detach().cpu().numpy().ravel()
    alpha_prev_np = alpha_prev.detach().cpu().numpy().ravel()
    energy_row = np.asarray([E_el.item(), E_d.item(), E_h.item()], dtype=float)
    total_energy = float(np.sum(energy_row))
    log10_total_energy = float(np.log10(max(total_energy, np.finfo(float).tiny)))

    if t_np is not None:
        triang = mtri.Triangulation(x_np, y_np, t_np)
        plot_fields(triang, u_np, v_np, alpha_np, step_idx, float(disp_val), plot_dir)

    np.savez_compressed(
        data_dir / f"step_{step_idx:03d}_fields.npz",
        x=x_np,
        y=y_np,
        triangles=t_np,
        area=area_np,
        u=u_np,
        v=v_np,
        alpha=alpha_np,
        alpha_prev=alpha_prev_np,
        energy=energy_row,
        total_energy=total_energy,
        log10_total_energy=log10_total_energy,
        displacement=float(disp_val),
        step=int(step_idx),
    )
    np.save(data_dir / "final_phase.npy", alpha_np)
    np.savez_compressed(
        data_dir / "final_fields.npz",
        x=x_np,
        y=y_np,
        triangles=t_np,
        area=area_np,
        u=u_np,
        v=v_np,
        alpha=alpha_np,
        alpha_prev=alpha_prev_np,
        energy=energy_row,
        total_energy=total_energy,
        log10_total_energy=log10_total_energy,
        displacement=float(disp_val),
        step=int(step_idx),
    )

    if was_training:
        field_comp.train()

    return alpha.detach(), energy_row


def plot_loss_curve(loss_values, output_path, title):
    loss_values = np.asarray(loss_values, dtype=float)
    if loss_values.size == 0:
        return

    epochs = np.arange(1, loss_values.size + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_values, color="tab:blue", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel(r"$\log_{10}(\mathcal{L})$")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()


def plot_step_time_curve(step_ids, step_times_minutes, output_path):
    if len(step_ids) == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(step_ids, step_times_minutes, "o-", color="tab:orange", linewidth=1.5, markersize=4)
    plt.xlabel("Load Step")
    plt.ylabel("Training Time (min)")
    plt.title("Training Time Per Load Step")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()


def save_training_summary(
    result_dir,
    pretrain_time_minutes,
    step_time_records,
    total_compute_time_minutes,
    refine_records=None,
    memory_records=None,
    wall_time_minutes=None,
):
    summary_dir = result_dir / "training_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if refine_records is None:
        refine_records = []
    if memory_records is None:
        memory_records = []

    total_refine_time = float(sum(record.get("refine_time_minutes", 0.0) for record in refine_records))
    timing_payload = {
        "pretrain_compute_time_minutes": float(pretrain_time_minutes),
        "step_compute_time_minutes": {str(step): float(t_minutes) for step, t_minutes in step_time_records},
        "total_compute_time_minutes": float(total_compute_time_minutes),
        "total_wall_time_minutes": None if wall_time_minutes is None else float(wall_time_minutes),
        "refinement_count": len(refine_records),
        "total_refinement_time_minutes": total_refine_time,
        "memory_records": memory_records,
    }

    with open(summary_dir / "timing_summary.json", "w", encoding="utf-8") as file:
        json.dump(timing_payload, file, indent=2)

    with open(summary_dir / "timing_summary.txt", "w", encoding="utf-8") as file:
        file.write(f"Pre-training compute time (min): {pretrain_time_minutes:.6f}\n")
        for step, t_minutes in step_time_records:
            file.write(f"Step {step} compute time (min): {t_minutes:.6f}\n")
        file.write(f"Refinement count: {len(refine_records)}\n")
        file.write(f"Total refinement time (min): {total_refine_time:.6f}\n")
        file.write(f"Total compute time (min): {total_compute_time_minutes:.6f}\n")
        if wall_time_minutes is not None:
            file.write(f"Total wall time including output (min): {wall_time_minutes:.6f}\n")

    if step_time_records:
        step_array = np.asarray([[step, t_minutes] for step, t_minutes in step_time_records], dtype=float)
        np.save(summary_dir / "step_time_minutes.npy", step_array)
        plot_step_time_curve(
            step_array[:, 0].astype(int),
            step_array[:, 1],
            summary_dir / "step_training_time.png",
        )

    if memory_records:
        with open(summary_dir / "memory_summary.json", "w", encoding="utf-8") as file:
            json.dump(memory_records, file, indent=2)
        rows = []
        for record in memory_records:
            rows.append([
                record.get("load_step", np.nan),
                record.get("displacement", np.nan),
                record.get("compute_time_minutes", np.nan),
                record.get("allocated_mib", np.nan) if record.get("allocated_mib") is not None else np.nan,
                record.get("reserved_mib", np.nan) if record.get("reserved_mib") is not None else np.nan,
                record.get("max_allocated_mib", np.nan) if record.get("max_allocated_mib") is not None else np.nan,
                record.get("max_reserved_mib", np.nan) if record.get("max_reserved_mib") is not None else np.nan,
                record.get("refinement_time_minutes", np.nan),
            ])
        np.save(summary_dir / "step_memory_mib.npy", np.asarray(rows, dtype=float))




def queue_refined_mesh_output(refine_cfg, record, refined_x, refined_y, refined_t, refined_alpha, refined_hist_alpha):
    if not (refine_cfg.get("save_refined_meshes", True) or refine_cfg.get("plot_refine_results", False)):
        return
    pending_outputs = refine_cfg.setdefault("_pending_refined_mesh_outputs", [])
    pending_outputs.append({
        "record": record,
        "x": np.asarray(refined_x, dtype=float).copy(),
        "y": np.asarray(refined_y, dtype=float).copy(),
        "triangles": np.asarray(refined_t, dtype=int).copy(),
        "alpha": np.asarray(refined_alpha, dtype=float).copy(),
        "hist_alpha": np.asarray(refined_hist_alpha, dtype=float).copy(),
    })


def flush_refined_mesh_outputs(result_dir, refine_cfg):
    pending_outputs = refine_cfg.get("_pending_refined_mesh_outputs", [])
    if not pending_outputs:
        return

    mesh_dir = result_dir / "refined_meshes"
    plot_dir = result_dir / "refine_plots"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    while pending_outputs:
        output = pending_outputs.pop(0)
        record = output["record"]
        refine_id = int(record["refine_id"])
        mesh = GmshTriMesh(x=output["x"], y=output["y"], t=output["triangles"])

        msh_path = mesh_dir / f"refined_mesh_{refine_id:03d}.msh"
        npz_path = mesh_dir / f"refined_mesh_{refine_id:03d}.npz"
        mesh_plot_path = plot_dir / f"mesh_refine_{refine_id:03d}.png"
        alpha_plot_path = plot_dir / f"alpha_refine_{refine_id:03d}.png"

        if refine_cfg.get("save_refined_meshes", True):
            TriMeshIO.write_msh(mesh, msh_path)
            np.savez_compressed(
                npz_path,
                x=output["x"],
                y=output["y"],
                triangles=output["triangles"],
                alpha=output["alpha"],
                hist_alpha=output["hist_alpha"],
                stage=record["stage"],
                load_step=record["load_step"],
                epoch=record["epoch"],
                refine_id=refine_id,
            )
            record["mesh_msh_file"] = str(msh_path.relative_to(result_dir))
            record["mesh_npz_file"] = str(npz_path.relative_to(result_dir))

        TriMeshIO.plot(
            mesh,
            filename=mesh_plot_path,
            title=f"Refined Mesh #{refine_id}",
            show_element_ids=False,
            show_node_ids=False,
        )
        record["mesh_plot_file"] = str(mesh_plot_path.relative_to(result_dir))

        if refine_cfg.get("plot_refine_results", False):
            TriMeshIO.plot_field(
                mesh,
                output["alpha"],
                filename=alpha_plot_path,
                title=f"Phase Field #{refine_id}",
                cmap="jet",
            )
            record["alpha_plot_file"] = str(alpha_plot_path.relative_to(result_dir))


def save_refinement_summary(result_dir, refine_records):
    summary_dir = result_dir / "training_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_dir / "refinement_summary.json", "w", encoding="utf-8") as file:
        json.dump(refine_records, file, indent=2)

    with open(summary_dir / "refinement_summary.txt", "w", encoding="utf-8") as file:
        for record in refine_records:
            file.write(
                "refine_id={refine_id}, stage={stage}, load_step={load_step}, epoch={epoch}, "
                "time_min={refine_time_minutes:.6f}, core_time_min={refine_core_time_minutes:.6f}, "
                "nodes {old_nodes}->{new_nodes}, elements {old_elements}->{new_elements}, "
                "marked={marked_elements}\n".format(**record)
            )


def generate_postprocess_plots(result_dir, checkpoint_dir):
    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_device = "cpu"
    pffmodel, matprop, disp_net, phase_net = construct_model_wrapper(
        PFF_model_dict, mat_prop_dict, network_dict, domain_extrema, plot_device
    )
    field_comp = FieldComputation(
        disp_net=disp_net.to(plot_device),
        phase_net=phase_net.to(plot_device),
        domain_extrema=domain_extrema.to(plot_device),
        lmbda=torch.tensor([0.0], device=plot_device),
        theta=loading_angle.to(plot_device),
        feature_dict=feature_dict,
        alpha_constraint=numr_dict["alpha_constraint"],
    )
    field_comp.eval()

    X, Y, T_conn_np, _ = parse_mesh(filename=fine_mesh_file, gradient_type=numr_dict["gradient_type"])
    X = X / L_CHAR
    Y = Y / L_CHAR
    inp = torch.from_numpy(np.column_stack((X, Y))).to(torch.float32).to(plot_device)
    T_conn = torch.from_numpy(T_conn_np).to(torch.long).to(plot_device)
    area_T = torch.from_numpy(
        0.5
        * (
            X[T_conn_np[:, 0]] * (Y[T_conn_np[:, 1]] - Y[T_conn_np[:, 2]])
            + X[T_conn_np[:, 1]] * (Y[T_conn_np[:, 2]] - Y[T_conn_np[:, 0]])
            + X[T_conn_np[:, 2]] * (Y[T_conn_np[:, 0]] - Y[T_conn_np[:, 1]])
        )
    ).to(torch.float32).to(plot_device)
    triang = mtri.Triangulation(X, Y, T_conn_np)

    alpha_prev = hist_alpha_init(inp, matprop, pffmodel, crack_dict).detach()
    energy_rows = []
    disp_rows = []

    for step_idx, disp_i in enumerate(disp):
        checkpoint_path = checkpoint_dir / f"trained_2NN_{step_idx}.pt"
        if not checkpoint_path.exists():
            print(f"[Plot] Skip step {step_idx}: checkpoint not found at {checkpoint_path}")
            continue

        field_comp.lmbda = torch.tensor([disp_i], device=plot_device)
        field_comp.set_phase_state(inp, alpha_prev, T_conn, area_T, matprop.l0)
        load_dual_checkpoint(field_comp, checkpoint_path, device=plot_device)

        with torch.no_grad():
            u, v, alpha = field_comp.fieldCalculation(inp)
            E_el, E_d, E_h = compute_energy(inp, u, v, alpha, alpha_prev, matprop, pffmodel, area_T, T_conn)

        disp_rows.append(float(disp_i))
        energy_rows.append([E_el.item(), E_d.item(), E_h.item()])
        plot_fields(
            triang,
            u.detach().cpu().numpy().ravel(),
            v.detach().cpu().numpy().ravel(),
            alpha.detach().cpu().numpy().ravel(),
            step_idx,
            float(disp_i),
            plot_dir,
        )
        alpha_prev = alpha.detach()

    if disp_rows:
        np.save(plot_dir / "displacement_history.npy", np.asarray(disp_rows))
        np.save(plot_dir / "energy_history.npy", np.asarray(energy_rows))
        plot_energy_curve(disp_rows, energy_rows, plot_dir)
        print(f"[Plot] Saved postprocess figures to: {plot_dir}")
    else:
        print("[Plot] No step checkpoints were found. Postprocess figures were not generated.")


# =============================================================================
# 4. Utils & Mesh Parsing
# =============================================================================

class DistanceFunction:
    def __init__(self, x_init, y_init, theta, L, d0, order=2):
        self.x_init = x_init
        self.y_init = y_init
        self.theta = theta
        self.L = L
        self.d0 = d0
        self.order = order

    def __call__(self, inp):
        L = torch.tensor([self.L], device=inp.device)
        d0 = torch.tensor([self.d0], device=inp.device)
        theta = torch.tensor([self.theta], device=inp.device)
        input_c = torch.clone(inp)
        input_c[:, -2:] = input_c[:, -2:] - torch.tensor([self.x_init, self.y_init], device=inp.device)
        Rt = torch.tensor(
            [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]],
            device=inp.device,
        )
        input_c[:, -2:] = torch.matmul(input_c[:, -2:], Rt)
        x = input_c[:, -2]
        y = input_c[:, -1]

        term1 = (1 - abs(y) / d0) ** 2
        term2 = (1 - torch.sqrt((x - L) ** 2 + y**2) / d0) ** 2
        term3 = (1 - torch.sqrt(x**2 + y**2) / d0) ** 2

        dist_fn_p1 = (
            nn.ReLU()(x * (L - x)) / (abs(x * (L - x)) + 1e-16)
            * nn.ReLU()(d0 - abs(y)) / (abs(d0 - abs(y)) + 1e-16)
            * term1
        )
        dist_fn_p2 = (
            nn.ReLU()(x - L) / (abs(x - L) + 1e-16)
            * nn.ReLU()(d0**2 - ((x - L) ** 2 + y**2)) / (abs(d0**2 - ((x - L) ** 2 + y**2)) + 1e-16)
            * term2
        )
        dist_fn_p3 = (
            nn.ReLU()(-x) / (abs(x) + 1e-16)
            * nn.ReLU()(d0**2 - (x**2 + y**2)) / (abs(d0**2 - (x**2 + y**2)) + 1e-16)
            * term3
        )

        return dist_fn_p1 + dist_fn_p2 + dist_fn_p3


def hist_alpha_init(inp, matprop, pffmodel, crack_dict):
    hist_alpha = torch.zeros((inp.shape[0],), device=inp.device)
    if crack_dict["L_crack"][0] > 0:
        l0 = matprop.l0
        for j, L_crack in enumerate(crack_dict["L_crack"]):
            Lc = torch.tensor([L_crack], device=inp.device)
            theta = torch.tensor([crack_dict["angle_crack"][j]], device=inp.device)
            input_c = torch.clone(inp)
            input_c[:, -2:] = input_c[:, -2:] - torch.tensor(
                [crack_dict["x_init"][j], crack_dict["y_init"][j]], device=inp.device
            )
            Rt = torch.tensor(
                [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]],
                device=inp.device,
            )
            input_c[:, -2:] = torch.matmul(input_c[:, -2:], Rt)
            x = input_c[:, -2]
            y = input_c[:, -1]

            if pffmodel.PFF_model == "AT1":
                hist_alpha_p1 = (
                    nn.ReLU()(x * (Lc - x)) / (abs(x * (Lc - x)) + 1e-16)
                    * nn.ReLU()(2 * l0 - abs(y)) / (abs(2 * l0 - abs(y)) + 1e-16)
                    * (1 - abs(y) / l0 / 2) ** 2
                )
                hist_alpha_p2 = (
                    nn.ReLU()(x - Lc + 1e-16) / (abs(x - Lc) + 1e-16)
                    * nn.ReLU()(2 * l0 - torch.sqrt((x - Lc) ** 2 + y**2) + 1e-16)
                    / (abs(2 * l0 - torch.sqrt((x - Lc) ** 2 + y**2)) + 1e-16)
                    * (1 - torch.sqrt((x - Lc) ** 2 + y**2) / l0 / 2) ** 2
                )
                hist_alpha_p3 = (
                    nn.ReLU()(-x + 1e-16) / (abs(x) + 1e-16)
                    * nn.ReLU()(2 * l0 - torch.sqrt(x**2 + y**2) + 1e-16)
                    / (abs(2 * l0 - torch.sqrt(x**2 + y**2)) + 1e-16)
                    * (1 - torch.sqrt(x**2 + y**2) / l0 / 2) ** 2
                )
            elif pffmodel.PFF_model == "AT2":
                hist_alpha_p1 = nn.ReLU()(x * (Lc - x)) / (abs(x * (Lc - x)) + 1e-16) * torch.exp(-abs(y) / l0)
                hist_alpha_p2 = nn.ReLU()(x - Lc + 1e-16) / (abs(x - Lc) + 1e-16) * torch.exp(
                    -torch.sqrt((x - Lc) ** 2 + y**2) / l0
                )
                hist_alpha_p3 = nn.ReLU()(-x + 1e-16) / (abs(x) + 1e-16) * torch.exp(
                    -torch.sqrt(x**2 + y**2) / l0
                )
            else:
                raise ValueError("Unsupported PFF model")

            hist_alpha = hist_alpha + hist_alpha_p1 + hist_alpha_p2 + hist_alpha_p3
    hist_alpha = torch.clamp(hist_alpha, min=0.0)
    hist_alpha_max = torch.max(hist_alpha)
    if hist_alpha_max > 0:
        hist_alpha = torch.clamp(hist_alpha / hist_alpha_max, max=1.0)
    return hist_alpha


def parse_mesh(filename="meshed_geom.msh", gradient_type="numerical"):
    try:
        mesh = gmshparser.parse(filename)
        X, Y, T = gmshparser.helpers.get_triangles(mesh)
        assert T != [], "Discretization must have only triangular elements"
        X, Y, T = np.asarray(X), np.asarray(Y), np.asarray(T)
    except Exception as exc:
        print(f"[Mesh] gmshparser failed for {filename}, fallback to meshio reader: {exc}")
        mesh = GmshTriMesh.from_file(filename)
        X, Y, T = mesh.x.copy(), mesh.y.copy(), mesh.t.copy()

    area = X[T[:, 0]] * (Y[T[:, 1]] - Y[T[:, 2]]) + X[T[:, 1]] * (Y[T[:, 2]] - Y[T[:, 0]]) + X[T[:, 2]] * (
        Y[T[:, 0]] - Y[T[:, 1]]
    )
    area = 0.5 * area
    if gradient_type == "autodiff":
        X = (X[T[:, 0]] + X[T[:, 1]] + X[T[:, 2]]) / 3
        Y = (Y[T[:, 0]] + Y[T[:, 1]] + Y[T[:, 2]]) / 3
    return X, Y, T, area


def compute_triangle_area(X, Y, T):
    area = (
        X[T[:, 0]] * (Y[T[:, 1]] - Y[T[:, 2]])
        + X[T[:, 1]] * (Y[T[:, 2]] - Y[T[:, 0]])
        + X[T[:, 2]] * (Y[T[:, 0]] - Y[T[:, 1]])
    )
    return 0.5 * area


def build_mesh_cache(inp, area_elem, T_conn):
    if T_conn is None:
        return None

    t0 = T_conn[:, 0]
    t1 = T_conn[:, 1]
    t2 = T_conn[:, 2]
    x0 = inp[t0, -2]
    x1 = inp[t1, -2]
    x2 = inp[t2, -2]
    y0 = inp[t0, -1]
    y1 = inp[t1, -1]
    y2 = inp[t2, -1]
    inv_two_area = 0.5 / area_elem

    return {
        "t0": t0,
        "t1": t1,
        "t2": t2,
        "gx0": (y1 - y2) * inv_two_area,
        "gx1": (y2 - y0) * inv_two_area,
        "gx2": (y0 - y1) * inv_two_area,
        "gy0": (x2 - x1) * inv_two_area,
        "gy1": (x0 - x2) * inv_two_area,
        "gy2": (x1 - x0) * inv_two_area,
    }


def compute_distance_to_damage(inp, alpha_prev, l0, alpha_threshold=0.10, chunk_size=2048):
    damaged_mask = alpha_prev >= alpha_threshold
    if not torch.any(damaged_mask):
        damaged_mask[torch.argmax(alpha_prev)] = True

    coords = inp[:, :2]
    damaged_coords = coords[damaged_mask]
    min_distances = []
    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        distances = torch.cdist(coords[start:end], damaged_coords)
        min_distances.append(torch.min(distances, dim=1).values)

    d = torch.cat(min_distances, dim=0)
    d_tilde = d / (l0 + 1e-12)
    psi = torch.exp(-d_tilde)
    return d_tilde, psi


def compute_nodal_grad_alpha(inp, alpha_prev, area_T, T_conn, l0):
    if T_conn is None:
        grad_alpha = torch.autograd.grad(alpha_prev.sum(), inp, create_graph=False, allow_unused=False)[0]
        grad_mag = torch.sqrt(grad_alpha[:, 0] ** 2 + grad_alpha[:, 1] ** 2 + 1e-12)
        return l0 * grad_mag

    grad_alpha_x, grad_alpha_y = field_grads(inp, alpha_prev, area_T, T_conn)
    grad_x_nodal = torch.zeros_like(alpha_prev)
    grad_y_nodal = torch.zeros_like(alpha_prev)
    counts = torch.zeros_like(alpha_prev)
    elem_ones = torch.ones(T_conn.shape[0], device=inp.device, dtype=inp.dtype)

    for local_node in range(T_conn.shape[1]):
        node_ids = T_conn[:, local_node]
        grad_x_nodal.index_add_(0, node_ids, grad_alpha_x)
        grad_y_nodal.index_add_(0, node_ids, grad_alpha_y)
        counts.index_add_(0, node_ids, elem_ones)

    counts = torch.clamp(counts, min=1.0)
    grad_mag = torch.sqrt((grad_x_nodal / counts) ** 2 + (grad_y_nodal / counts) ** 2 + 1e-12)
    return l0 * grad_mag


# =============================================================================
# 5. Energy Computation
# =============================================================================

def field_grads(inp, field, area_elem, T=None, mesh_cache=None):
    if T is None:
        grad_field = torch.autograd.grad(field.sum(), inp, create_graph=True)[0]
        grad_x = grad_field[:, 0]
        grad_y = grad_field[:, 1]
    elif mesh_cache is not None:
        grad_x = (
            mesh_cache["gx0"] * field[mesh_cache["t0"]]
            + mesh_cache["gx1"] * field[mesh_cache["t1"]]
            + mesh_cache["gx2"] * field[mesh_cache["t2"]]
        )
        grad_y = (
            mesh_cache["gy0"] * field[mesh_cache["t0"]]
            + mesh_cache["gy1"] * field[mesh_cache["t1"]]
            + mesh_cache["gy2"] * field[mesh_cache["t2"]]
        )
    else:
        grad_x = (
            (inp[T[:, 1], -1] - inp[T[:, 2], -1]) * field[T[:, 0]]
            + (inp[T[:, 2], -1] - inp[T[:, 0], -1]) * field[T[:, 1]]
            + (inp[T[:, 0], -1] - inp[T[:, 1], -1]) * field[T[:, 2]]
        )
        grad_y = (
            (inp[T[:, 2], -2] - inp[T[:, 1], -2]) * field[T[:, 0]]
            + (inp[T[:, 0], -2] - inp[T[:, 2], -2]) * field[T[:, 1]]
            + (inp[T[:, 1], -2] - inp[T[:, 0], -2]) * field[T[:, 2]]
        )
        grad_x = grad_x / area_elem / 2
        grad_y = grad_y / area_elem / 2
    return grad_x, grad_y


def gradients(inp, u, v, alpha, area_elem, T_conn=None, mesh_cache=None):
    grad_u_x, grad_u_y = field_grads(inp, u, area_elem, T_conn, mesh_cache)
    grad_v_x, grad_v_y = field_grads(inp, v, area_elem, T_conn, mesh_cache)
    grad_alpha_x, grad_alpha_y = field_grads(inp, alpha, area_elem, T_conn, mesh_cache)
    strain_11 = grad_u_x
    strain_22 = grad_v_y
    strain_12 = 0.5 * (grad_u_y + grad_v_x)
    return strain_11, strain_22, strain_12, grad_alpha_x, grad_alpha_y


def strain_energy_with_split(strain_11, strain_22, strain_12, alpha, matprop, pffmodel):
    fun_EDegrade, _ = pffmodel.Edegrade(alpha)
    if pffmodel.se_split == "miehe":
        trace = strain_11 + strain_22
        diff = strain_11 - strain_22
        radius = torch.sqrt(0.25 * diff**2 + strain_12**2 + 1e-16)
        eps_1 = 0.5 * trace + radius
        eps_2 = 0.5 * trace - radius

        trace_p = nn.ReLU()(trace)
        trace_n = -nn.ReLU()(-trace)
        eps_1_p = nn.ReLU()(eps_1)
        eps_2_p = nn.ReLU()(eps_2)
        eps_1_n = -nn.ReLU()(-eps_1)
        eps_2_n = -nn.ReLU()(-eps_2)

        E_el_p = 0.5 * matprop.mat_lmbda * trace_p**2 + matprop.mat_mu * (eps_1_p**2 + eps_2_p**2)
        E_el_n = 0.5 * matprop.mat_lmbda * trace_n**2 + matprop.mat_mu * (eps_1_n**2 + eps_2_n**2)
        E_el = fun_EDegrade * E_el_p + E_el_n
    elif pffmodel.se_split == "volumetric":
        mat_K = matprop.mat_lmbda + 2.0 / 3.0 * matprop.mat_mu
        strain_k = (strain_11 + strain_22) / 3.0
        strain_deviatoric_11 = strain_11 - strain_k
        strain_deviatoric_22 = strain_22 - strain_k
        strain_deviatoric_33 = 0 - strain_k
        E_elV_p = 0.5 * mat_K * (nn.ReLU()(3.0 * strain_k)) ** 2
        E_elV_n = 0.5 * mat_K * (-nn.ReLU()(-3.0 * strain_k)) ** 2
        E_el_dev = matprop.mat_mu * (
            strain_deviatoric_11**2 + strain_deviatoric_22**2 + strain_deviatoric_33**2 + 2 * strain_12**2
        )
        E_el_p = E_elV_p + E_el_dev
        E_el = fun_EDegrade * E_el_p + E_elV_n
    else:
        E_el = fun_EDegrade * (
            0.5 * matprop.mat_lmbda * (strain_11 + strain_22) ** 2
            + matprop.mat_mu * (strain_11**2 + strain_22**2 + 2 * strain_12**2)
        )
    return E_el, None

def compute_energy_per_elem(inp, u, v, alpha, hist_alpha, matprop, pffmodel, area_elem, T_conn=None, mesh_cache=None):
    strain_11, strain_22, strain_12, grad_alpha_x, grad_alpha_y = gradients(
        inp, u, v, alpha, area_elem, T_conn, mesh_cache
    )

    if T_conn is None:
        alpha_elem = alpha
        dAlpha_elem = alpha - hist_alpha
    else:
        alpha_elem = (alpha[T_conn[:, 0]] + alpha[T_conn[:, 1]] + alpha[T_conn[:, 2]]) / 3
        dAlpha = alpha - hist_alpha
        dAlpha_elem = (dAlpha[T_conn[:, 0]] + dAlpha[T_conn[:, 1]] + dAlpha[T_conn[:, 2]]) / 3

    damageFn, _, c_w = pffmodel.damageFun(alpha_elem)
    weight_penalty = pffmodel.irrPenalty()
    E_el_elem, _ = strain_energy_with_split(strain_11, strain_22, strain_12, alpha_elem, matprop, pffmodel)
    E_el = area_elem * E_el_elem
    E_d = (matprop.w1 / c_w * (damageFn + matprop.l0**2 * (grad_alpha_x**2 + grad_alpha_y**2))) * area_elem

    hist_penalty = nn.ReLU()(-dAlpha_elem)
    E_hist_penalty = 0.5 * matprop.w1 * weight_penalty * hist_penalty**2 * area_elem
    return E_el, E_d, E_hist_penalty


def compute_energy(inp, u, v, alpha, hist_alpha, matprop, pffmodel, area_elem, T_conn=None, mesh_cache=None):
    E_el, E_d, E_hist_penalty = compute_energy_per_elem(
        inp, u, v, alpha, hist_alpha, matprop, pffmodel, area_elem, T_conn, mesh_cache
    )
    return torch.sum(E_el), torch.sum(E_d), torch.sum(E_hist_penalty)


# =============================================================================
# 6. Field Computation
# =============================================================================

class NonsmoothSigmoid(nn.Module):
    def __init__(self, support=2.0, coeff=1e-3):
        super().__init__()
        self.support = support
        self.coeff = coeff

    def forward(self, x):
        a = x > self.support
        b = x < -self.support
        c = torch.logical_not(torch.logical_or(a, b))
        out = (
            a * (self.coeff * (x - self.support) + 1.0)
            + b * (self.coeff * (x + self.support))
            + c * (x / 2.0 / self.support + 0.5)
        )
        return out


class FieldComputation(nn.Module):
    def __init__(self, disp_net, phase_net, domain_extrema, lmbda, theta, feature_dict, alpha_constraint="nonsmooth"):
        super().__init__()
        self.disp_net = disp_net
        self.phase_net = phase_net
        self.domain_extrema = domain_extrema
        self.theta = theta
        self.lmbda = lmbda
        self.feature_dict = feature_dict
        self.alpha_prev = None
        self.d_tilde = None
        self.psi = None
        self.g_tilde = None
        if alpha_constraint == "smooth":
            self.alpha_constraint = torch.sigmoid
        else:
            self.alpha_constraint = NonsmoothSigmoid(2.0, 1e-3)
        # Pin parameters (nondimensional, L = 120 mm)
        # Lower pin: center (20/120, 20/120) = (1/6, 1/6), radius 5/120 = 1/24
        # Upper pin: center (20/120, 100/120) = (1/6, 5/6), radius 5/120 = 1/24
        self.pin_cx = 1.0 / 6.0
        self.pin_cy_lower = 1.0 / 6.0
        self.pin_cy_upper = 5.0 / 6.0
        self.pin_radius = 5.0 / 120.0
        self.pin_transition = 0.02  # smooth transition width

    def set_phase_state(self, inp, alpha_prev, T_conn, area_T, l0):
        alpha_prev = alpha_prev.detach().to(inp.device)
        self.alpha_prev = alpha_prev
        mode = self.feature_dict.get("phase_feature_mode", "distance")

        self.d_tilde = None
        self.psi = None
        self.g_tilde = None

        if mode in ("distance", "both"):
            self.d_tilde, self.psi = compute_distance_to_damage(
                inp,
                alpha_prev,
                l0,
                alpha_threshold=self.feature_dict["damage_threshold"],
                chunk_size=self.feature_dict["distance_chunk_size"],
            )

        if mode in ("gradient", "both"):
            self.g_tilde = compute_nodal_grad_alpha(inp, alpha_prev, area_T, T_conn, l0).detach()

    def _build_phase_features(self, inp):
        if self.alpha_prev is None:
            raise RuntimeError("Phase state has not been initialized. Call set_phase_state(...) before training or inference.")

        mode = self.feature_dict.get("phase_feature_mode", "distance")
        features = [inp, self.alpha_prev.unsqueeze(-1)]
        if mode in ("distance", "both"):
            if self.d_tilde is None or self.psi is None:
                raise RuntimeError("Distance-based phase features were requested but have not been precomputed.")
            features.extend([self.d_tilde.unsqueeze(-1), self.psi.unsqueeze(-1)])
        if mode in ("gradient", "both"):
            if self.g_tilde is None:
                raise RuntimeError("Gradient-based phase features were requested but have not been precomputed.")
            features.append(self.g_tilde.unsqueeze(-1))
        return torch.cat(features, dim=1)

    def _pin_mask(self, x, y, cx, cy, r, transition):
        """Smooth mask for pin region: 1 inside pin, 0 outside, smooth transition."""
        dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        val = (r + transition - dist) / transition
        return torch.clamp(val, 0.0, 1.0)

    def fieldCalculation(self, inp):
        x = inp[:, -2]
        y = inp[:, -1]

        phase_features = self._build_phase_features(inp)
        disp_latent = self.disp_net(inp)
        phase_latent = self.phase_net(phase_features).squeeze(-1)
        alpha = self.alpha_constraint(phase_latent)

        lower_mask = self._pin_mask(x, y, self.pin_cx, self.pin_cy_lower, self.pin_radius, self.pin_transition)
        upper_mask = self._pin_mask(x, y, self.pin_cx, self.pin_cy_upper, self.pin_radius, self.pin_transition)
        free_mask = 1.0 - lower_mask - upper_mask

        # Lower pin: fixed (u=0, v=0)
        # Upper pin: vertical displacement control (u=0, v=lmbda)
        # Free region: predicted by network
        u = free_mask * disp_latent[:, 0] * self.lmbda
        v = free_mask * disp_latent[:, 1] * self.lmbda + upper_mask * self.lmbda

        return u, v, alpha

    def forward(self, inp):
        return self.fieldCalculation(inp)

    def update_hist_alpha(self, inp):
        _, _, pred_alpha = self.fieldCalculation(inp)
        return pred_alpha.detach()


# =============================================================================
# 7. Training Logic
# =============================================================================

def prep_input_data_wrapper(matprop, pffmodel, crack_dict, numr_dict, mesh_file, device):
    X, Y, T_conn, area_T = parse_mesh(filename=mesh_file, gradient_type=numr_dict["gradient_type"])
    # Normalize coordinates by characteristic length (L = 120 mm)
    X = X / L_CHAR
    Y = Y / L_CHAR
    area_T = area_T / (L_CHAR ** 2)
    inp = torch.from_numpy(np.column_stack((X, Y))).to(torch.float).to(device)
    T_conn = torch.from_numpy(T_conn).to(torch.long).to(device)
    area_T = torch.from_numpy(area_T).to(torch.float).to(device)
    if numr_dict["gradient_type"] == "autodiff":
        T_conn = None
    hist_alpha = hist_alpha_init(inp, matprop, pffmodel, crack_dict)
    return inp, T_conn, area_T, hist_alpha


def get_optimizer(params, optimizer_type="LBFGS"):
    if optimizer_type == "LBFGS":
        return optim.LBFGS(
            params,
            lr=0.5,
            max_iter=20000,
            max_eval=20000000,
            history_size=250,
            line_search_fn="strong_wolfe",
            tolerance_change=1.0 * np.finfo(float).eps,
            tolerance_grad=1.0 * np.finfo(float).eps,
        )
    if optimizer_type == "ADAM":
        return optim.Adam(params, lr=5e-4, betas=(0.9, 0.999), eps=1.0 * np.finfo(float).eps, weight_decay=0)
    if optimizer_type == "RPROP":
        return optim.Rprop(params, lr=1e-5, step_sizes=(1e-10, 50))
    raise ValueError("Optimizer type not recognized.")


class EarlyStopping:
    def __init__(self, tol_steps=10, min_delta=1e-3, device="cpu"):
        self.tol_steps = torch.tensor([tol_steps], dtype=torch.int, device=device)
        self.min_delta = torch.tensor([min_delta], dtype=torch.float, device=device)
        self.counter = torch.tensor([0], dtype=torch.int, device=device)
        self.early_stop = False

    def __call__(self, train_loss, train_loss_prev):
        delta = torch.abs(train_loss - train_loss_prev) / (torch.abs(train_loss_prev) + 1e-16)
        if delta > self.min_delta:
            self.counter = self.counter * 0
        else:
            self.counter += 1
            if self.counter >= self.tol_steps:
                self.early_stop = True


def compute_weight_regularization(field_comp):
    loss_reg = 0.0
    for name, param in field_comp.named_parameters():
        if "weight" in name:
            loss_reg += torch.sum(param**2)
    return loss_reg


def refine_mesh(inp_train, area_T, T_conn, alpha_inp, hist_alpha_inp, refine_cfg):
    if T_conn is None:
        return inp_train, T_conn, area_T, hist_alpha_inp, False, None

    alpha_threshold = float(refine_cfg.get("alpha_threshold", 0.25))
    target_size_ratio = float(refine_cfg.get("target_size_ratio", 0.3))
    transition_layers = int(refine_cfg.get("transition_layers", 6))
    target_size = float(refine_cfg.get("target_size", target_size_ratio * mat_prop_dict["l0"]))

    with torch.no_grad():
        alpha_elem = alpha_inp[T_conn].mean(dim=1)
        marked_mask = alpha_elem >= alpha_threshold
        marked_indices = torch.nonzero(marked_mask, as_tuple=False).flatten()

    log_progress = refine_cfg.get("log_training_progress", False)

    if marked_indices.numel() == 0:
        if log_progress:
            print(f"[Mesh Refine] Skip: no elements with mean alpha >= {alpha_threshold:.3f}.")
        return inp_train, T_conn, area_T, hist_alpha_inp, False, None

    with torch.no_grad():
        tri_pts = inp_train[T_conn]
        edge01 = torch.linalg.norm(tri_pts[:, 1] - tri_pts[:, 0], dim=1)
        edge12 = torch.linalg.norm(tri_pts[:, 2] - tri_pts[:, 1], dim=1)
        edge20 = torch.linalg.norm(tri_pts[:, 0] - tri_pts[:, 2], dim=1)
        max_edge_lengths = torch.maximum(torch.maximum(edge01, edge12), edge20)
        marked_max_size = float(max_edge_lengths[marked_indices].max().item())

    if marked_max_size <= target_size:
        if log_progress:
            print(
                f"[Mesh Refine] Skip: marked-region max_edge={marked_max_size:.6f} "
                f"<= target_size={target_size:.6f}."
            )
        return inp_train, T_conn, area_T, hist_alpha_inp, False, None

    synchronize_if_cuda(inp_train.device)
    refine_start = time.perf_counter()
    input_xy = inp_train.detach().cpu().numpy()
    T_conn_np = T_conn.detach().cpu().numpy()
    alpha_np = alpha_inp.detach().cpu().numpy().reshape(-1)
    hist_alpha_np = hist_alpha_inp.detach().cpu().numpy().reshape(-1)
    marked_elements = marked_indices.detach().cpu().numpy() + 1

    old_nodes = int(inp_train.shape[0])
    old_elements = int(T_conn.shape[0])
    refine_core_start = time.perf_counter()
    mesh = GmshTriMesh(x=input_xy[:, 0], y=input_xy[:, 1], t=T_conn_np)
    refined_x, refined_y, refined_t, refined_fields = TargetSizeTransitionTriMeshRefiner(
        mesh,
        target_size=target_size,
        transition_layers=transition_layers,
    ).refine(marked_elements.tolist(), f=np.column_stack((alpha_np, hist_alpha_np)))
    refine_core_time_minutes = (time.perf_counter() - refine_core_start) / 60.0

    refined_alpha = refined_fields[:, 0]
    refined_hist_alpha = refined_fields[:, 1]
    refined_mesh = GmshTriMesh(x=refined_x, y=refined_y, t=refined_t)
    refined_area = compute_triangle_area(refined_x, refined_y, refined_t)
    refined_inp = torch.from_numpy(np.column_stack((refined_x, refined_y))).to(dtype=inp_train.dtype, device=inp_train.device)
    refined_T_conn = torch.from_numpy(refined_t).to(dtype=T_conn.dtype, device=T_conn.device)
    refined_area_T = torch.from_numpy(refined_area).to(dtype=area_T.dtype, device=area_T.device)
    refined_hist_alpha_t = torch.from_numpy(refined_hist_alpha).to(dtype=hist_alpha_inp.dtype, device=hist_alpha_inp.device)
    synchronize_if_cuda(inp_train.device)
    refine_time_minutes = (time.perf_counter() - refine_start) / 60.0


    if log_progress:
        print(
            f"[Mesh Refine] alpha>={alpha_threshold:.3f}, marked={marked_elements.size}, "
            f"nodes {old_nodes} -> {refined_inp.shape[0]}, elements {old_elements} -> {refined_T_conn.shape[0]}, "
            f"time={refine_time_minutes:.03f} min."
        )
        if refine_cfg.get("plot_refine_results", False) and not refine_cfg.get("suppress_io_during_timing", True):
            print(f"[Mesh Refine] Saved plots to {refinePlot_path}.")

    refine_id = int(refine_cfg.get("_refine_plot_index", 0)) + 1
    refine_cfg["_refine_plot_index"] = refine_id

    record = {
        "refine_id": refine_id,
        "stage": str(refine_cfg.get("_stage", "unknown")),
        "load_step": int(refine_cfg.get("_load_step", -1)),
        "epoch": int(refine_cfg.get("_epoch", -1)),
        "refine_time_minutes": float(refine_time_minutes),
        "refine_core_time_minutes": float(refine_core_time_minutes),
        "old_nodes": old_nodes,
        "new_nodes": int(refined_inp.shape[0]),
        "old_elements": old_elements,
        "new_elements": int(refined_T_conn.shape[0]),
        "marked_elements": int(marked_elements.size),
    }
    queue_refined_mesh_output(
        refine_cfg,
        record,
        refined_x,
        refined_y,
        refined_t,
        refined_alpha,
        refined_hist_alpha,
    )
    return refined_inp, refined_T_conn, refined_area_T, refined_hist_alpha_t, True, record


def fit(
    field_comp,
    training_set,
    T_conn,
    area_T,
    hist_alpha,
    matprop,
    pffmodel,
    weight_decay,
    num_epochs,
    optimizer,
    intermediateModel_path=None,
    writer=None,
    training_dict=None,
):
    del writer
    if training_dict is None:
        training_dict = {}
    loss_data = []
    inp_train = training_set
    mesh_cache = build_mesh_cache(inp_train, area_T, T_conn)
    for epoch in range(num_epochs):
        def closure():
            optimizer.zero_grad()
            if T_conn is None:
                inp_train.requires_grad = True
            u, v, alpha = field_comp.fieldCalculation(inp_train)
            loss_E_el, loss_E_d, loss_hist = compute_energy(
                inp_train, u, v, alpha, hist_alpha, matprop, pffmodel, area_T, T_conn, mesh_cache
            )
            loss_var = torch.log10(loss_E_el + loss_E_d + loss_hist)

            loss_reg = compute_weight_regularization(field_comp) if weight_decay != 0 else 0.0
            loss = loss_var + weight_decay * loss_reg
            loss.backward()
            return loss

        loss = optimizer.step(closure=closure)
        if training_dict.get("log_training_progress", False):
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
        loss_data.append(loss.item())
        if (
            intermediateModel_path is not None
            and training_dict.get("save_intermediate_models", False)
            and not training_dict.get("suppress_io_during_timing", True)
        ):
            save_dual_checkpoint(field_comp, intermediateModel_path / f"interm_{int(field_comp.lmbda * 1e5)}_InitialTrain.pt")
    return loss_data


def fit_with_early_stopping(
    field_comp,
    training_set,
    T_conn,
    area_T,
    hist_alpha,
    matprop,
    pffmodel,
    weight_decay,
    num_epochs,
    optimizer,
    min_delta,
    intermediateModel_path=None,
    writer=None,
    training_dict=None,
):
    del writer
    if training_dict is None:
        training_dict = {}

    loss_data = []
    refine_records = []
    early_stopping = EarlyStopping(tol_steps=10, min_delta=min_delta, device=area_T.device)
    loss_prev = torch.tensor([0.0], device=area_T.device)
    refine_count = 0
    inp_train = training_set
    mesh_cache = build_mesh_cache(inp_train, area_T, T_conn)

    for epoch in range(num_epochs):
        training_dict["_epoch"] = epoch + 1
        optimizer.zero_grad()
        if T_conn is None:
            inp_train.requires_grad = True
        u, v, alpha = field_comp.fieldCalculation(inp_train)
        loss_E_el, loss_E_d, loss_hist = compute_energy(
            inp_train, u, v, alpha, hist_alpha, matprop, pffmodel, area_T, T_conn, mesh_cache
        )
        loss_var = torch.log10(loss_E_el + loss_E_d + loss_hist)

        loss_reg = compute_weight_regularization(field_comp) if weight_decay != 0 else 0.0
        loss = loss_var + weight_decay * loss_reg

        loss.backward()
        optimizer.step()
        if training_dict.get("log_training_progress", False) and epoch % 500 == 0:
            print(f"[RProp] Epoch {epoch}: L={loss}, l_var={loss_var}, l_E={loss_E_el}, l_d={loss_E_d}, l_h={loss_hist}")
        loss_data.append(loss.item())

        phase_monitor_every = int(training_dict.get("phase_monitor_every_n", 0))
        if (
            phase_monitor_every > 0
            and not training_dict.get("suppress_io_during_timing", True)
            and len(loss_data) % phase_monitor_every == 0
        ):
            monitor_start_disp = float(training_dict.get("phase_monitor_start_disp", -np.inf))
            if float(field_comp.lmbda.item()) >= monitor_start_disp:
                phase_monitor_path = training_dict.get("_phase_monitor_path")
                if phase_monitor_path is not None:
                    save_phase_monitor(
                        field_comp,
                        inp_train,
                        T_conn,
                        int(training_dict.get("_load_step", -1)),
                        len(loss_data),
                        float(field_comp.lmbda.item()),
                        Path(phase_monitor_path),
                    )

        if (
            intermediateModel_path is not None
            and training_dict.get("save_intermediate_models", False)
            and not training_dict.get("suppress_io_during_timing", True)
        ):
            steps = training_dict.get("save_model_every_n", 0)
            if steps > 0 and len(loss_data) >= steps and len(loss_data) % steps == 0:
                save_dual_checkpoint(field_comp, intermediateModel_path / f"interm_{int(field_comp.lmbda * 1e5)}_{len(loss_data)}.pt")

        refine_steps = training_dict.get("refine_every_n", 0)
        if refine_steps > 0 and len(loss_data) >= refine_steps and len(loss_data) % refine_steps == 0:
            if training_dict.get("adaptive_refine", False) and refine_count < training_dict.get("max_refine_calls", 0):
                inp_train, T_conn, area_T, hist_alpha, did_refine, refine_record = refine_mesh(
                    inp_train, area_T, T_conn, alpha.detach(), hist_alpha.detach(), training_dict
                )
                if did_refine:
                    field_comp.set_phase_state(inp_train, hist_alpha, T_conn, area_T, matprop.l0)
                    mesh_cache = build_mesh_cache(inp_train, area_T, T_conn)
                    refine_count += 1
                    early_stopping = EarlyStopping(tol_steps=10, min_delta=min_delta, device=area_T.device)
                    loss_prev = torch.tensor([0.0], device=area_T.device)
                    if refine_record is not None:
                        refine_records.append(refine_record)

        early_stopping(loss, loss_prev)
        if early_stopping.early_stop:
            if training_dict.get("log_training_progress", False):
                print("Early stopping triggered.")
            break
        loss_prev = loss
    return loss_data, inp_train, T_conn, area_T, hist_alpha, refine_records


# =============================================================================
# 8. Main Execution
# =============================================================================

def construct_model_wrapper(PFF_model_dict, mat_prop_dict, network_dict, domain_extrema, device):
    pffmodel = PFFModel(
        PFF_model=PFF_model_dict["PFF_model"],
        se_split=PFF_model_dict["se_split"],
        tol_ir=torch.tensor(PFF_model_dict["tol_ir"], device=device),
    )

    matprop = MaterialProperties(
        mat_E=torch.tensor(mat_prop_dict["mat_E"], device=device),
        mat_nu=torch.tensor(mat_prop_dict["mat_nu"], device=device),
        w1=torch.tensor(mat_prop_dict["w1"], device=device),
        l0=torch.tensor(mat_prop_dict["l0"], device=device),
    )

    disp_input_dimension, phase_input_dimension = get_feature_dimensions(feature_dict)
    disp_net = build_network(network_dict["disp_net"], disp_input_dimension, output_dimension=2)
    phase_net = build_network(network_dict["phase_net"], phase_input_dimension, output_dimension=1)
    return pffmodel, matprop, disp_net, phase_net


def _unwrap_compiled_module(module):
    return getattr(module, "_orig_mod", module)


def reinitialize_network_module(module, config, step_idx, seed_offset=0):
    raw_module = _unwrap_compiled_module(module)
    seed = int(config.get("seed", 1)) + int(seed_offset) + 10000 * (int(step_idx) + 1)
    torch.manual_seed(seed)

    if getattr(raw_module, "use_fourier", False):
        input_encoding = getattr(raw_module, "input_encoding", None)
        if config.get("fourier_trainable", False) and hasattr(input_encoding, "B"):
            with torch.no_grad():
                input_encoding.B.copy_(torch.randn_like(input_encoding.B) * float(config.get("ff_scale", 1.0)))

    init_xavier(raw_module)
    if getattr(raw_module, "trainable_activation", False):
        with torch.no_grad():
            for activation in raw_module.activations:
                if hasattr(activation, "coeff"):
                    activation.coeff.fill_(float(config.get("init_coeff", 1.0)))


def train():
    global model_path, trainedModel_path, intermediateModel_path, refinePlot_path, writer
    model_path, trainedModel_path, intermediateModel_path, refinePlot_path, writer = setup_run_artifacts()
    loss_plot_dir = model_path / "loss_plots"
    loss_plot_dir.mkdir(parents=True, exist_ok=True)
    total_wall_start = time.perf_counter()
    try:
        pffmodel, matprop, disp_net, phase_net = construct_model_wrapper(
            PFF_model_dict, mat_prop_dict, network_dict, domain_extrema, device
        )
        step_time_records = []
        memory_records_all = []
        pretrain_time_minutes = 0.0
        refine_records_all = []
        recorded_disp_rows = []
        recorded_energy_rows = []

        field_comp = FieldComputation(
            disp_net=disp_net,
            phase_net=phase_net,
            domain_extrema=domain_extrema,
            lmbda=torch.tensor([0.0], device=device),
            theta=loading_angle,
            feature_dict=feature_dict,
            alpha_constraint=numr_dict["alpha_constraint"],
        )
        field_comp.disp_net = field_comp.disp_net.to(device)
        field_comp.phase_net = field_comp.phase_net.to(device)
        field_comp.domain_extrema = field_comp.domain_extrema.to(device)
        field_comp.theta = field_comp.theta.to(device)
        field_comp.disp_net = torch.compile(field_comp.disp_net, dynamic=training_dict.get("compile_dynamic", True))
        field_comp.phase_net = torch.compile(field_comp.phase_net, dynamic=training_dict.get("compile_dynamic", True))
        parameter_summary = save_parameter_summary(model_path, field_comp)
        print(f"[Info] Trainable parameters: {parameter_summary['total']['trainable']}")

        print("[NN Train] --- Starting Pre-training (Coarse Mesh) ---")
        inp, T_conn, area_T, hist_alpha = prep_input_data_wrapper(
            matprop, pffmodel, crack_dict, numr_dict, mesh_file=coarse_mesh_file, device=device
        )
        alpha_prev = hist_alpha.clone().detach()
        field_comp.set_phase_state(inp, alpha_prev, T_conn, area_T, matprop.l0)
        pretrain_training_dict = copy.deepcopy(training_dict)
        pretrain_training_dict["max_refine_calls"] = int(training_dict.get("pretrain_max_refine_calls", 0))
        pretrain_training_dict["_stage"] = "pretrain"
        pretrain_training_dict["_load_step"] = -1

        field_comp.lmbda = torch.tensor(disp[0]).to(device)
        loss_data = []
        reset_peak_memory_stats(device)
        synchronize_if_cuda(device)
        start = time.perf_counter()

        n_epochs = max(optimizer_dict["n_epochs_LBFGS"], 1)
        optimizer = get_optimizer(field_comp.parameters(), "LBFGS")
        loss_d = fit(
            field_comp,
            inp,
            T_conn,
            area_T,
            hist_alpha,
            matprop,
            pffmodel,
            optimizer_dict["weight_decay"],
            num_epochs=n_epochs,
            optimizer=optimizer,
            intermediateModel_path=None,
            writer=writer,
            training_dict=pretrain_training_dict,
        )
        loss_data.extend(loss_d)

        refine_records = []
        n_epochs = optimizer_dict["n_epochs_RPROP"]
        optimizer = get_optimizer(field_comp.parameters(), "RPROP")
        loss_d, inp, T_conn, area_T, hist_alpha, refine_records = fit_with_early_stopping(
            field_comp,
            inp,
            T_conn,
            area_T,
            hist_alpha,
            matprop,
            pffmodel,
            optimizer_dict["weight_decay"],
            num_epochs=n_epochs,
            optimizer=optimizer,
            min_delta=optimizer_dict["optim_rel_tol_pretrain"],
            intermediateModel_path=None,
            writer=writer,
            training_dict=pretrain_training_dict,
        )
        loss_data.extend(loss_d)
        refine_records_all.extend(refine_records)
        synchronize_if_cuda(device)
        end = time.perf_counter()
        pretrain_time_minutes = (end - start) / 60.0
        pretrain_memory = get_memory_stats(device)
        pretrain_memory.update({
            "stage": "pretrain",
            "load_step": -1,
            "displacement": float(disp[0]),
            "compute_time_minutes": float(pretrain_time_minutes),
        })
        memory_records_all.append(pretrain_memory)
        print(f"[NN Train] Pre-training compute time: {pretrain_time_minutes:.03f} minutes")
        flush_refined_mesh_outputs(model_path, pretrain_training_dict)

        save_dual_checkpoint(field_comp, trainedModel_path / "trained_2NN_initTraining.pt")
        loss_array = np.asarray(loss_data, dtype=float)
        with open(trainedModel_path / "trainLoss_initTraining.npy", "wb") as file:
            np.save(file, loss_array)
        plot_loss_curve(
            loss_array,
            loss_plot_dir / "loss_initTraining.png",
            "Pre-training Loss History",
        )

        print("[NN Train] --- Starting Main Training (Notched Hole, Fixed Mesh) ---")
        main_training_dict = copy.deepcopy(training_dict)
        main_training_dict["max_refine_calls"] = int(training_dict.get("main_max_refine_calls", 0))
        main_training_dict["_refine_plot_index"] = int(pretrain_training_dict.get("_refine_plot_index", 0))
        main_training_dict["_stage"] = "main"
        main_training_dict["_phase_monitor_path"] = model_path / "phase_monitor"
        main_training_dict["_phase_monitor_path"].mkdir(parents=True, exist_ok=True)
        alpha_prev = hist_alpha.clone().detach()

        for j, disp_i in enumerate(disp):
            field_comp.lmbda = torch.tensor(disp_i).to(device)
            reinitialize_network_module(field_comp.phase_net, network_dict["phase_net"], j, seed_offset=200000)
            print(f"[NN Train] Reinitialized phase_net for load step {j}.")
            field_comp.set_phase_state(inp, alpha_prev, T_conn, area_T, matprop.l0)
            main_training_dict["_load_step"] = j
            print(f"[NN Train] ########## Step {j}: displacement = {field_comp.lmbda.item()} ##########")
            loss_data = []
            reset_peak_memory_stats(device)
            synchronize_if_cuda(device)
            start = time.perf_counter()

            if j == 0 or optimizer_dict["n_epochs_LBFGS"] > 0:
                n_epochs = max(optimizer_dict["n_epochs_LBFGS"], 1)
                optimizer = get_optimizer(field_comp.parameters(), "LBFGS")
                loss_d = fit(
                    field_comp,
                    inp,
                    T_conn,
                    area_T,
                    hist_alpha,
                    matprop,
                    pffmodel,
                    optimizer_dict["weight_decay"],
                    num_epochs=n_epochs,
                    optimizer=optimizer,
                    intermediateModel_path=None,
                    writer=writer,
                    training_dict=main_training_dict,
                )
                loss_data.extend(loss_d)

            refine_records = []
            if optimizer_dict["n_epochs_RPROP"] > 0:
                n_epochs = optimizer_dict["n_epochs_RPROP"]
                optimizer = get_optimizer(field_comp.parameters(), "RPROP")
                loss_d, inp, T_conn, area_T, hist_alpha, refine_records = fit_with_early_stopping(
                    field_comp,
                    inp,
                    T_conn,
                    area_T,
                    hist_alpha,
                    matprop,
                    pffmodel,
                    optimizer_dict["weight_decay"],
                    num_epochs=n_epochs,
                    optimizer=optimizer,
                    min_delta=optimizer_dict["optim_rel_tol"],
                    intermediateModel_path=intermediateModel_path,
                    writer=writer,
                    training_dict=main_training_dict,
                )
                loss_data.extend(loss_d)
                refine_records_all.extend(refine_records)

            synchronize_if_cuda(device)
            end = time.perf_counter()
            step_time_minutes = (end - start) / 60.0
            step_time_records.append((j, step_time_minutes))
            step_memory = get_memory_stats(device)
            step_memory.update({
                "stage": "main",
                "load_step": int(j),
                "displacement": float(disp_i),
                "compute_time_minutes": float(step_time_minutes),
                "refinement_time_minutes": float(
                    sum(
                        record.get("refine_time_minutes", 0.0)
                        for record in refine_records
                        if record.get("load_step", -1) == j
                    )
                ),
            })
            memory_records_all.append(step_memory)
            print(f"[NN Train] Step compute time: {step_time_minutes:.03f} minutes")
            flush_refined_mesh_outputs(model_path, main_training_dict)

            pred_alpha, energy_row = save_current_step_result(
                field_comp,
                inp,
                T_conn,
                area_T,
                hist_alpha,
                matprop,
                pffmodel,
                j,
                disp_i,
                model_path,
            )
            recorded_disp_rows.append(float(disp_i))
            recorded_energy_rows.append(energy_row)
            write_recorded_energy_history(model_path, recorded_disp_rows, recorded_energy_rows)
            history_alpha_cut = float(feature_dict["history_alpha_cut"])
            hist_alpha = torch.where(pred_alpha > history_alpha_cut, pred_alpha, torch.zeros_like(pred_alpha)).detach()
            alpha_prev = hist_alpha

            save_dual_checkpoint(field_comp, trainedModel_path / f"trained_2NN_{j}.pt")
            loss_array = np.asarray(loss_data, dtype=float)
            with open(trainedModel_path / f"trainLoss_{j}.npy", "wb") as file:
                np.save(file, loss_array)
            plot_loss_curve(
                loss_array,
                loss_plot_dir / f"loss_step_{j:03d}.png",
                f"Load Step {j} Loss History (disp = {disp_i:.4f})",
            )

        print("[Plot] --- Finalizing recorded postprocess figures ---")
        write_recorded_energy_history(model_path, recorded_disp_rows, recorded_energy_rows)

        total_compute_time_minutes = float(pretrain_time_minutes + sum(t_minutes for _, t_minutes in step_time_records))
        wall_time_minutes = (time.perf_counter() - total_wall_start) / 60.0
        save_refinement_summary(model_path, refine_records_all)
        save_training_summary(
            model_path,
            pretrain_time_minutes,
            step_time_records,
            total_compute_time_minutes,
            refine_records_all,
            memory_records_all,
            wall_time_minutes=wall_time_minutes,
        )
        print(f"[NN Train] Total compute time: {total_compute_time_minutes:.03f} minutes")
        print(f"[NN Train] Total wall time including output: {wall_time_minutes:.03f} minutes")
        print(
            f"[Mesh Refine] Total refinement time: "
            f"{sum(record['refine_time_minutes'] for record in refine_records_all):.03f} minutes"
        )
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    train()
