#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script generates candidate 4-connected T-atom frameworks by placing
symmetry-unique T sites relative to a point cloud contour, followed by
space-group expansion, density screening, and topological validation based on
nearest-four-neighbor graphs.

Key implementation notes
------------------------
- Candidate space groups are inferred solely from the metric of the
  ``--contour-cif`` lattice.
- A hexagonal-axis metric (a = b != c, alpha = beta = 90 deg, gamma = 120 deg)
  is treated as ambiguous between trigonal and hexagonal settings, so both
  space-group ranges are enumerated.
- General-position multiplicities are approximated as
  ``len(SpaceGroup.from_int_number(n).symmetry_ops)``. For rhombohedral
  trigonal groups, pymatgen/spglib provides symmetry operations in the
  conventional hexagonal setting, which is the intended behavior here.

"""

import argparse
import os
from typing import List, Tuple, Dict, Set
import numpy as np
from scipy.spatial import cKDTree

from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.groups import SpaceGroup


# ============================= Geometry Utilities ============================= #

def wrap01(frac: np.ndarray) -> np.ndarray:
    """Wrap fractional coordinates into the interval [0, 1)."""
    return frac - np.floor(frac)


def min_image_cart_deltas(frac_point: np.ndarray,
                          frac_array: np.ndarray,
                          lattice: Lattice) -> np.ndarray:
    """
    Compute minimum-image displacement vectors in Cartesian coordinates (A).
    """
    dfrac = frac_array - frac_point[None, :]
    dfrac -= np.round(dfrac)
    return dfrac @ lattice.matrix


def min_image_distances(frac_point: np.ndarray,
                        frac_array: np.ndarray,
                        lattice: Lattice) -> np.ndarray:
    """Return minimum-image distances from ``frac_point`` to ``frac_array``."""
    vecs = min_image_cart_deltas(frac_point, frac_array, lattice)
    return np.linalg.norm(vecs, axis=1)


def tile_with_pbc_images(points_cart: np.ndarray,
                         lattice: Lattice,
                         shell: int = 1) -> np.ndarray:
    """
    Tile a Cartesian point cloud into neighboring cells for PBC-aware KDTree queries.
    """
    a, b, c = lattice.matrix.T
    shifts = []
    for i in range(-shell, shell + 1):
        for j in range(-shell, shell + 1):
            for k in range(-shell, shell + 1):
                shifts.append(i * a + j * b + k * c)
    shifts = np.asarray(shifts)
    return (points_cart[None, :, :] + shifts[:, None, :]).reshape(-1, 3)


# ==================== Point Cloud Contour and Lattice Metric ==================== #

def load_contour_and_lattice(path: str) -> Tuple[np.ndarray, Lattice]:
    """
    Read point-cloud-contour coordinates and the lattice from a CIF file.

    Returns
    -------
    contour_frac
        Array of shape ``(Nf, 3)`` with values wrapped into ``[0, 1)``.
    lattice_contour
        Reference lattice read from the same CIF file.
    """
    s = Structure.from_file(path)
    contour_frac = wrap01(s.frac_coords.copy())
    lattice_contour = s.lattice
    return contour_frac, lattice_contour


# ---------- Lattice metric -> candidate space groups ----------
def _eq_len(x: float, y: float, tol: float = 1e-5) -> bool:
    """Return whether two lengths are approximately equal for metric classification."""
    return abs(x - y) <= tol

def _eq_ang(x: float, y: float, tol: float = 1e-3) -> bool:
    """Return whether two angles are approximately equal for metric classification."""
    return abs(x - y) <= tol

def classify_metric_by_contour_lattice(lat: Lattice) -> str:
    """
    Classify the contour-lattice metric with a small tolerance.

    The tolerance is used only for metric classification so that small floating-
    point deviations in CIF input do not misclassify the lattice family.
    """
    a, b, c = lat.a, lat.b, lat.c
    alpha, beta, gamma = lat.alpha, lat.beta, lat.gamma

    # 1) Cubic: a = b = c, alpha = beta = gamma = 90 deg
    if _eq_len(a, b) and _eq_len(b, c) and _eq_ang(alpha, 90) and _eq_ang(beta, 90) and _eq_ang(gamma, 90):
        return 'cubic'

    # 2) Hexagonal-axis metric: a = b != c, alpha = beta = 90 deg, gamma = 120 deg
    if _eq_len(a, b) and not _eq_len(b, c) and _eq_ang(alpha, 90) and _eq_ang(beta, 90) and _eq_ang(gamma, 120):
        return 'hex_trigonal_ambiguous'

    # 3) Rhombohedral metric: a = b = c, alpha = beta = gamma != 90 deg
    if _eq_len(a, b) and _eq_len(b, c) and _eq_ang(alpha, beta) and _eq_ang(beta, gamma) and (not _eq_ang(alpha, 90)):
        return 'rhombohedral_metric'

    # 4) Tetragonal: a = b != c, alpha = beta = gamma = 90 deg
    if _eq_len(a, b) and not _eq_len(b, c) and _eq_ang(alpha, 90) and _eq_ang(beta, 90) and _eq_ang(gamma, 90):
        return 'tetragonal'

    # 5) Orthorhombic: alpha = beta = gamma = 90 deg, with unequal lengths
    if _eq_ang(alpha, 90) and _eq_ang(beta, 90) and _eq_ang(gamma, 90) and (not (_eq_len(a, b) and _eq_len(b, c))):
        return 'orthorhombic'

    # 6) Monoclinic: exactly two angles are approximately 90 deg
    ninety = sum(int(_eq_ang(x, 90)) for x in (alpha, beta, gamma))
    if ninety == 2:
        return 'monoclinic'

    # 7) Otherwise treat as triclinic
    return 'triclinic'


def sg_candidates_from_metric(lat: Lattice) -> List[int]:
    """
    Return the space-group numbers to be sampled for a given lattice metric.

    Rules
    -----
    - ``hex_trigonal_ambiguous`` -> trigonal (143-167) union hexagonal (168-194)
    - ``rhombohedral_metric`` -> trigonal (143-167)
    - otherwise -> direct mapping to the conventional crystal-family range
    """
    tag = classify_metric_by_contour_lattice(lat)

    ranges = {
        "triclinic":    (1, 2),
        "monoclinic":   (3, 15),
        "orthorhombic": (16, 74),
        "tetragonal":   (75, 142),
        "trigonal":     (143, 167),
        "hexagonal":    (168, 194),
        "cubic":        (195, 230),
    }

    if tag == 'hex_trigonal_ambiguous':
        lo1, hi1 = ranges["trigonal"]
        lo2, hi2 = ranges["hexagonal"]
        sgs = list(range(lo1, hi1 + 1)) + list(range(lo2, hi2 + 1))
        return sorted(sgs)

    if tag == 'rhombohedral_metric':
        lo, hi = ranges["trigonal"]
        return list(range(lo, hi + 1))

    # Standard direct mapping for all remaining metric classes.
    if tag in ranges:
        lo, hi = ranges[tag]
        return list(range(lo, hi + 1))

    # Reaching this branch indicates an unexpected classification state.
    raise ValueError(
        f"Unhandled lattice metric tag: '{tag}'. "
        "Please check the point-cloud-contour CIF cell parameters and classification logic."
    )


def estimate_general_multiplicity(sg_number: int) -> int:
    """
    Estimate the general-position multiplicity as ``len(symmetry_ops)``.

    For rhombohedral trigonal groups, pymatgen/spglib exposes symmetry
    operations in the conventional hexagonal setting, which is the intended
    convention for this workflow.
    """
    sg = SpaceGroup.from_int_number(sg_number)
    return len(sg.symmetry_ops)


def pre_filter_spacegroups_by_density(
    sg_list: List[int],
    num_unique: int,
    lattice_ref: Lattice,
    t_density_min: float,
    t_density_max: float | None = None,
    gm_cache: Dict[int, int] | None = None,
) -> List[int]:
    """
    Quickly pre-filter space groups by the target T-density window.

    The screening uses an estimated number of T atoms,
    ``n_T_est ~= num_unique * general_multiplicity(sg)``, and converts it to
    ``T / 1000 A^3`` at the current lattice volume.

    Parameters
    ----------
    gm_cache
        Optional cache mapping ``sg -> multiplicity`` to avoid repeated calls.
    """
    V = lattice_ref.volume
    filtered = []
    for sg in sg_list:
        gm = gm_cache.get(sg) if gm_cache is not None else None
        if gm is None:
            gm = estimate_general_multiplicity(sg)
            if gm_cache is not None:
                gm_cache[sg] = gm

        nT_est = num_unique * gm
        t_per_1000_est = (nT_est / V) * 1000.0

        if t_per_1000_est < t_density_min:
            # Too sparse under the current estimate.
            continue
        if (t_density_max is not None) and (t_per_1000_est > t_density_max):
            # Too dense under the current estimate.
            continue

        filtered.append(sg)

    return filtered


# ===================== Symmetry Expansion and T-Site Sampling ==================== #

class ConstrainedAssembler:
    """
    Randomly place symmetry-unique T sites subject to geometric constraints,
    then expand them by the selected space-group operations.
    """

    def __init__(
        self,
        trial_lattice: Lattice,
        contour_tree: cKDTree,
        spacegroup_number: int,
        excl_radius: float,
        shell_min: float,
        shell_max: float,
        enable_shell: bool,
        rng: np.random.Generator,
        tt_min: float | None,
    ):
        self.lattice = trial_lattice
        self.contour_tree = contour_tree
        self.sg_number = int(spacegroup_number)

        self.excl_radius = float(excl_radius)
        self.shell_min = float(shell_min)
        self.shell_max = float(shell_max)
        self.enable_shell = bool(enable_shell)

        self.rng = rng
        self.tt_min = None if tt_min is None else float(tt_min)

        # For rhombohedral trigonal groups, pymatgen/spglib returns operations
        # in the conventional hexagonal setting, which is intentional here.
        self.sg_ops = SpaceGroup.from_int_number(self.sg_number).symmetry_ops

        self.accepted_unique: List[np.ndarray] = []
        self.accepted_expanded_frac: np.ndarray | None = None

    def expand_points_by_symmetry(self, frac_array: np.ndarray) -> np.ndarray:
        """Apply all symmetry operations and deduplicate the resulting orbit."""
        frac_array = np.atleast_2d(frac_array)
        all_list = []
        for op in self.sg_ops:
            all_list.append(wrap01(op.operate_multi(frac_array)))
        allf = np.vstack(all_list)
        key = np.round(allf, 6)
        _, idx = np.unique(key, axis=0, return_index=True)
        return allf[idx]

    def _update_accepted_expanded(self):
        """Refresh the expanded set corresponding to accepted unique sites."""
        if not self.accepted_unique:
            self.accepted_expanded_frac = None
            return
        uniq = np.vstack(self.accepted_unique)
        expanded = self.expand_points_by_symmetry(uniq)
        self.accepted_expanded_frac = expanded

    # ---------- Point-cloud-contour and shell constraints ----------

    def gaps_to_contour(self, orbit_frac: np.ndarray) -> np.ndarray:
        """Return the distance-to-contour margin for every point in an orbit."""
        orbit_cart = self.lattice.get_cartesian_coords(orbit_frac)
        dmin, _ = self.contour_tree.query(orbit_cart)
        return dmin - self.excl_radius

    def orbit_enters_contour(self, orbit_frac: np.ndarray) -> bool:
        """Return ``True`` if any point in the orbit enters the contour exclusion region."""
        gaps = self.gaps_to_contour(orbit_frac)
        return np.any(gaps < 0.0)

    def orbit_on_surface_shell(self, orbit_frac: np.ndarray) -> bool:
        """
        If shell filtering is enabled, require every point in the orbit to lie
        within ``[shell_min, shell_max]`` from the point cloud contour.
        """
        if not self.enable_shell:
            return True
        orbit_cart = self.lattice.get_cartesian_coords(orbit_frac)
        dmin, _ = self.contour_tree.query(orbit_cart)
        inband = (dmin >= self.shell_min) & (dmin <= self.shell_max)
        return bool(np.all(inband))

    # ---------- Minimum-distance criteria under periodic boundary conditions ----------

    def orbit_self_collision(self, orbit_frac: np.ndarray) -> bool:
        """Check whether the orbit contains any T-T distance below ``tt_min``."""
        if self.tt_min is None:
            return False
        n = len(orbit_frac)
        if n <= 1:
            return False
        for i in range(n):
            d = min_image_distances(orbit_frac[i], orbit_frac, self.lattice)
            if np.any((d > 1e-8) & (d < self.tt_min)):
                return True
        return False

    def orbit_too_close_to_accepted(self, orbit_frac: np.ndarray) -> bool:
        """Check whether an orbit is too close to previously accepted sites."""
        if self.tt_min is None or self.accepted_expanded_frac is None:
            return False
        for p in orbit_frac:
            d = min_image_distances(p, self.accepted_expanded_frac, self.lattice)
            if np.any(d < self.tt_min):
                return True
        return False

    # ---------- Main sampling routine ----------

    def sample_unique_sites(self,
                            n_unique: int,
                            max_tries_per_site: int = 20000) -> np.ndarray:
        """
        Randomly place ``n_unique`` independent T sites subject to all active
        contour, shell, and minimum-distance constraints.
        """
        accepted_local = []
        tries = 0

        while len(accepted_local) < n_unique and tries < max_tries_per_site * n_unique:
            trial = wrap01(self.rng.random(3))
            orbit = self.expand_points_by_symmetry(trial)

            if self.orbit_enters_contour(orbit):
                tries += 1
                continue

            if len(accepted_local) == 0 and (not self.orbit_on_surface_shell(orbit)):
                tries += 1
                continue

            if self.orbit_self_collision(orbit):
                tries += 1
                continue
            if self.orbit_too_close_to_accepted(orbit):
                tries += 1
                continue

            accepted_local.append(trial)
            self.accepted_unique.append(trial)
            self._update_accepted_expanded()

        if len(accepted_local) < n_unique:
            raise RuntimeError(
                "Unable to place enough symmetry-unique T sites under the current "
                "constraints; consider loosening the geometry filters, reducing the "
                "symmetry, or adjusting the shell / tt-min parameters."
            )

        return np.vstack(accepted_local)


def expand_all_sites(unique_frac: np.ndarray, sg_number: int) -> np.ndarray:
    """Expand all symmetry-unique sites and return deduplicated T coordinates."""
    sg = SpaceGroup.from_int_number(sg_number)
    all_list = []
    for op in sg.symmetry_ops:
        all_list.append(wrap01(op.operate_multi(unique_frac)))
    allf = np.vstack(all_list)
    key = np.round(allf, 6)
    _, idx = np.unique(key, axis=0, return_index=True)
    return allf[idx]


def group_sites_by_unique(unique_frac: np.ndarray,
                          sg_number: int,
                          all_frac: np.ndarray) -> List[List[int]]:
    """
    Reconstruct the indices in ``all_frac`` that belong to each unique-site orbit.
    """
    sg = SpaceGroup.from_int_number(sg_number)
    groups = []
    all_key = np.round(all_frac, 6)

    for u in unique_frac:
        orbit_u = []
        for op in sg.symmetry_ops:
            orbit_u.append(wrap01(op.operate(u)))
        orbit_u = np.vstack(orbit_u)
        orbit_u_key = np.round(orbit_u, 6)
        idxs = []
        for k in orbit_u_key:
            hits = np.where(np.all(all_key == k, axis=1))[0]
            idxs.extend(list(hits))
        idxs = sorted(set(idxs))
        groups.append(idxs)

    return groups


# ======================= Topological Graph and CS Analysis ====================== #

def nearest4_for_each_site(all_frac: np.ndarray,
                           lattice: Lattice,
                           i: int) -> List[int]:
    """Return the indices of the four nearest T neighbors for site ``i``."""
    dists = min_image_distances(all_frac[i], all_frac, lattice)
    order = np.argsort(dists)
    order = [idx for idx in order if idx != i]
    return order[:4]


def build_undirected_graph_from_4nn(all_frac: np.ndarray,
                                    lattice: Lattice) -> Dict[int, Set[int]]:
    """Build an undirected T-T graph from the nearest-four-neighbor rule."""
    n = len(all_frac)
    graph: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        nbrs_i = nearest4_for_each_site(all_frac, lattice, i)
        for j in nbrs_i:
            graph[i].add(j)
            graph[j].add(i)
    return graph


def coordination_sequence_bfs(graph: Dict[int, Set[int]],
                              start: int,
                              max_depth: int = 3) -> List[int]:
    """Compute the coordination sequence ``[N1, N2, N3, ...]`` by BFS."""
    visited = {start}
    current_shell = {start}
    cs_counts = []

    for depth in range(1, max_depth + 1):
        next_shell = set()
        for node in current_shell:
            next_shell |= graph[node]
        next_shell -= visited

        cs_counts.append(len(next_shell))

        visited |= next_shell
        current_shell = next_shell

        if len(next_shell) == 0:
            break

    return cs_counts


def check_unique_sites_degree_four(
    groups: List[List[int]],
    graph: Dict[int, Set[int]],
    verbose: bool = True,
) -> Tuple[bool, List[int], List[List[int]]]:
    """
    Enforce strict four-connectivity for each symmetry-unique representative.
    """
    ok_all = True
    deg_list = []
    cs_list_all = []

    for u, idx_list in enumerate(groups):
        if len(idx_list) == 0:
            if verbose:
                print(f"[WARN] unique site #{u} has empty orbit after symmetry expansion?")
            ok_all = False
            deg_list.append(0)
            cs_list_all.append([])
            continue

        rep_idx = idx_list[0]
        degree_u = len(graph[rep_idx])
        deg_list.append(degree_u)

        cs_u = coordination_sequence_bfs(graph, rep_idx, max_depth=3)
        cs_list_all.append(cs_u)

        if verbose:
            print(f"[CS] unique#{u} rep={rep_idx}: degree={degree_u}, CS={cs_u}")

        if degree_u != 4:
            ok_all = False

    return ok_all, deg_list, cs_list_all

def is_strictly_increasing(seq: List[int]) -> bool:
    """Return whether an integer sequence is strictly increasing."""
    if len(seq) <= 1:
        return True
    return all(x < y for x, y in zip(seq, seq[1:]))

def add_oxygen_on_edges_by_graph(all_frac: np.ndarray,
                                 lattice: Lattice,
                                 graph: Dict[int, Set[int]]) -> np.ndarray:
    """
    Place O atoms at midpoints of minimum-image T-T edges as approximate bridges.
    """
    o_list = []
    n = len(all_frac)
    for i in range(n):
        for j in graph[i]:
            if j <= i:
                continue
            vec_ij = min_image_cart_deltas(all_frac[i],
                                           np.array([all_frac[j]]),
                                           lattice)[0]
            T_i_cart = lattice.get_cartesian_coords(all_frac[i])
            O_cart = T_i_cart + 0.5 * vec_ij
            O_frac = wrap01(lattice.get_fractional_coords(O_cart))
            o_list.append(O_frac)

    if len(o_list) == 0:
        return np.zeros((0, 3))

    return np.vstack(o_list)


# =========================== Scale Scan and Output =========================== #

def parse_scale_scan(spec: str) -> List[float]:
    """Parse ``--scale-scan`` from either ``1.0`` or ``start:stop:step``."""
    spec = spec.strip()
    if ":" not in spec:
        return [float(spec)]
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError("--scale-scan must be either 'start:stop:step' or a single value such as '1.0'.")
    start, stop, step = map(float, parts)
    if step <= 0:
        if abs(stop - start) < 1e-12:
            return [start]
        raise ValueError("step must be positive.")
    n = int(np.floor((stop - start) / step + 0.5)) + 1
    return [start + i * step for i in range(max(n, 1))]


def scale_lattice(lat: Lattice, s: float) -> Lattice:
    """Scale the lattice matrix isotropically by ``s``."""
    return Lattice(lat.matrix * float(s))


# ================================= Main Program ================================= #

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate 4-connected T frameworks by constrained random placement + "
            "symmetry expansion + topological four-coordination check (no tt_max)."
        )
    )

    parser.add_argument("--contour-cif", required=True,
                        help="CIF containing the point cloud contour; its lattice is also used as the reference metric.")

    parser.add_argument("--excl-radius", type=float, default=0.81,
                        help="Contour exclusion radius in A. All T images must remain outside this distance.")
    parser.add_argument("--num-unique", type=int, default=2,
                        help="Number of symmetry-unique T sites to place randomly.")

    parser.add_argument("--tt-min", type=float, default=3.0,
                        help="Minimum allowed T-T distance in A. No upper bound is applied.")

    parser.add_argument("--enable-shell", action="store_true",
                        help="Require the first unique T site to lie within a shell around the point cloud contour.")
    parser.add_argument("--shell-min", type=float, default=0.0,
                        help="Lower bound of the shell window in A.")
    parser.add_argument("--shell-max", type=float, default=0.5,
                        help="Upper bound of the shell window in A.")

    parser.add_argument("--contour-mode", choices=["fixed", "scaled"], default="fixed",
                        help=("fixed: build the point-cloud-contour KDTree once in the reference lattice; "
                              "scaled: rebuild it after each isotropic lattice scaling step."))

    parser.add_argument("--scale-scan", type=str, default="1.0",
                        help="Either a single value such as '1.0' or a range '0.90:1.10:0.02'.")

    parser.add_argument("--t-density-min", type=float, default=12.0,
                        help="Minimum allowed T density in T per 1000 A^3.")
    parser.add_argument("--t-density-max", type=float, default=20.0,
                        help="Maximum allowed T density in T per 1000 A^3.")

    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed.")
    parser.add_argument("--output-dir", default="Results",
                        help="Root output directory.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress information.")
    
    parser.add_argument(
        "--stop-after-first-scale",
        action="store_true",
        help="Skip later scale values for a space group after its first successful structure."
    )

    args = parser.parse_args()
    verbose_print = print if args.verbose else (lambda *items, **kwargs: None)

    rng = np.random.default_rng(args.seed)
    verbose_print(f"[INFO] RNG seed = {args.seed}")

    # 1) Read the point cloud contour and the reference lattice.
    contour_frac, contour_lattice = load_contour_and_lattice(args.contour_cif)

    # 2) Infer candidate space groups solely from the contour-lattice metric.
    metric_tag = classify_metric_by_contour_lattice(contour_lattice)
    sg_candidates = sg_candidates_from_metric(contour_lattice)

    verbose_print(f"[INFO] metric_tag={metric_tag}; "
                  f"SG candidates: {sg_candidates[0]}–{sg_candidates[-1]} "
                  f"({len(sg_candidates)} total)")

    # 3) Prepare a multiplicity cache for repeated density pre-screening.
    gm_cache: Dict[int, int] = {sg: estimate_general_multiplicity(sg) for sg in sg_candidates}

    # 4) Build the lattice-scaling schedule.
    scales = parse_scale_scan(args.scale_scan)
    verbose_print("[SCAN] scales to try:", ", ".join(f"{s:.4f}" for s in scales))

    # 5) Prepare the point-cloud-contour KDTree.
    if args.contour_mode == "fixed":
        contour_cart_ref = contour_lattice.get_cartesian_coords(contour_frac)
        contour_cart_imgs_ref = tile_with_pbc_images(contour_cart_ref, contour_lattice, shell=1)
        fixed_tree = cKDTree(contour_cart_imgs_ref)
        verbose_print("[INFO] Contour mode=fixed: KDTree built once in reference lattice.")
    else:
        fixed_tree = None

    os.makedirs(args.output_dir, exist_ok=True)

    success_count = 0

    # Track space groups that already succeeded at an earlier scale value.
    finished_sg: set[int] = set()

    # 7) Main loop: pre-filter per scale, then iterate over ``(sg, scale)``.
    for s in scales:
        # 7.1 Scale the lattice isotropically.
        lat_s = scale_lattice(contour_lattice, s)
        verbose_print(
            f"\n[SCAN] s={s:.4f} -> "
            f"a={lat_s.a:.3f}, b={lat_s.b:.3f}, c={lat_s.c:.3f}, "
            f"alpha={lat_s.alpha:.2f}, beta={lat_s.beta:.2f}, gamma={lat_s.gamma:.2f}"
        )

        # 7.2 Apply density-based pre-screening at the current scale.
        sg_list_s = pre_filter_spacegroups_by_density(
            sg_candidates,
            num_unique=args.num_unique,
            lattice_ref=lat_s,
            t_density_min=args.t_density_min,
            t_density_max=args.t_density_max,
            gm_cache=gm_cache,
        )
        if len(sg_list_s) == 0:
            verbose_print(f"[INFO][s={s:.4f}] after pre-filter: 0 SG -> "
                          f"no candidates satisfy density window [{args.t_density_min}, {args.t_density_max}]")
            continue
        verbose_print(f"[INFO][s={s:.4f}] after pre-filter: {len(sg_list_s)} SG -> {sg_list_s}")

        # 7.3 Prepare the point-cloud-contour KDTree for this scale.
        if args.contour_mode == "fixed":
            contour_tree = fixed_tree
        else:
            # In ``scaled`` mode, the contour cloud is rebuilt in the scaled lattice.
            contour_cart_s = lat_s.get_cartesian_coords(contour_frac)
            contour_cart_s_imgs = tile_with_pbc_images(contour_cart_s, lat_s, shell=1)
            contour_tree = cKDTree(contour_cart_s_imgs)

        # 7.4 Loop over the surviving space groups.
        for sg in sg_list_s:
            if args.stop_after_first_scale and sg in finished_sg:
                verbose_print(f"[SKIP][s={s:.4f}] SG {sg} already succeeded at an earlier scale; skipping.")
                continue

            verbose_print(f"[SG] === Trying SG {sg} @ s={s:.4f} ===")

            # Sample symmetry-unique T sites.
            assembler = ConstrainedAssembler(
                trial_lattice=lat_s,
                contour_tree=contour_tree,
                spacegroup_number=sg,
                excl_radius=args.excl_radius,
                shell_min=args.shell_min,
                shell_max=args.shell_max,
                enable_shell=args.enable_shell,
                rng=rng,
                tt_min=args.tt_min,
            )
            try:
                unique_frac = assembler.sample_unique_sites(args.num_unique)
            except Exception as e:
                verbose_print(f"[FAIL][SG={sg}][s={s:.4f}] sampling unique sites failed: {e}")
                continue

            # Expand all T sites under symmetry.
            all_frac = expand_all_sites(unique_frac, sg)
            n_T = len(all_frac)
            vol = lat_s.volume
            t_per_1000 = (n_T / vol) * 1000.0
            verbose_print(f"[INFO][SG={sg}][s={s:.4f}] n_T={n_T}, Vol={vol:.3f} A^3, T/1000A^3={t_per_1000:.2f}")

            # Final density screening based on the actual number of T sites.
            if not (args.t_density_min <= t_per_1000 <= args.t_density_max):
                verbose_print(f"[REJECT][SG={sg}][s={s:.4f}] T-density {t_per_1000:.2f} not in "
                              f"[{args.t_density_min}, {args.t_density_max}]")
                continue

            # Build the nearest-four-neighbor graph and evaluate connectivity / CS.
            graph = build_undirected_graph_from_4nn(all_frac, lat_s)
            groups = group_sites_by_unique(unique_frac, sg, all_frac)
            ok_fourfold, deg_list, cs_list_all = check_unique_sites_degree_four(
                groups, graph, verbose=args.verbose,
            )
            if not ok_fourfold:
                verbose_print(f"[FAIL][SG={sg}][s={s:.4f}] "
                              f"At least one symmetry-unique T site is not strictly 4-connected (degree!=4).")
                continue

            cs_bad = any((len(cs_u) >= 2 and not is_strictly_increasing(cs_u)) for cs_u in cs_list_all)
            if cs_bad:
                verbose_print(f"[REJECT][SG={sg}][s={s:.4f}] CS not strictly increasing for at least one unique site; dropping.")
                continue

            o_frac = add_oxygen_on_edges_by_graph(all_frac, lat_s, graph)
            species_all = ["Si"] * len(all_frac) + ["O"] * len(o_frac)
            frac_all = np.vstack([all_frac, o_frac])

            td = os.path.join(args.output_dir, f"sg_{sg}", f"s_{s:.4f}")
            os.makedirs(td, exist_ok=True)
            out_path = os.path.join(td, "structure.cif")
            Structure(lat_s, species_all, frac_all).to(fmt="cif", filename=out_path)
            print(f"[OK][SG={sg}][s={s:.4f}] -> {out_path}")

            if args.stop_after_first_scale:
                finished_sg.add(sg)

            success_count += 1

    if success_count == 0:
        print("[INFO] No structures satisfied all filters.")



if __name__ == "__main__":
    main()
