#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.neighborlist import neighbor_list
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ─── Hyperparameters ────────────────────────────────────────────────────────────
CUT_OFF    = 6.0
WIDTH      = 0.5
ALPHA      = 2           # direction index: 0=x,1=y,2=z
DIM        = 8
STRATEGY   = 'weighted'  # or 'augmented'
WEIGHTTYPE = 'electronegativity'  # or 'atomic_number', 'constant'

# Pauling electronegativities
PAULING_EN = {1:2.20, 6:2.55, 7:3.04, 8:3.44}


def load_site_positions(fn: Path, n_sites: int) -> np.ndarray:
    sp = np.loadtxt(fn)
    print(sp)
    if sp.shape == (3, n_sites):
        return sp.T
    if sp.shape == (n_sites, 3):
        return sp
    raise ValueError(f"site_positions.dat has unexpected shape {sp.shape}")


class FingerprintProperties:
    def __init__(self, cutoff, width, alpha, dimensionality, strategy, weight_type, atomtypes=None):
        self.cutoff = cutoff
        self.width = width
        self.alpha = alpha
        self.dimensionality = dimensionality
        self.strategy = strategy
        self.weight_type = weight_type
        self.atomtypes = atomtypes


class AGNICalculator:
    def __init__(self, props: FingerprintProperties):
        self.cutoff = props.cutoff
        self.width = props.width
        self.alpha = props.alpha
        self.dim = props.dimensionality
        self.strategy = props.strategy
        self.weight_type = props.weight_type
        self.atomtypes = props.atomtypes

        # determine fingerprint size
        if self.strategy == 'weighted':
            self.fpsize = self.dim * 2
        else:
            self.fpsize = self.dim * len(self.atomtypes)

        # centers equally spaced between 0 and cutoff
        self.centers = np.linspace(0.0, self.cutoff, self.dim)

    def cutoff_function(self, r: np.ndarray) -> np.ndarray:
        return 0.5 * (np.cos(np.pi * r / self.cutoff) + 1.0) * (r < self.cutoff)

    def calculate_component(self,
                            neighs: np.ndarray,
                            distances: np.ndarray,
                            vecs: np.ndarray,
                            a_k: float,
                            weight_mode: str,
                            atom_symbols: list) -> float:
        R = distances
        Ra = vecs[:, self.alpha]
        fc = self.cutoff_function(R)
        val = 0.0
        for r, ra, f_c, j in zip(R, Ra, fc, neighs):
            # determine weight
            if weight_mode == 'atomic_number':
                weight = atomic_numbers[atom_symbols[j]]
            elif weight_mode == 'electronegativity':
                Z = atomic_numbers[atom_symbols[j]]
                weight = PAULING_EN.get(Z, 1.0)
            else:
                weight = 1.0
            term = (ra/r) * (1.0/(np.sqrt(2*np.pi)*self.width)) * np.exp(-0.5*((r-a_k)/self.width)**2) * f_c
            val += weight * term
        return val

    def fingerprint_for_atom(self,
                              atom_idx: int,
                              i_list: np.ndarray,
                              j_list: np.ndarray,
                              distances: np.ndarray,
                              vecs: np.ndarray,
                              atom_symbols: list) -> np.ndarray:
        desc = np.zeros(self.fpsize)
        mask = (i_list == atom_idx)
        neighs = j_list[mask]
        R = distances[mask]
        V = vecs[mask]

        ptr = 0
        if self.strategy == 'weighted':
            for mode in [None, self.weight_type]:
                wt = mode if mode is not None else 'constant'
                for a_k in self.centers:
                    desc[ptr] = self.calculate_component(neighs, R, V, a_k, wt, atom_symbols)
                    ptr += 1
        else:  # augmented
            for atype in self.atomtypes:
                sel = [atom_symbols[j] == atype for j in neighs]
                neigh_sub = neighs[sel]
                R_sub = R[sel]
                V_sub = V[sel]
                for a_k in self.centers:
                    fc = self.cutoff_function(R_sub)
                    Ra = V_sub[:, self.alpha]
                    term = (Ra/R_sub) * (1.0/(np.sqrt(2*np.pi)*self.width)) * np.exp(-0.5*((R_sub-a_k)/self.width)**2) * fc
                    desc[ptr] = np.sum(term)
                    ptr += 1
        return desc


def process_subdir(d: Path, outdir: Path):
    # load static structure
    Zs      = np.loadtxt(d/'element.dat', dtype=int)
    n_sites = len(Zs)
    symbols = [chemical_symbols[z] for z in Zs]
    lat     = np.loadtxt(d/'lat.dat')
    baseline_pos = load_site_positions(d/'site_positions.dat', n_sites)

    # prepare fingerprint properties
    atomtypes = None
    if STRATEGY == 'augmented':
        atomtypes = sorted(set(symbols))
    props = FingerprintProperties(
        cutoff=CUT_OFF, width=WIDTH, alpha=ALPHA,
        dimensionality=DIM, strategy=STRATEGY,
        weight_type=WEIGHTTYPE, atomtypes=atomtypes
    )
    calculator = AGNICalculator(props)

    # build ASE Atoms and neighbor list
    atoms = Atoms(symbols=symbols, positions=baseline_pos, cell=lat, pbc=True)
    i_list, j_list, distances, vecs = neighbor_list('ijdD', atoms, cutoff=CUT_OFF)

    # compute fingerprints for each site
    desc = np.zeros((n_sites, calculator.fpsize))
    for idx in range(n_sites):
        desc[idx] = calculator.fingerprint_for_atom(
            atom_idx=idx,
            i_list=i_list, j_list=j_list,
            distances=distances, vecs=vecs,
            atom_symbols=symbols
        )

    # pack into single-sample array
    X = desc.ravel()[None, :]
    # print(X[0])
    sample_keys = ['baseline']

    # save output
    outfile = outdir / f"{d.name}_agni.npz"
    np.savez(outfile, fingerprints=X[0], sample_keys=sample_keys)
    print(f"Processed {d.name}: shape {X.shape} → {outfile}")


def main():
    parser = argparse.ArgumentParser(description="Compute AGNI fingerprints (static) for dataset subdirectories.")
    parser.add_argument("root", type=Path, help="Root directory of dataset")
    parser.add_argument("--out", type=Path, default=Path("agni_out"), help="Output directory")
    parser.add_argument("-j", "--jobs", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    args.out.mkdir(exist_ok=True, parents=True)
    subdirs = [sub for sub in sorted(args.root.iterdir()) if sub.is_dir() and (sub/'element.dat').exists()]

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for sub in subdirs:
            ex.submit(process_subdir, sub, args.out)

if __name__ == "__main__":
    main()
