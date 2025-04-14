#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.neighborlist import neighbor_list

# ─── Hyperparameters ────────────────────────────────────────────────────────────
CUT_OFF    = 6.0
WIDTH      = 0.5
ALPHA      = 2           # 0=x,1=y,2=z
DIM        = 8
STRATEGY   = 'weighted'  # or 'augmented'
WEIGHTTYPE = 'electronegativity'

pauling_en = {
    1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44,
    # Extend as needed
}

def cutoff_function(r, rc):
    return 0.5 * (np.cos(np.pi * r/rc) + 1.0) * (r < rc)

def load_site_positions(fn: Path, n_sites: int) -> np.ndarray:
    sp = np.loadtxt(fn)
    if sp.shape == (3, n_sites):
        return sp.T
    if sp.shape == (n_sites, 3):
        return sp
    raise ValueError(f"site_positions.dat has unexpected shape {sp.shape}")

def load_displacements(npzfile: Path, key: str) -> np.ndarray:
    arr = np.load(npzfile)[key]
    if arr.shape[1] != 3:
        raise ValueError(f"rc.npz['{key}'] has unexpected shape {arr.shape}")
    return arr  # shape is (n, 3)


def process_subdir(d: Path, outdir: Path):
    print(d)
    # ─── Load Base Info ─────────────────────────────────────────────────────────
    Zs      = np.loadtxt(d/'element.dat', dtype=int)
    n_sites = len(Zs)
    symbols = [chemical_symbols[z] for z in Zs]
    lat     = np.loadtxt(d/'lat.dat')  # 3x3 lattice
    baseline_pos = load_site_positions(d/'site_positions.dat', n_sites)

    # ─── Read displacements ─────────────────────────────────────────────────────
    rc_archive = np.load(d/'rc.npz')
    sample_keys = sorted(rc_archive.keys())
    n_samples = len(sample_keys)

    # ─── Determine descriptor size ──────────────────────────────────────────────
    if STRATEGY == 'weighted':
        fpsize = 2 * DIM
        atomtypes = None
    else:
        atomtypes = sorted(set(symbols))
        fpsize = DIM * len(atomtypes)

    X = np.zeros((n_samples, n_sites * fpsize))
    centers = np.linspace(0.0, CUT_OFF, DIM)

    for i, key in enumerate(sample_keys):
        disp = load_displacements(d/'rc.npz', key)  # (3, 3)

        # These are only displacements for 3 atoms → apply them to baseline
        coords = np.copy(baseline_pos)
        affected_atoms = [int(i) - 1 for i in key.strip("[]").split(",")[3:]]
        # print(affected_atoms)
        # Check for out-of-bound indices
        if any(i == baseline_pos.shape[0] for i in affected_atoms):
            print(f"⚠️  Skipping key {key} due to invalid atom indices: {affected_atoms}")

        disp = load_displacements(d/'rc.npz', key)
        disp = disp[:len(affected_atoms)]

        coords = np.copy(baseline_pos)
        coords[affected_atoms] += disp


        atoms = Atoms(symbols=symbols, positions=coords, cell=lat, pbc=True)
        i_list, j_list, dists, vecs = neighbor_list('ijdD', atoms, cutoff=CUT_OFF)

        desc = np.zeros((n_sites, fpsize))
        for idx in range(n_sites):
            mask = (i_list == idx)
            neighs = j_list[mask]
            R = dists[mask]
            V = vecs[mask]
            R_alpha = V[:, ALPHA]
            fc = cutoff_function(R, CUT_OFF)

            ptr = 0
            if STRATEGY == 'weighted':
                for wt in (None, WEIGHTTYPE):
                    for a_k in centers:
                        val = 0.0
                        for j, r, ra, f_c in zip(neighs, R, R_alpha, fc):
                            if wt == 'atomic_number':
                                weight = atomic_numbers[atoms[j].symbol]
                            elif wt == 'electronegativity':
                                Z = atomic_numbers[atoms[j].symbol]
                                weight = pauling_en.get(Z, 1.0)
                            else:
                                weight = 1.0
                            term = (
                                (ra/r)
                                * (1.0/(np.sqrt(2*np.pi)*WIDTH))
                                * np.exp(-0.5*((r-a_k)/WIDTH)**2)
                                * f_c
                            )
                            val += weight * term
                        desc[idx, ptr] = val
                        ptr += 1
            else:  # augmented
                for atype in atomtypes:
                    sel = [atoms[j].symbol == atype for j in neighs]
                    R_t = R[sel]
                    Ra_t = R_alpha[sel]
                    fc_t = fc[sel]
                    for a_k in centers:
                        val = np.sum(
                            (Ra_t/R_t)
                            * (1.0/(np.sqrt(2*np.pi)*WIDTH))
                            * np.exp(-0.5*((R_t-a_k)/WIDTH)**2)
                            * fc_t
                        )
                        desc[idx, ptr] = val
                        ptr += 1

        X[i, :] = desc.ravel()

    outfile = outdir / f"{d.name}_agni.npz"
    np.savez(outfile,
             fingerprints=X,
             sample_keys=sample_keys)
    print(f"  • {d.name}: {X.shape} → {outfile}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("root",  type=Path, help="root of graphene_dataset")
    p.add_argument("--out", type=Path, default=Path("agni_out"))
    args = p.parse_args()

    args.out.mkdir(exist_ok=True, parents=True)
    for sub in sorted(args.root.iterdir()):
        if sub.is_dir() and (sub/"element.dat").exists():
            process_subdir(sub, args.out)

if __name__ == "__main__":
    main()
