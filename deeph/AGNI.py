import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

def cutoff_function(r, rc):
    return 0.5 * (np.cos(np.pi * r / rc) + 1) * (r < rc)

def agni_descriptor(atoms, eta_list, rc):
    positions = atoms.get_positions()
    n_atoms = len(atoms)
    desc = np.zeros((n_atoms, len(eta_list)))

    i_list, j_list, distances = neighbor_list('ijd', atoms, cutoff=rc)

    for idx in range(n_atoms):
        mask = i_list == idx
        r_ij = distances[mask]
        fc = cutoff_function(r_ij, rc)

        for k, eta in enumerate(eta_list):
            desc[idx, k] = np.sum(np.exp(-eta * (r_ij**2)) * fc)

    return desc

from ase.build import molecule

atoms = molecule('H2O')
eta_values = np.linspace(0.01, 4.0, 8)  # Like Gaussian widths
cutoff_radius = 6.0

features = agni_descriptor(atoms, eta_values, cutoff_radius)
print("AGNI features:\n", features)
