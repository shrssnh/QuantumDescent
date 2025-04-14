import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.data import atomic_numbers

pauling_en = {
    1: 2.20,   2: 0.0,   3: 0.98,  4: 1.57,  5: 2.04,  6: 2.55,  7: 3.04,
    8: 3.44,   9: 3.98, 10: 0.0,  11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90,
   15: 2.19,  16: 2.58, 17: 3.16, 18: 0.0,  19: 0.82, 20: 1.00,
}


def cutoff_function(r, rc):
    """
    Smooth cutoff: 
      0.5 * (cos(pi * r / rc) + 1)   for r < rc,
      0                            otherwise.
    """
    return 0.5 * (np.cos(np.pi * r / rc) + 1.0) * (r < rc)


class AGNICalculator:
    def __init__(self,
                 atoms: Atoms,
                 cutoff: float,
                 width: float,
                 alpha: int,
                 dim: int,
                 strategy: str = 'weighted',
                 weight_type: str = 'atomic_number',
                 atomtypes: list[str] | None = None):
        """
        Parameters
        ----------
        atoms
          ASE Atoms object.
        cutoff
          radial cutoff (rc).
        width
          Gaussian width (w).
        alpha
          index 0,1,2 for which Cartesian component to project onto.
        dim
          number of Gaussian centers (dimensionality).
        strategy
          'weighted'  or  'augmented'
        weight_type
          one of 'atomic_number', 'electronegativity', or anything else→constant=1
        atomtypes
          list of element symbols for 'augmented'; if None, inferred from atoms.
        """
        self.atoms = atoms
        self.cutoff = cutoff
        self.width = width
        self.alpha = alpha
        self.dim = dim
        self.strategy = strategy
        self.weight_type = weight_type

        if atomtypes is None:
            atomtypes = sorted(set(atoms.get_chemical_symbols()))
        self.atomtypes = atomtypes
        self.natomtypes = len(atomtypes)

        self.centers = np.linspace(0.0, cutoff, dim)

        if strategy == 'weighted':
            self.fpsize = 2 * dim
        elif strategy == 'augmented':
            self.fpsize = dim * self.natomtypes
        else:
            raise ValueError("strategy must be 'weighted' or 'augmented'")

    def calculate(self) -> np.ndarray:
        """
        Compute AGNI fingerprint for all atoms.
        
        Returns
        -------
        desc : array, shape (n_atoms, fpsize)
          The AGNI descriptor matrix.
        """
        n_atoms = len(self.atoms)
        desc = np.zeros((n_atoms, self.fpsize))

        i_list, j_list, distances, vectors = neighbor_list(
            'ijdD', self.atoms, cutoff=self.cutoff
        )

        for idx in range(n_atoms):
            mask = (i_list == idx)
            neigh_js = j_list[mask]
            r_ijs = distances[mask]
            vecs = vectors[mask]
            r_alphas = vecs[:, self.alpha]
            fc = cutoff_function(r_ijs, self.cutoff)

            pos = 0
            if self.strategy == 'weighted':
                for wt in [None, self.weight_type]:
                    for a_k in self.centers:
                        val = 0.0
                        for j, R, R_alpha, f_c in zip(neigh_js, r_ijs, r_alphas, fc):
                            if wt == 'atomic_number':
                                weight = atomic_numbers[self.atoms[j].symbol]
                            elif wt == 'electronegativity':
                                Z = atomic_numbers[self.atoms[j].symbol]
                                weight = pauling_en.get(Z, 1.0)
                            else:
                                weight = 1.0
                            term = (
                                (R_alpha / R)
                                * (1.0 / (np.sqrt(2*np.pi) * self.width))
                                * np.exp(-0.5 * ((R - a_k) / self.width)**2)
                                * f_c
                            )
                            val += weight * term
                        desc[idx, pos] = val
                        pos += 1

            else:
                for atype in self.atomtypes:
                    mask_type = [self.atoms[j].symbol == atype for j in neigh_js]
                    r_t  = r_ijs[mask_type]
                    ra_t = r_alphas[mask_type]
                    fc_t = fc[mask_type]
                    js_t = [j for j,m in zip(neigh_js, mask_type) if m]

                    for a_k in self.centers:
                        val = 0.0
                        for j, R, R_alpha, f_c in zip(js_t, r_t, ra_t, fc_t):
                            term = (
                                (R_alpha / R)
                                * (1.0 / (np.sqrt(2*np.pi) * self.width))
                                * np.exp(-0.5 * ((R - a_k) / self.width)**2)
                                * f_c
                            )
                            val += term
                        desc[idx, pos] = val
                        pos += 1

        return desc


from ase.build import molecule

atoms = molecule('H2O')

calc = AGNICalculator(
    atoms,
    cutoff=6.0,
    width=0.5,
    alpha=2,
    dim=8,
    strategy='weighted',       # or 'augmented'
    weight_type='electronegativity'
)

features = calc.calculate()
print("AGNI features shape:", features.shape)
print(features)
