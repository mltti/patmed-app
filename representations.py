from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator,AdditionalOutput
import numpy as np
import pandas as pd

class Featurizer:
        def __init__(self, radius=2,n_bits=2048):
            self.radius = radius
            self.n_bits = n_bits

        def featurize(self,data):
            fingerprints = []
            rdkit_fps = []
            bitInfo = []
            for smiles in data['Drug']:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return np.zeros(self.n_bits)
                ao = AdditionalOutput()
                ao.AllocateBitInfoMap()
                generator = GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)
                fp = generator.GetFingerprint(mol,additionalOutput=ao)
                rdkit_fps.append(fp)
                fp_array = np.array(fp)
                fingerprints.append(fp_array)
                bitInfo.append(ao.GetBitInfoMap())
            return fingerprints, data['Y'], rdkit_fps, bitInfo