import shap
import json
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
from datetime import datetime
import os
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
from patmed.NN_models import MODEL_REGISTRY


def explain(SMILES, model_dir, number):

    # Loading representation module

    module_repr = __import__("patmed.representations",fromlist=["Representations"])

    config_path = os.path.join(model_dir, 'config.json')
    
    with open(config_path) as f:
        data_json = json.load(f)
    representation_name = data_json["args"]["representation"]
    representation = getattr(module_repr,representation_name)
    if representation_name == "Featurizer":
            
            featurizer = representation()

            # Preparing SMILES dataframe for featurization

            SMILES_df = pd.DataFrame(SMILES,columns=['Drug'])
            SMILES_df['Y'] = pd.NA

            # Featurizing SMILES

            SMILES_featurized, _, rdkit_fps, bitInfo = featurizer.featurize(SMILES_df)

            SMILES_featurized_t = torch.tensor(np.array(SMILES_featurized),dtype=torch.float32)

            # Getting feature names

            SMILES_featurized = pd.DataFrame(SMILES_featurized)

            # Loading model, making predictions and creating SHAP explainer

            if 'NN' in str(model_dir):
                model, optimizer, criterion, epochs = MODEL_REGISTRY[os.path.basename(model_dir)](input_size=int(np.asarray(SMILES_featurized).shape[1]))
                state_dict = torch.load(str(model_dir)+"\\weights.pth", weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()

                with torch.no_grad():
                            y_pred = model(SMILES_featurized_t)
                            y_pred = y_pred.cpu().flatten().numpy()

                # Loading training data used for model training for background dataset

                X_train_featurized_path = os.path.join(model_dir, 'X_train_featurized.csv')
                X_train_featurized = pd.read_csv(X_train_featurized_path).to_numpy()

                background = X_train_featurized[np.random.choice(X_train_featurized.shape[0], 100, replace=False)]
                background_tensor = torch.tensor(background, dtype=torch.float32)

                explainer = shap.GradientExplainer(model, background_tensor)

                shap_values = explainer(SMILES_featurized_t)

                shap_values.values = shap_values.values.squeeze(2)

            else:
                
                model = joblib.load(str(model_dir)+"\\model.pkl")

                y_pred = model.predict_proba(SMILES_featurized)[:,1].tolist()

                explainer = shap.TreeExplainer(model)

                shap_values = explainer(SMILES_featurized)

            shap_values.feature_names = SMILES_featurized.columns.tolist()

            # Generating highlights based on SHAP values

            highlights = []

            for smiles in SMILES:
                single_highlight = []

                # If predicted inactive, no highlights, empty lists

                if y_pred[SMILES.index(smiles)] < 0.5:
                    single_highlight = [[],[]]

                else:

                    mol = Chem.MolFromSmiles(smiles)

                    shap_vals = shap_values.values[0]

                    # Getting oredered list of SHAP values

                    order = np.argsort(np.abs(shap_vals))#[::-1]

                    # Getting bits present in the molecule ordered by decreasing SHAP value importance

                    present_bits = []

                    fp = rdkit_fps[0]

                    on_bits = fp.GetOnBits()

                    for bit in order:
                        if bit in on_bits:
                            present_bits.append(bit)

                    # Getting top contributing bits based on specified number
                    
                    important_bits = present_bits[:number]

                    # Highlighting atoms and bonds corresponding to important bits

                    highlight_atoms = set()
                    highlight_bonds = set()
                    
                    for bit in important_bits:
                        for atom_idx, radius in bitInfo[0][bit]:
                    
                            env = Chem.FindAtomEnvironmentOfRadiusN(
                                mol, radius, atom_idx)
                    
                            for bond_idx in env:
                                highlight_bonds.add(bond_idx)
                    
                                bond = mol.GetBondWithIdx(bond_idx)
                                highlight_atoms.add(bond.GetBeginAtomIdx())
                                highlight_atoms.add(bond.GetEndAtomIdx())
                    
                            # single atom -> radius = 0
                            if radius == 0:
                                highlight_atoms.add(atom_idx)

                    single_highlight.append(list(highlight_atoms))
                    single_highlight.append(list(highlight_bonds))

                highlights.append(single_highlight)
                
            return highlights
    
    else:
        raise ValueError("Bit mapping is only implemented for Featurizer representation.")

