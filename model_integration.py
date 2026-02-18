import argparse
import json
import joblib
import torch
import os
import random
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from NN_models import MODEL_REGISTRY
from explain_bits import explain

def predicted_toxicity(smiles_list, model, highlight_target):
    tox_data = []
    if model == "full":
        for i in range(len(smiles_list)):
            tox_data.append([smiles_list[i], 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, [[],[]]])
    elif model == "random":
        for i in range(len(smiles_list)):
            tox_data.append([smiles_list[i], (random.random() * 10)**2, (random.random() * 10)**2, (random.random() * 10)**2, (random.random() * 10)**2, (random.random() * 10)**2, (random.random() * 10)**2, (random.random() * 10)**2, (random.random() * 10)**2, [[],[]]])
    elif model == "none":
        for i in range(len(smiles_list)):
            tox_data.append([smiles_list[i], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [[],[]]])
    else:
        tox_CYP3A4 = predict(os.path.join("Models/CYP3A4", model), smiles_list)
        tox_CYP1A2 = predict(os.path.join("Models/CYP1A2", model), smiles_list)
        tox_CYP2C9 = predict(os.path.join("Models/CYP2C9", model), smiles_list)
        tox_CYP2D6 = predict(os.path.join("Models/CYP2D6", model), smiles_list)
        tox_CYP2C19 = predict(os.path.join("Models/CYP2C19", model), smiles_list)
        tox_hERG = predict(os.path.join("Models/hERG", model), smiles_list)
        tox_M2 = predict(os.path.join("Models/M2", model), smiles_list)
        tox_5HT2B = predict(os.path.join("Models/5HT2B", model), smiles_list)
        if highlight_target != "-":
            highlights = explain(smiles_list, os.path.join("Models", highlight_target, model), 6)
        for i in range(len(smiles_list)):
            if highlight_target == "-":
                highlight = [[],[]]
            else:
                highlight = highlights[i]
            tox_data.append([smiles_list[i], tox_CYP3A4[i]*100, tox_CYP1A2[i]*100, tox_CYP2C9[i]*100, tox_CYP2D6[i]*100, tox_CYP2C19[i]*100, tox_hERG[i]*100, tox_M2[i]*100, tox_5HT2B[i]*100, highlight])
    return tox_data

def predict(model_dir, dataset):
    module_repr = __import__("patmed.representations",fromlist=["Representations"])

    config_path = os.path.join(model_dir, 'config.json')
    
    with open(config_path) as f:
        data_json = json.load(f)
    representation_name = data_json["args"]["representation"]
    representation = getattr(module_repr,representation_name)
    if representation_name == "RDKitDescriptors":
        descriptor_name = data_json["args"]["descriptors"]       
        featurizer = representation(descriptor_name)
    else:
        featurizer = representation()

    X_test, Y, rdkit_fps, bitInfo = featurizer.featurize(dict({'Drug': dataset, 'Y': None}))
    X_test = np.array(X_test)

    if "NN" in model_dir:
        model, optimizer, criterion, epochs = MODEL_REGISTRY["NN_MLP"]()
        state_dict = torch.load(os.path.join(model_dir, "weights.pth"),weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        X_test = torch.tensor(X_test,dtype=torch.float32)
        with torch.no_grad():
            y_pred = model(X_test)
            y_pred = y_pred.flatten().numpy()
    elif "RDKit" in model_dir:
        model = joblib.load(os.path.join(model_dir, "model.pkl"))
        y_pred = model.predict(pd.DataFrame.from_dict(dict({'Drug': dataset})), newpath=os.path.join(model_dir, "feature_selectors"))
    else:
        model = joblib.load(os.path.join(model_dir, "model.pkl"))
        y_pred = model.predict(X_test)

    return y_pred