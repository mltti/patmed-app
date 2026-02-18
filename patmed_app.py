import io
import os
from typing import Optional, Tuple, List
import math
import sys
import subprocess
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import pandas as pd
from model_integration import predicted_toxicity
import matplotlib.pyplot as plt
import numpy as np
from rdkit.Chem import Descriptors, Lipinski
import plotly.express as px
import numpy as np
import plotly.express as px
from rdkit.Chem import rdMolDescriptors

# --- Funkcja do generowania fingerprintów ---
def mol_to_fp(mol, nbits=1024):
    if mol is None:
        return np.zeros((nbits,))
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
# Bezpieczny import dla PCA/t-SNE/UMAP
_sklearn_available = False
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    _sklearn_available = True
except ImportError:
    pass

_umap_available = False
try:
    import umap
    _umap_available = True
except ImportError:
    pass
# Ustawienie stylu wykresów
plt.style.use("seaborn-v0_8-whitegrid")

def calc_properties(mol: Chem.Mol) -> dict:
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RB": Lipinski.NumRotatableBonds(mol)
    }
def render_expander_legend(title: str, smiles: list):
    """Wyświetla rozwijaną legendę z numerami cząsteczek."""
    with st.expander(f"Legenda ({title})"):
        for i, s in enumerate(smiles):
            st.markdown(f"**{i+1}** → `{s}`")
def mol_to_fp(mol, nbits=512):
    """Konwertuje cząsteczkę na wektor bitowy Morgan Fingerprint."""
    from rdkit.Chem import rdMolDescriptors
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nbits)
    arr = np.zeros((nbits,), dtype=int)
    for i, b in enumerate(fp):
        arr[i] = int(b)
    return arr

def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None

def mol_to_png_bytes(mol: Chem.Mol, size: Tuple[int, int] = (300, 300), atom_indices: bool = False, highlight_atoms: Optional[List] = None, highlight_bonds: Optional[List] = None) -> bytes:
    width, height = size
    # try Cairo drawer first
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        opts = drawer.drawOptions()
        opts.addAtomIndices = atom_indices
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass
        if highlight_atoms:
            hl_map = {int(a): (0.8, 0.2, 0.2) for a in highlight_atoms}
        else:
            hl_map = None
        if highlight_bonds:
            hlb_map = {int(a): (0.8, 0.2, 0.2) for a in highlight_atoms}
        else:
            hlb_map = None
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=highlight_atoms, higlightBonds=highlight_bonds, highlightAtomColors=hl_map, highlightBondColors=hlb_map)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()  # bytes or str
        if isinstance(png, str):
            png = png.encode('utf-8', errors='ignore')
        return png
    except Exception:
        # fallback: use PIL image from Draw.MolToImage
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass
        img = Draw.MolToImage(mol, size=(width, height))  # PIL Image
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        bio.seek(0)
        return bio.read()

def parse_uploaded_file(uploaded) -> List[str]:
    try:
        if uploaded is None:
            return []
        name = uploaded.name.lower()
        content = uploaded.getvalue()
        if name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
            # znajdź kolumnę ze SMILES
            candidates = [c for c in df.columns if 'smiles' in c.lower()]
            if candidates:
                col = candidates[0]
                return [str(s).strip() for s in df[col].dropna().astype(str).tolist()]
            else:
                # jeśli brak kolumny, spróbuj pierwszej kolumny
                col = df.columns[0]
                return [str(s).strip() for s in df[col].dropna().astype(str).tolist()]
        else:
            text = content.decode('utf-8')
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            # jeśli linie mają dodatkową kolumnę, weź pierwszą token
            smiles = [l.split()[0] for l in lines]
            return smiles
    except Exception:
        return []

def summary(toxicity, mol, highlight_a, highlight_b):
    st.subheader(toxicity[0])
    if mol is None:
        if (input_mode == "Wgraj plik z listą cząsteczek"):
            st.error("RDKit nie może wczytać pliku.")
        else:
            st.error("Niepoprawny kod SMILES.")
        return
    try:
        png_bytes = mol_to_png_bytes(mol, size=(width, height), highlight_atoms=highlight_a, highlight_bonds=highlight_b)
        st.image(png_bytes)
        # Ostrzeżenie jeśli model przewiduje toksyczność
        safe = True
        receptor_names = ["CYP3A4", "CYP1A2", "CYP2C9", "CYP2D6", "CYP2C19", "hERG", "receptorów M2", "receptorów 5-HT2B"]
        for i in range (1,9):
            if toxicity[i] >= tox_high and (st.session_state["tag_filter"] == "Tylko o toksyczności wysokiej" or
                                            st.session_state["tag_filter"] == "O toksyczności średniej lub wyższej" or
                                            st.session_state["tag_filter"] == "O toksyczności lekkiej lub wyższej" or
                                            st.session_state["tag_filter"] == "Wszystkie"):
                st.badge(f"Toksyczność wzgl. {receptor_names[i-1]}", icon=":material/skull:", color="red")
                safe = False
            elif toxicity[i] >= tox_medium and (st.session_state["tag_filter"] == "O toksyczności średniej lub wyższej" or
                                                st.session_state["tag_filter"] == "O toksyczności lekkiej lub wyższej" or
                                                st.session_state["tag_filter"] == "Wszystkie"):
                st.badge(f"Prawdopodobna toksyczność wzgl. {receptor_names[i-1]}", icon=":material/warning:", color="orange")
                safe = False
            elif toxicity[i] >= tox_low and (st.session_state["tag_filter"] == "O toksyczności lekkiej lub wyższej" or
                                             st.session_state["tag_filter"] == "Wszystkie"):
                st.badge(f"Możliwa toksyczność względem {receptor_names[i-1]}", icon=":material/warning:", color="yellow")
                safe = False                                        
        if safe == True and (st.session_state["tag_filter"] == "Tylko bez toksyczności" or
                             st.session_state["tag_filter"] == "Wszystkie"):
            st.badge("Brak przewidywanej toksyczności", icon=":material/done_outline:", color="green")
    except Exception as e:
        st.exception(e)

st.set_page_config(page_title="PATmed", layout="wide")
st.title("PATmed")
st.caption("Platforma do Analizy Toksyczności bioaktywnych związków małocząsteczkowych - Prototyp strony")
if "highlights" not in st.session_state:
    st.session_state["highlights"] = [[],[]]
with st.sidebar:
    st.header("Ustawienia")
    st.subheader("Rysunki RDKit")
    col1, col2 = st.columns([1,1])
    with col1:
        width = st.number_input("Szerokość (px):", min_value=100, max_value=2000, value=250)
    with col2:
        height = st.number_input("Wysokość (px):", min_value=100, max_value=2000, value=200)
    st.subheader("Progi Toksyczności (%):")
    tox_low = st.number_input("Próg lekkiej toksyczności (%):", min_value=1, max_value=100, value=25, label_visibility="collapsed")
    tox_medium = st.number_input("Próg średniej toksyczności (%):", min_value=1, max_value=100, value=50, label_visibility="collapsed")
    tox_high = st.number_input("Próg wysokiej toksyczności (%):", min_value=1, max_value=100, value=75, label_visibility="collapsed")

def main_page():
    tab_pred, tab_conf = st.tabs(["Predykcja", "Modele"])

    # --- Wybór modeli ---
    with tab_conf:
        st.subheader("Wybierz modele, które mają być użyte do predykcji:")
        col1, col2 = st.columns([1,1])
        with col1:
            select_0 = st.checkbox("(brak)", value=True)
            select_100 = st.checkbox("(100%)", value=True)
            select_random = st.checkbox("(losowo)", value=True)
            st.divider()
            select_NN = st.checkbox("NN MLP", value=False)
            select_RF1 = st.checkbox("RF1", value=False)
            select_XGB1 = st.checkbox("XGB1", value=False)
            select_XGB2 = st.checkbox("XGB2", value=False)
            select_RDKit = st.checkbox("RDKit Ensamble", value=False)
            if select_RDKit:
                st.warning("Brak modeli RDKitEnsamble dla receptorów hERG i M2.")
        with col2:
            st.write("NN MLP - Neural Network Multi-layered Perceptron.")
            if st.button("Więcej informacji", key=0):
                st.session_state["info tab"] = "NN MLP"
                st.switch_page("patmed_app_info.py")
            st.divider()
            st.write("RF1 - Model Lasu Losowego (Random Forest).")
            if st.button("Więcej informacji", key=1):
                st.session_state["info tab"] = "RF1"
                st.switch_page("patmed_app_info.py")
            st.divider()
            st.write("XGB1 - Model XGBoost 1.")
            if st.button("Więcej informacji", key=2):
                st.session_state["info tab"] = "XGB1"
                st.switch_page("patmed_app_info.py")
            st.divider()
            st.write("XGB2 - Model XGBoost 2.")
            if st.button("Więcej informacji", key=3):
                st.session_state["info tab"] = "XGB2"
                st.switch_page("patmed_app_info.py")
            st.divider()
            st.write("RDKit Ensamble.")
            if st.button("Więcej informacji", key=4):
                st.session_state["info tab"] = "RDKit"
                st.switch_page("patmed_app_info.py")
        
    # --- Wprowadzanie danych ---
    with tab_pred:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("Wprowadź jeden lub więcej kod SMILES lub wgraj plik .smi/.txt/.csv z listą cząsteczek.")          
            input_mode = st.radio("Tryb wejścia:", ("Pojedyncza cząsteczka", "Wiele cząsteczek", "Wgraj plik z listą cząsteczek"))
            smiles_input = ""
            uploaded = None
            if input_mode == "Pojedyncza cząsteczka":
                smiles_input = st.text_input("Kod SMILES:").strip()
            elif input_mode == "Wiele cząsteczek":
                smiles_input = st.text_area("Kody SMILES:")
            else:
                uploaded = st.file_uploader("Wgraj plik .smi/.txt/.csv", type=['smi', 'txt', 'csv'])

        with col2:
            st.header("Opcje wyświetlania")
            st.session_state["highlight_model"] = st.selectbox(
                "Podświetlanie podstruktury:",
                [
                    "-",
                    "NN_MLP",
                    "RF1",
                    "XGB1",
                    "XGB2",
                    "RDKit Ensamble"
                ],
                width=500
            )
            st.session_state["highlight_target"] = st.selectbox(
                "Podświetlanie podstruktury2:",
                [
                    "-",
                    "CYP3A4",
                    "CYP1A2",
                    "CYP2C9",
                    "CYP2D6",
                    "CYP2C19",
                    "hERG",
                    "M2",
                    "5HT2B"
                ],
                label_visibility="collapsed",
                width=500
            )
            st.session_state["tag_filter"] = st.selectbox(
                "Wyswietlaj cząsteczki:",
                [
                    "Wszystkie",
                    "Tylko o toksyczności wysokiej",
                    "O toksyczności średniej lub wyższej",
                    "O toksyczności lekkiej lub wyższej",
                    "Tylko bez toksyczności"
                ],
                width=500
            )
            st.session_state["warning_mode"] = st.selectbox(
                "Tagi toksyczności według:",
                [
                    "Uśrednionych predykcji",
                    "Najwyższych predykcji",
                    "Najniższych predykcji"
                ],
                width=500
            )
        
        # --- PRZYCISK (POZA KOLUMNAMI!) ---
        if "generated" not in st.session_state:
            st.session_state["generated"] = False
        if st.button("Sprawdź toksyczność"):
            st.session_state["generated"] = True

        # --- Przetwarzanie danych ---
        smiles_list: List[str] = []
        tab1, tab2, tab3, tab4 = st.tabs(["Podsumowanie", "Szczegóły", "Analiza zbioru", "Wizualizacja t-SNE/UMAP"])
        if not select_0 and not select_100 and not select_random and not select_NN and not select_RF1 and not select_XGB1 and not select_XGB2 and not select_RDKit:
            st.error("Nie wybrano modeli - wybierz modele w zakładce 'Modele' powyżej.")
            st.stop()
        if st.session_state["generated"]:
            if uploaded is not None:
                smiles_list = parse_uploaded_file(uploaded)
            elif smiles_input:
                smiles_list = [s.strip() for s in smiles_input.splitlines() if s.strip()]
            if not smiles_list:
                if "smiles_list" in st.session_state:
                    smiles_list = st.session_state["smiles_list"]
                else:
                    st.error("Nie wprowadzono cząsteczek — wprowadź kod SMILES lub wgraj plik.")
                    st.stop()
            else:
                st.session_state["smiles_list"] = smiles_list
            
            tox_dataset = []
            if select_0:
                tox_data_0 = predicted_toxicity(smiles_list, "none", "-")
                st.session_state["tox_data_0"] = tox_data_0
                tox_dataset.append(tox_data_0)
            if select_100:
                tox_data_100 = predicted_toxicity(smiles_list, "full", "-")
                st.session_state["tox_data_100"] = tox_data_100
                tox_dataset.append(tox_data_100)
            if select_random:
                tox_data_random = predicted_toxicity(smiles_list, "random", "-")
                st.session_state["tox_data_random"] = tox_data_random
                tox_dataset.append(tox_data_random)
            if select_NN:
                tox_data_NN = predicted_toxicity(smiles_list, "NN_MLP", st.session_state["highlight_target"])
                st.session_state["tox_data_NN"] = tox_data_NN
                tox_dataset.append(tox_data_NN)
                if st.session_state["highlight_model"] == "NN_MLP":
                    tox_data_hl = []
                    for entry in tox_data_NN_MLP:
                        tox_data_hl.append(entry[9])
                    st.session_state["highlights"] = tox_data_hl
            if select_RF1:
                tox_data_RF1 = predicted_toxicity(smiles_list, "RF1", st.session_state["highlight_target"])
                st.session_state["tox_data_RF1"] = tox_data_RF1
                tox_dataset.append(tox_data_RF1)
                if st.session_state["highlight_model"] == "RF1":
                    tox_data_hl = []
                    for entry in tox_data_RF1:
                        tox_data_hl.append(entry[9])
                    st.session_state["highlights"] = tox_data_hl
            if select_XGB1:
                tox_data_XGB1 = predicted_toxicity(smiles_list, "XGB1", st.session_state["highlight_target"])
                st.session_state["tox_data_XGB1"] = tox_data_XGB1
                tox_dataset.append(tox_data_XGB1)
                if st.session_state["highlight_model"] == "XGB1":
                    tox_data_hl = []
                    for entry in tox_data_XGB1:
                        tox_data_hl.append(entry[9])
                    st.session_state["highlights"] = tox_data_hl
            if select_XGB2:
                tox_data_XGB2 = predicted_toxicity(smiles_list, "XGB2", st.session_state["highlight_target"])
                st.session_state["tox_data_XGB2"] = tox_data_XGB2
                tox_dataset.append(tox_data_XGB2)
                if st.session_state["highlight_model"] == "XGB2":
                    tox_data_hl = []
                    for entry in tox_data_XGB2:
                        tox_data_hl.append(entry[9])
                    st.session_state["highlights"] = tox_data_hl
            if select_RDKit:
                tox_data_RDKit = predicted_toxicity(smiles_list, "RDKit", st.session_state["highlight_target"])
                st.session_state["tox_data_RDKit"] = tox_data_RDKit
                tox_dataset.append(tox_data_RDKit)
                if st.session_state["highlight_target"] == "RDKit":
                    tox_data_hl = []
                    for entry in tox_data_RDKit:
                        tox_data_hl.append(entry[9])
                    st.session_state["highlights"] = tox_data_hl

            tox_data_max = []
            tox_data_min = []
            tox_data_avg = []
            for j in range(len(smiles_list)):
                entry_max = [smiles_list[j]]
                entry_min = [smiles_list[j]]
                entry_avg = [smiles_list[j]]
                for k in range(1,9):
                    set = []
                    for i in range(len(tox_dataset)):
                        set.append(tox_dataset[i][j][k])
                    entry_max.append(max(set))
                    entry_min.append(min(set))
                    entry_avg.append(sum(set)/len(set))
                tox_data_max.append(entry_max)
                tox_data_min.append(entry_min)
                tox_data_avg.append(entry_avg)
        
            with tab1:
                tox_data_filtered = []
                highlights_filtered = []
                if st.session_state["warning_mode"] == "Najwyższych predykcji":
                    tox_data_tagged = tox_data_max
                elif st.session_state["warning_mode"] == "Najniższych predykcji":
                    tox_data_tagged = tox_data_min
                else:
                    tox_data_tagged = tox_data_avg
                for i in range(len(tox_data_tagged)):
                    if st.session_state["tag_filter"] == "Wszystkie":
                        tox_data_filtered.append(tox_data_tagged[i])  
                        if st.session_state["highlight_model"] != "-":
                            highlights_filtered.append(st.session_state["highlights"][i])
                    elif st.session_state["tag_filter"] == "Tylko o toksyczności wysokiej" and max(tox_data_tagged[i][1:9]) >= tox_high:
                        tox_data_filtered.append(tox_data_tagged[i])
                        if st.session_state["highlight_model"] != "-":
                            highlights_filtered.append(st.session_state["highlights"][i])
                    elif st.session_state["tag_filter"] == "O toksyczności średniej lub wyższej" and max(tox_data_tagged[i][1:9]) >= tox_medium:
                        tox_data_filtered.append(tox_data_tagged[i])
                        if st.session_state["highlight_model"] != "-":
                            highlights_filtered.append(st.session_state["highlights"][i])
                    elif st.session_state["tag_filter"] == "O toksyczności lekkiej lub wyższej" and max(tox_data_tagged[i][1:9]) >= tox_low:
                        tox_data_filtered.append(tox_data_tagged[i])
                        if st.session_state["highlight_model"] != "-":
                            highlights_filtered.append(st.session_state["highlights"][i])
                    elif st.session_state["tag_filter"] == "Tylko bez toksyczności" and max(tox_data_tagged[i][1:9]) < tox_low:
                        tox_data_filtered.append(tox_data_tagged[i])
                        if st.session_state["highlight_model"] != "-":
                            highlights_filtered.append(st.session_state["highlights"][i])
                mols = []
                for tox in tox_data_filtered:
                    mols.append(mol_from_smiles(tox[0]))
                column0, column1, column2 = st.columns([1,1,1])
                for i, tox in enumerate(tox_data_filtered):
                    if i % 3 == 0:
                        with column0:
                            if st.session_state["highlight_model"] != "-":
                                summary(tox, mols[i], highlights_filtered[i][0], highlights_filtered[i][1])
                            else:
                                summary(tox, mols[i], [], [])
                    elif i % 3 == 1:
                        with column1:
                            if st.session_state["highlight_model"] != "-":
                                summary(tox, mols[i], highlights_filtered[i][0], highlights_filtered[i][1])
                            else:
                                summary(tox, mols[i], [], [])
                    else:
                        with column2:
                            if st.session_state["highlight_model"] != "-":
                                summary(tox, mols[i], highlights_filtered[i][0], highlights_filtered[i][1])
                            else:
                                summary(tox, mols[i], [], [])

            with tab2:
                table_choices = ["Uśrednione predykcje"]
                if select_0:
                    table_choices.append("(brak)")
                if select_100:
                    table_choices.append("(100%)")
                if select_random:
                    table_choices.append("(losowo)")
                if select_NN:
                    table_choices.append("NN MLP")
                if select_RF1:
                    table_choices.append("RF1")
                if select_XGB1:
                    table_choices.append("XGB1")
                if select_XGB2:
                    table_choices.append("XGB2")
                if select_RDKit:
                    table_choices.append("RDKit Ensamble")
                table_model = st.selectbox("Wyświetl predykcje wg. modelu:", table_choices, width=300)
                if table_model == "Uśrednione predykcje":
                    st.write("Uśrednione predykcje toksyczności (%) względem wybranych targetów:")
                    data = tox_data_avg
                else:
                    st.write("Predykcja toksyczności (%) względem wybranych targetów według", table_model, ":")
                    if table_model == "(brak)":
                        raw_data = st.session_state["tox_data_0"]
                    elif table_model == "(100%)":
                        raw_data = st.session_state["tox_data_100"]
                    elif table_model == "(losowo)":
                        raw_data = st.session_state["tox_data_random"]
                    elif table_model == "NN MLP":
                        raw_data = st.session_state["tox_data_NN"]
                    elif table_model == "RF1":
                        raw_data = st.session_state["tox_data_RF1"]
                    elif table_model == "XGB1":
                        raw_data = st.session_state["tox_data_XGB1"]
                    elif table_model == "XGB2":
                        raw_data = st.session_state["tox_data_XGB2"]
                    elif table_model == "RDKit Ensamble":
                        raw_data = st.session_state["tox_data_RDKit"]
                    data = []
                    for datapoint in raw_data:
                        data.append(datapoint[:9])
                df = pd.DataFrame(
                    data,
                    columns=[
                        "SMILES", "CYP3A4", "CYP1A2", "CYP2C9", "CYP2D6",
                        "CYP2C19", "hERG", "M2", "5-HT2B"
                    ]
                )
                st.dataframe(df)
            
                col1, col2 = st.columns([3, 1])
                with col2:
                    format_choice = st.selectbox("Wybierz format pliku:", ["CSV", "TXT", "PDF"])
                    if format_choice == "CSV":
                        file_bytes = df.to_csv(index=False).encode("utf-8")
                        filename = "toxicity_results.csv"
                    elif format_choice == "TXT":
                        file_bytes = df.to_string(index=False).encode("utf-8")
                        filename = "toxicity_results.txt"
                    elif format_choice == "PDF":
                        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
                        from reportlab.lib.pagesizes import letter, landscape
                        from reportlab.lib.styles import ParagraphStyle
                        from reportlab.lib import colors
                        import tempfile

                        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
                        pdf = SimpleDocTemplate(
                            temp_path,
                            pagesize=landscape(letter),
                            leftMargin=20,
                            rightMargin=20,
                            topMargin=20,
                            bottomMargin=20
                        )
                        wrap_style = ParagraphStyle(
                            name='WrapStyle',
                            fontSize=7,
                            leading=8
                        )
                        wrapped_data = []
                        wrapped_data.append(df.columns.tolist())  # nagłówki bez zmian
                        for row in df.values.tolist():
                            new_row = []
                            for cell in row:
                                cell = str(cell)
                                if len(cell) > 10:   # SMILES i inne długie teksty
                                    new_row.append(Paragraph(cell, wrap_style))
                                else:
                                    new_row.append(cell)
                            wrapped_data.append(new_row)
                        table = Table(wrapped_data)
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTSIZE', (0, 0), (-1, -1), 8),
                        ]))
                        initial_width = 80
                        table._argW = [initial_width] * len(df.columns)
                        available_width = landscape(letter)[0] - pdf.leftMargin - pdf.rightMargin
                        current_width = sum(table._argW)
                        if current_width > available_width:
                            scale_factor = available_width / current_width
                            table._argW = [w * scale_factor for w in table._argW]
                        pdf.build([table])
                        with open(temp_path, "rb") as f:
                            file_bytes = f.read()
                        filename = "toxicity_results.pdf"

                    # --- Przycisk pobierania---
                    st.download_button(
                        key=f"download_button_{format_choice}",
                        label="⬇ Pobierz",
                        data=file_bytes,
                        file_name=filename,
                        mime="application/octet-stream",
                        use_container_width=True
                    )
            with tab3:
                            st.header("Rozkłady właściwości fizykochemicznych")
                            
                            # Przygotowanie danych (mols i smiles_list są już w st.session_state lub zmiennych lokalnych)
                            pairs = []
                            for idx, (s, m) in enumerate(zip(smiles_list, mols)):
                                if m is not None:
                                    props = calc_properties(m)
                                    props["SMILES"] = s
                                    props["Index"] = idx + 1
                                    pairs.append(props)

                            if not pairs:
                                st.warning("Brak poprawnych cząsteczek do analizy.")
                            else:
                                df_props = pd.DataFrame(pairs)
                                st.dataframe(df_props[["Index", "SMILES", "MW", "LogP", "TPSA", "HBD", "HBA", "RB"]])

                                fig_cols = st.columns(3)

                                # --- Wykresy dla wartości ciągłych (MW, LogP, TPSA) ---
                                def draw_property_bar(prop_name, column_obj, color="#1f77b4"):
                                    with column_obj:
                                        fig, ax = plt.subplots(figsize=(5, 3))
                                        x = np.arange(len(df_props))
                                        ax.bar(x, df_props[prop_name], color=color, edgecolor="black")
                                        ax.set_xticks(x)
                                        ax.set_xticklabels(df_props["Index"])
                                        ax.set_title(f"Wartości {prop_name}")
                                        ax.set_ylabel(prop_name)
                                        fig.tight_layout()
                                        st.pyplot(fig)
                                        render_expander_legend(prop_name, df_props["SMILES"].tolist())

                                draw_property_bar("MW", fig_cols[0])
                                draw_property_bar("LogP", fig_cols[1], color="#2ca02c")
                                draw_property_bar("TPSA", fig_cols[2], color="#9467bd")

                                st.divider()
                                fig_cols_discrete = st.columns(3)

                                # --- Wykresy dla wartości dyskretnych (HBD, HBA, RB) ---
                                def draw_discrete_hist(prop_name, column_obj):
                                    with column_obj:
                                        vals = df_props[prop_name].values
                                        fig, ax = plt.subplots(figsize=(5, 3))
                                        min_v, max_v = int(np.min(vals)), int(np.max(vals))
                                        bins = range(min_v, max_v + 2)
                                        ax.hist(vals, bins=bins, color="#ff7f0e", edgecolor="black", align="left", rwidth=0.8)
                                        ax.set_xticks(range(min_v, max_v + 1))
                                        ax.set_title(f"Rozkład {prop_name}")
                                        ax.set_ylabel("Liczba cząsteczek")
                                        fig.tight_layout()
                                        st.pyplot(fig)
                                        
                                        # Specjalna legenda dla histogramu
                                        mapping = {}
                                        for i, val in enumerate(vals):
                                            mapping.setdefault(int(val), []).append(f"**{i+1}**")
                                        with st.expander(f"Legenda ({prop_name})"):
                                            for k in sorted(mapping.keys()):
                                                st.markdown(f"Wartość **{k}**: cząsteczki " + ", ".join(mapping[k]))

                                draw_discrete_hist("HBD", fig_cols_discrete[0])
                                draw_discrete_hist("HBA", fig_cols_discrete[1])
                                draw_discrete_hist("RB", fig_cols_discrete[2])
            with tab4:
                            if not _sklearn_available:
                                st.error("Brak pakietu scikit-learn. Zainstaluj go: pip install scikit-learn")
                            else:
                                st.header("t-SNE / UMAP — wizualizacja podobieństwa cząsteczek")
                                st.caption("Punkty są kolorowane według poziomu toksyczności wybranego w opcjach (Max/Min/Avg).")

                                # --- Przygotowanie danych z uwzględnieniem wybranych modeli ---
                                # Używamy tox_data_tagged, które zawiera już przetworzone dane wg. wyboru użytkownika
                                valid_data = []
                                for i, mol in enumerate(mols):
                                    if mol is not None:
                                        # Pobieramy dane dla konkretnej cząsteczki z tox_data_tagged
                                        # tox_data_tagged[i] to [SMILES, CYP3A4, ..., 5-HT2B]
                                        valid_data.append({
                                            "mol": mol,
                                            "smiles": smiles_list[i],
                                            "tox_values": tox_data_tagged[i][1:] # Same wartości numeryczne
                                        })

                                n_samples = len(valid_data)
                                max_samples = 500 # Limit dla płynności działania

                                if n_samples < 3:
                                    st.warning("Potrzeba co najmniej 3 poprawnych cząsteczek do wizualizacji.")
                                else:
                                    # --- Przygotowanie macierzy ---
                                    fps = np.array([mol_to_fp(d["mol"]) for d in valid_data])
                                    tox_matrix = np.array([d["tox_values"] for d in valid_data])
                                    smiles_local = [d["smiles"] for d in valid_data]
                                    
                                    tox_labels = ["CYP3A4", "CYP1A2", "CYP2C9", "CYP2D6", "CYP2C19", "hERG", "M2", "5-HT2B"]
                                    overall_toxicity = np.max(tox_matrix, axis=1)

                                    # --- Obliczenia rzutowania (PCA, t-SNE, UMAP) ---
                                    with st.spinner("Trwa obliczanie mapy przestrzeni chemicznej..."):
                                        # PCA
                                        pca = PCA(n_components=2)
                                        pca_emb = pca.fit_transform(fps)
                                        var_exp = pca.explained_variance_ratio_ * 100

                                        # t-SNE
                                        perp = min(30, max(2, n_samples // 3))
                                        tsne_emb = TSNE(
                                            n_components=2, perplexity=perp, init="pca", random_state=42
                                        ).fit_transform(fps)

                                        # UMAP
                                        umap_emb = None
                                        if _umap_available:
                                            n_neigh = min(15, n_samples - 1)
                                            reducer = umap.UMAP(n_neighbors=n_neigh, min_dist=0.1, random_state=42)
                                            umap_emb = reducer.fit_transform(fps)

                                    # --- Przygotowanie DataFrame do wykresu ---
                                    df_plot_base = pd.DataFrame({
                                        "SMILES": smiles_local,
                                        "Overall toxicity": overall_toxicity
                                    })
                                    for i, label in enumerate(tox_labels):
                                        df_plot_base[label] = tox_matrix[:, i]

                                    # Klasyfikacja kolorystyczna
                                    def get_tox_class(v):
                                        if v < tox_low: return "Brak toksyczności"
                                        if v < tox_medium: return "Niska toksyczność"
                                        if v < tox_high: return "Średnia toksyczność"
                                        return "Wysoka toksyczność"

                                    df_plot_base["Klasa toksyczności"] = [get_tox_class(v) for v in overall_toxicity]
                                    tox_colors = {"Brak toksyczności": "#4CCF50", "Niska toksyczność": "#FFD107", "Średnia toksyczność": "#FF9107", "Wysoka toksyczność": "#F44336"}

                                    # --- Funkcja rysująca (identyczna z Twoim wzorem) ---
                                    def interactive_plot(emb, title, xlab="X", ylab="Y"):
                                        df_plot = df_plot_base.copy()
                                        df_plot["x"] = emb[:, 0]
                                        df_plot["y"] = emb[:, 1]

                                        # Generowanie Hover Text z kropkami
                                        h_texts = []
                                        for _, row in df_plot.iterrows():
                                            parts = [f"<b>SMILES:</b> {row['SMILES']}"]
                                            for lbl in tox_labels:
                                                v = row[lbl]
                                                c = tox_colors[get_tox_class(v)]
                                                parts.append(f"<span style='color:{c}'>●</span> {lbl}: {v:.1f}%")
                                            h_texts.append("<br>".join(parts))
                                        df_plot["hover"] = h_texts

                                        fig = px.scatter(
                                            df_plot, x="x", y="y", color="Klasa toksyczności",
                                            color_discrete_map=tox_colors,
                                            title=title,
                                            category_orders={"Klasa toksyczności": ["Brak toksyczności", "Niska toksyczność", "Średnia toksyczność", "Wysoka toksyczność"]}
                                        )
                                        fig.update_traces(
                                            marker=dict(size=10, line=dict(width=0.7, color="black")),
                                            hovertemplate="%{customdata[0]}<extra></extra>",
                                            customdata=df_plot[["hover"]]
                                        )
                                        fig.update_layout(height=600, plot_bgcolor="white", xaxis_title=xlab, yaxis_title=ylab)
                                        
                                        # Dodanie brakujących legend (jeśli w zbiorze nie ma np. wysokiej toksyczności)
                                        for name, color in tox_colors.items():
                                            if name not in df_plot["Klasa toksyczności"].unique():
                                                fig.add_scatter(x=[None], y=[None], mode='markers', 
                                                                marker=dict(size=10, color=color), name=name)
                                        
                                        st.plotly_chart(fig, use_container_width=True)

                                    # Wyświetlanie wykresów jeden pod drugim
                                    st.subheader("PCA")
                                    interactive_plot(pca_emb, "Przestrzeń Chemiczna — PCA", f"PC1 ({var_exp[0]:.1f}%)", f"PC2 ({var_exp[1]:.1f}%)")
                                    
                                    st.subheader("t-SNE")
                                    interactive_plot(tsne_emb, "Przestrzeń Chemiczna — t-SNE", "t-SNE 1", "t-SNE 2")

                                    if umap_emb is not None:
                                        st.subheader("UMAP")
                                        interactive_plot(umap_emb, "Przestrzeń Chemiczna — UMAP", "UMAP 1", "UMAP 2")

pg = st.navigation([main_page, "patmed_app_info.py"], position="hidden")
pg.run()