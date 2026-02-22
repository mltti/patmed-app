import streamlit as st
from patmed_app import main_page

if st.button("Wróć do predykcji"):
    st.switch_page(st.Page(main_page))
tab1, tab2, tab3, tab4, tab5 = st.tabs(["NN MLP", "RF1", "XGB1", "XGB2", "RDKit Ensamble"], default=st.session_state["info tab"])

with tab1:
    st.subheader("Neural Network Multilayer Perceptron")
    st.write("[placeholder description]")

with tab2:
    st.subheader("Model lasów losowych (Random Forest)")
    st.write("[placeholder description]")

with tab3:
    st.subheader("Model XGBoost 1")
    st.write("[placeholder description]")

with tab4:
    st.subheader("Model XGBoost 2")
    st.write("[placeholder description]")

with tab5:
    st.subheader("RDKit Ensamble")
    st.write("[placeholder description]")