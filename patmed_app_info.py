import streamlit as st
from patmed_app import main_page

if st.button("Wróć do predykcji"):
    st.switch_page(st.Page(main_page))
tab1, tab2, tab3, tab4, tab5 = st.tabs(["NN MLP", "RF1", "XGB1", "XGB2", "RDKit Ensamble"], default=st.session_state["info tab"])

with tab1:
    st.subheader("Neural Network Multilayer Perceptron")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")

with tab2:
    st.subheader("Model lasów losowych (Random Forest)")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed posuere sed nisi in rhoncus. Nam eu justo eu ex laoreet egestas. In faucibus purus quam, dignissim sagittis purus facilisis ornare. Nunc ut sodales nisi. Morbi gravida vestibulum metus in efficitur. Vestibulum a massa convallis, lacinia quam ac, pharetra eros. Mauris tincidunt purus nisi. Aliquam congue augue auctor ex dictum ultricies. Donec molestie tellus ante, eget consectetur nisl tincidunt eu. Nunc sed turpis ut orci ultricies tincidunt. Donec lobortis tellus orci, ut tristique massa porta quis. Pellentesque vulputate sem in ante placerat, ac aliquam nibh tincidunt. Fusce a nulla consectetur, venenatis lacus sit amet, hendrerit lacus. Nam condimentum ante vitae justo efficitur fermentum. In quis ante ac nulla auctor convallis sit amet vitae urna.")

with tab3:
    st.subheader("Model XGBoost 1")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce sed consectetur tellus, quis convallis nunc. Quisque pellentesque dignissim velit at condimentum. Aenean dignissim, sapien nec sodales aliquet, magna eros pellentesque est, in mollis nunc nisi nec augue. Proin faucibus nibh nibh, quis faucibus erat lobortis ut. Ut sit amet leo sed sapien tempus porta. Morbi iaculis tincidunt ex at facilisis. Integer faucibus ultrices arcu. Vestibulum ornare neque nec nulla sagittis finibus vitae at dui.")

with tab4:
    st.subheader("Model XGBoost 2")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin vel congue metus. Proin varius massa id convallis lobortis. Nulla eros arcu, consectetur pharetra purus nec, ultricies tincidunt diam. Aliquam erat volutpat. Sed feugiat ex non dolor posuere viverra. Vestibulum laoreet nibh ligula, sed porttitor enim tristique vel. Donec posuere tortor pretium augue sollicitudin semper. Ut a nisi blandit, euismod orci vitae, venenatis justo. Maecenas id velit nec leo vestibulum sollicitudin sed efficitur urna. Nunc ac interdum ligula, id euismod eros. Integer auctor eros ipsum, vel semper purus sagittis vitae. Mauris vehicula finibus nisl eleifend tempus. Nullam porttitor dui augue, volutpat venenatis nisl hendrerit id. Nulla volutpat ante non varius suscipit. Proin porttitor egestas sem ac sollicitudin.")

with tab5:
    st.subheader("RDKit Ensamble")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Phasellus felis mauris, consequat et enim elementum, dignissim volutpat orci. Maecenas tempus ipsum vel aliquam blandit. Nam rutrum vitae ante vitae vulputate. Ut pharetra sagittis tortor. Cras et viverra ante. In ultricies tortor id congue volutpat. Cras sed nulla in lacus euismod porttitor. Quisque aliquet massa eget urna interdum, a placerat ante volutpat. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Donec nec consequat nisl. Praesent ut porttitor velit, vitae efficitur tellus. Nunc bibendum semper metus, nec rutrum tellus bibendum sit amet. Suspendisse et bibendum arcu, ut semper nibh. Nullam volutpat, velit sit amet lacinia vehicula, felis tortor cursus diam, a efficitur dolor erat at magna. Praesent egestas ut ligula sed commodo.")