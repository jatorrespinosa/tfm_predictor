# --- TFM - TorresEspinosa,JoseAntonio ---
import os
import sys
import streamlit as st
# import plotly.express as px
import time

from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('.')
sys.path.append('..')
sys.path.append('tfm_predictor')

from utils.opendata import OpenData

class App():
    def __init__(self):
        self.opd = OpenData()
        self.data = ''

   
    # --------------------------
    # -------- MAIN ------------
    # --------------------------
    def main(self):
        # Título Pestaña
        st.set_page_config(page_title="TFM", layout="wide")  # , page_icon="blue-flower.ico"

        opciones = ('Almeria','Cadiz','Cordoba','Granada',
                        'Huelva','Jaen','Malaga','Sevilla')
        years = (2023,2022,2021,2020,2019)
        # --- SideBar ---
        with st.sidebar:
            st.markdown('''# TFM - Torres Espinosa''')

    # --- Página central ---
        # Título
        st.title("TFM - Wheather Predictor")
        # Descripción
        st.write("Machine Learning App for predict wheather of Andalucía.\
                 Uses Aemet data through its Open Data Api.")

        st.title('Soruce Data:')
        st.subheader('Data Location:')
        
        c1, c2, c3 = st.columns(3)
        with c1:
            # Selectbox para elegir que datos ver
            provincia = st.selectbox('Province:', opciones, 6)
            stations = self.opd.get_stations().query(
                f'provincia == "{(provincia.upper())}"')
        with c2:
            idema_name = st.selectbox('Station:', stations[['nombre']],
                                      len(stations)//2)
            idema = stations.query(f'nombre == "{idema_name}"'
                                                )['indicativo'].values[0]
        with c3:
            year = st.selectbox('Year:', years)
            self.opd.set_idema(idema)
            self.data = self.opd.get_year(year, source=True)

        st.write(self.data)  


if __name__ == '__main__':
    main = App()
    main.main()