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
sys.path.append('TFM')

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
        years = (2022,2021,2020,2019)
        # --- SideBar ---
        with st.sidebar:
            st.markdown('''# TFM - Torres Espinosa''')
            
            
            # # Selectbox para elegir que datos ver
            # provincia = st.selectbox("Province: ", opciones, 6)
            # stations = self.opd.get_stations().query(
            #     f'provincia == "{(provincia.upper())}"')
            
            # idema_name = st.selectbox( "Station: ", stations[['nombre']],
            #                           len(stations)//2)
            # idema = stations.query(f'nombre == "{idema_name}"'
            #                                     )['indicativo'].values[0]
            # year = st.selectbox('Year:', years)


    # --- Página central ---
        # Título
        st.title("TFM - Weather Predictor ")
        
        # Descripción
        st.write("Machine Learning app to predict the weather in Andalucía.\
                 Uses data from Aemet through Api Open Data.")


        
        st.title("Source Data:")
        st.subheader('Data Location:')
        c1, c2, c3 = st.columns(3)
        # Selectbox para elegir que datos ver
        with c1:
            provincia = st.selectbox("Province: ", opciones, 6)
            stations = self.opd.get_stations().query(
                f'provincia == "{(provincia.upper())}"')
        with c2:
            idema_name = st.selectbox( "Station: ", stations[['nombre']],
                                      len(stations)//2)
            idema = stations.query(f'nombre == "{idema_name}"'
                                                )['indicativo'].values[0]
        with c3:
            year = st.selectbox('Year:', years)
        
        self.opd.set_idema(idema)
        self.data = self.opd.get_year(year)

        st.write(f"Indicativo: {idema}\n")
        st.write(self.data)


if __name__ == '__main__':
    main = App()
    main.main()