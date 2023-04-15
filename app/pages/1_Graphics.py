import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('.')
sys.path.append('..')
sys.path.append('tfm_predictor')
from utils.opendata import OpenData
from app.home import App

# opd = OpenData()
def view_temp(df):
    if not isinstance(df, str):
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(nrows=4, figsize=(23,23))
        ax[0].grid()
        ax[0].bar(df.index, df["rain"]*4, label="Raining Day")
        ax[0].plot(df.index, df["tmed"], label="Media", color='green')
        ax[0].plot(df.index, df["tmin"], label="Mínima", color='blue')
        ax[0].plot(df.index, df["tmax"], label="Máxima", color='red')
        #ax[0].set_xlabel("Fecha")
        ax[0].set_ylabel("ºC")
        ax[0].set_title("Temperaturas diarias")
        ax[0].legend(ncols=2)

        ax[1].grid()
        ax[1].plot(df.index, df["velmedia"], label="Velocidad media", color='orange')
        ax[1].plot(df.index, df["racha"], label="Racha máxima", color='purple')
        ax[1].bar(df.index, df["rain"], label="Rain Day")
        #ax[1].set_xlabel("Fecha")
        ax[1].set_ylabel("m/s")
        ax[1].set_title("Condiciones del viento")
        ax[1].legend(ncols=3, framealpha=.4)

        ax[2].grid()
        ax[2].plot(df.index, df["dir"], label="Dirección de la racha máx.", color='cyan')
        ax[2].bar(df.index, df["rain"]*4, label="Rain Day")

        #ax[2].set_xlabel("Fecha")
        ax[2].set_ylabel("º (grados)")
        ax[2].set_title("Dirección")
        ax[2].legend(loc=2, framealpha=.4)

        ax[3].grid()
        ax[3].plot(df.index, df["presMax"], label="Presión Máxima", color='orange')
        ax[3].plot(df.index, df["presMin"], label="Presión Mínima", color='purple')
        # ax[3].bar(_df.index, _df["rain"]*900, label="Rain Day")
        #ax[3].set_xlabel("Fecha")
        ax[3].set_ylabel("h/Pa")
        ax[3].set_title("Presiones")
        ax[3].legend()

        st.pyplot(fig)
    else:
        st.write(df)



def main():
    st.set_page_config(layout="wide")

    with st.sidebar:
        opciones = ('Almeria','Cadiz','Cordoba','Granada',
                        'Huelva','Jaen','Malaga','Sevilla')
        years = (2023,2022,2021,2020,2019)

        st.markdown('''# TFM - Torres Espinosa\n ### Data Location:''')
        
        # Selectbox para elegir que datos ver
        provincia = st.selectbox("Province: ", opciones, 6)
        stations = app.opd.get_stations().query(
            f'provincia == "{(provincia.upper())}"')
        
        idema_name = st.selectbox( "Station: ", stations[['nombre']],
                                  len(stations)//2)
        idema = stations.query(f'nombre == "{idema_name}"'
                                            )['indicativo'].values[0]
        year = st.selectbox('Year:', years)
        app.opd.set_idema(idema)
        app.data = app.opd.get_year(year)

    # Título Pestaña
    st.title('Graphics')

    st.write(app.data)
    
    view_temp(app.data)

if __name__ == '__main__':
    app = App()
    main()

    