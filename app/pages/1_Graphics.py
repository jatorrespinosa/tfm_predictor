# ----------------------------------------------------------------------------
#   File to see the data graphics.
#   Torres Espinosa, Jose Antonio.
# ----------------------------------------------------------------------------

# Dependencies
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Own Modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('.')
sys.path.append('..')
sys.path.append('tfm_predictor')
from utils.opendata import OpenData
from app.home import App

# opd = OpenData()
def view_graphs(df):
    """Creates graphics for temperature, wind and pressure data.

    :param df: formatted data for opendata() one or several years.
    :type df: DataFrame
    """
    if not isinstance(df, str):  # If contains data
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(nrows=4, figsize=(23,23))

        # - Temperature -
        ax[0].grid()
        ax[0].bar(df.index, df["rain"]*4, label="Raining Day")
        ax[0].plot(df.index, df["tmed"], label="Media", color='green')
        ax[0].plot(df.index, df["tmin"], label="Mínima", color='blue')
        ax[0].plot(df.index, df["tmax"], label="Máxima", color='red')
        #ax[0].set_xlabel("Fecha")
        ax[0].set_ylabel("ºC")
        ax[0].set_title("Temperaturas diarias")
        ax[0].legend(ncols=2)

        # - Wind -
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

        # - Pressure -
        ax[3].grid()
        ax[3].plot(df.index, df["presMax"], label="Presión Máxima", color='orange')
        ax[3].plot(df.index, df["presMin"], label="Presión Mínima", color='purple')
        # ax[3].bar(_df.index, _df["rain"]*900, label="Rain Day")
        #ax[3].set_xlabel("Fecha")
        ax[3].set_ylabel("h/Pa")
        ax[3].set_title("Presiones")
        ax[3].legend()

        st.pyplot(fig)  # Shows subplots
    else:
        st.write(df)  # If not cotains data



def main():  # Graphics page
    st.set_page_config(layout="wide")  # Set wide option
    # - sidebar -
    with st.sidebar:
        st.markdown('''# TFM - Torres Espinosa\n ### Data Location:''')
        
        # Selectbox to choose year, station and province.
        options = ('Almeria','Cadiz','Cordoba','Granada',
                        'Huelva','Jaen','Malaga','Sevilla')
        years = (2023,2022,2021,2020,2019)
        province = st.selectbox("Province: ", options, 6)  # choose province
        stations = app.opd.get_stations().query(  # obtains province stations
            f'provincia == "{(province.upper())}"')
        idema_name = st.selectbox( "Station: ", stations[['nombre']],
                                  len(stations)//2)
        idema = stations.query(f'nombre == "{idema_name}"')\
                               ['indicativo'].values[0]  # gets idema
        year = st.selectbox('Year:', years)  # choose year
        app.opd.set_idema(idema)
        app.data = app.opd.get_year(year)  # gets formatted data

    # Tab title
    st.title('Graphics')
    
    view_graphs(app.data)  # Show graphics of data


if __name__ == '__main__':  # Run the application
    app = App()
    main()

    