import os
import sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('.')
sys.path.append('..')
sys.path.append('tfm_predictor')
from utils.opendata import OpenData
from app.home import App

# opd = OpenData()


def main():
    with st.sidebar:
        opciones = ('Almeria','Cadiz','Cordoba','Granada',
                        'Huelva','Jaen','Malaga','Sevilla')
        years = (2022,2021,2020,2019)

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

if __name__ == '__main__':
    app = App()
    main()

    