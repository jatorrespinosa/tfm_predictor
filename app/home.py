# ----------------------------------------------------------------------------
#   File to init the tfm streamlit App, deploy with 'streamlit run app/home.py'.
#   Torres Espinosa, Jose Antonio.
# ----------------------------------------------------------------------------

# Dependencies
import os
import sys
import streamlit as st
# Own Modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('.')
sys.path.append('..')
sys.path.append('tfm_predictor')
from utils.opendata import OpenData

class App():
    def __init__(self):
        """App Constructor.
        """
        self.opd = OpenData()  # Data provider instance
        self.data = ''

    # --------------------------
    # -------- MAIN ------------
    # --------------------------
    def main(self):  # Home page
        # Tab title
        st.set_page_config(page_title="TFM", layout="wide")  # , page_icon="blue-flower.ico"

        # --- SideBar ---
        with st.sidebar:
            st.markdown('''# TFM - Torres Espinosa''')

        # --- Principal section ---
        # Title
        st.title("TFM - Wheather Predictor")
        # Description
        st.write("""Its a Machine learning App to predict wheather in Andalusia. 
            Uses Aemet data through its Open Data Api. 
            Requests the stations data and processes it with internal code.
            This app shows data graphs, analysis graps and accuracy results of 
            Machine Learning models.
            This app runs with streamlit and the machine learning section uses 
            scikit-learn.
            Developed by Torres Espinosa, Jose Antonio.""")

        st.title('Data Location:')
        c1, c2, c3 = st.columns(3)  # Creates 3 columns on page
        # Selectbox to choose year, station and province.
        with c1:
            options = ('Almeria','Cadiz','Cordoba','Granada',  # for province
                        'Huelva','Jaen','Malaga','Sevilla')
            province = st.selectbox('Province:', options, 6)
            stations = self.opd.get_stations().query(  # gets province stations
                f'provincia == "{(province.upper())}"')
        with c2:
            idema_name = st.selectbox('Station:', stations[['nombre']],  # choose station
                                      len(stations)//2)
            idema = stations.query(f'nombre == "{idema_name}"')\
                                  ['indicativo'].values[0]  # gets idema
        with c3:
            years = (2023,2022,2021,2020,2019)
            year = st.selectbox('Year:', years)  # choose year
            self.opd.set_idema(idema)
            self.data = self.opd.get_year(year, source=True)  # Obtains source data
        # Shows source data
        st.subheader('Source Data:')
        st.write(self.data)
        # Shows formatted data
        st.subheader('Formatted Data:')
        st.write(self.opd.get_year(year))  # Obtains formatted data


if __name__ == '__main__':  # Run the application
    main = App()
    main.main()