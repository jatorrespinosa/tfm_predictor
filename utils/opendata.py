# ----------------------------------------------------------------------------
#   File to interact with Open Data Api, provides the data for the app.
#   Torres Espinosa, Jose Antonio.
# ----------------------------------------------------------------------------

# Dependecies
import sys
from datetime import datetime
from tqdm import tqdm
import requests
import pandas as pd

# import streamlit as st

# Own Modules
sys.path.append('../')
from utils.keygen import Keygen
from log_handler import root_logger

class OpenData():
    def __init__(self, idema=''):  # 6156X
        """Opendata Constructor

        :param idema: Station code, use get_stations(), defaults to ''
        :type idema: str, optional
        """
        root_logger.info(f'OpenData({idema})')  # LOG - INFO
        self.key = Keygen().get_key()  # api key
        self.url = 'https://opendata.aemet.es/opendata/api/'  # url base
        if idema: self.idema = idema


    def __format_data(self, df):
        """Transform data:
        - str to float 
        - set index (fecha)

        In addition, cleans dir errors and prec's string values.

        :param df: Data to transforms.
        :type df: DataFrame
        :return: Data transformed.
        :rtype: DataFrame
        """
        root_logger.debug(f'__format_data({df.head()})')  # LOG - DEBUG
        df = df[['fecha','tmed','tmin','tmax','dir','velmedia',
                 'racha','presMin','presMax','prec']].dropna(subset='prec')  # clean class
        
        # index & cast to float
        df = df.astype({'fecha':'datetime64[ns]'})
        df.replace({'Ip': '0.09','Acum':'999'}, inplace=True)  # str values
        df = df.set_index('fecha').astype(str)\
            .applymap(lambda x: x.replace(',','.')).astype(float)
        
        try: # prec to rain column
            df['rain'] = df['prec'].apply(lambda x: 0 if x == 0 else 1)
        except Exception as e:
            root_logger.error(str(e))

        # Null data
        df.loc[df['dir'] == 88, 'dir'] = pd.NA

        return df


    def set_idema(self, idema):
        """Changes idema value.

        :param idema: Station code, see Constructor.
        :type idema: str
        """
        root_logger.info(f'set_idema({idema})')  # LOG - INFO
        self.idema = idema


    def read_api(self, url):
        """Api request, if ok(200) returns it as json.

        :param url: Url for request.
        :type url: str
        :return: json response
        :rtype: dict
        """

        query = {'api_key': self.key}
        headers = {'cache-control': 'no-cache'}

        response = requests.request("GET", url, headers=headers, params=query)
        root_logger.debug(f"""get_api({url}) - {response.status_code}
                  {str(response.content)}""")  # LOG - DEBUG
        if response.status_code == 200:
            return response.json()
    
    #-------------------------------------
    #  GETTERS
    #-------------------------------------
    # - stations -
    # @st.cache_resource
    def get_stations(self):
        """Gets description of all AEMET stations.

        :return: Stations data: idema (indicativo), name, province, etc.
        :rtype: DataFrame
        """

        url = self.url + 'valores/climatologicos/inventarioestaciones/todasestaciones'

        response = self.read_api(url)
        root_logger.debug(f'get_stations() - {response["estado"]}')  # LOG - DEBUG
        if response:
            stations = self.read_api(response['datos'])
            return pd.DataFrame(stations)
        
    # - Daily data YEAR -
    def get_year(self, year, source=False, legend=False):
        """Gets daily data of a year from select station (set_idema function).

        :param year: year requested.
        :type year: int
        :param source: if True returns source data else formatted, defaults to False
        :type source: bool, optional
        :param legend: if True returns legend data, defaults to False
        :type legend: bool, optional
        :return: response data: the description if request isn't ok else the weather data.
        :rtype: DataFrame/str
        """
        init = f'{year}-01-01T00:00:00UTC'
        end = f'{year}-12-31T23:59:59UTC'
        url = self.url + 'valores/climatologicos/diarios/datos/fechaini/' +\
              f'{init}/fechafin/{end}/estacion/{self.idema}'

        response = self.read_api(url)  # Data request
        root_logger.debug(f'get_year({year}) - {response["estado"]}')  # LOG - DEBUG
        if response['estado'] == 200:
            data = pd.DataFrame(self.read_api(response['datos']))
            if legend:
                meta = pd.DataFrame(self.read_api(response['metadatos'])['campos'])
                return data , meta
            return data if source else self.__format_data(data)
        else:
            return response['descripcion']  # If 'estado' != 200. Failed request.
    
    def get_data_range(self, init_year, end_year):
        """Gets daily data of a year range from select station (set_idema function).

        :param init_year: Init value of year range.
        :type init_year: int
        :param end_year: End value of year range.
        :type end_year: int
        :return: the weather data.
        :rtype: DataFrame
        """
        root_logger.info(f'get_data_range({init_year}, {end_year})')  # LOG - INFO

        col_names = ['fecha','indicativo','nombre','provincia','altitud','tmed',
                     'prec','tmin','horatmin','tmax','horatmax','dir','velmedia',
                     'racha','horaracha','presMax','horaPresMax','presMin',
                     'horaPresMin']
        data = pd.DataFrame(columns=col_names)

        for year in tqdm(range(init_year,end_year+1)):  # Years loop
            data = pd.concat([data, self.get_year(year, source=True)],
                             ignore_index=True)  # Concat all years data
        
        return self.__format_data(data)

    #-------------------------------------
    #  READ FILES
    #-------------------------------------
    def read_csv(self, path):
        """Obtains data of a csv file.

        :param path: file path
        :type path: str
        :return: data if file exits
        :rtype: DataFrame
        """
        root_logger.info(f'read_csv({path})')  # LOG - INFO
        
        try:
            data = pd.read_csv(path)
            
            data['fecha'] = data['fecha'].apply(  # Transform date
                lambda x: datetime.strptime(x, '%Y-%m-%d').date())
            data = data.set_index('fecha')

            return data
        
        except FileNotFoundError:
            root_logger.error(f'Not Found <{path}>')  # LOG - ERROR
        except Exception as e:
            root_logger.error(str(e))  # LOG - ERROR