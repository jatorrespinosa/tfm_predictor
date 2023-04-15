# Dependecies
import sys
import time
from datetime import datetime
from tqdm import tqdm
import requests
import pandas as pd

import streamlit as st

# Own Modules
sys.path.append('../')
from utils.keygen import Keygen
from log_handler import root_logger

class OpenData():
    def __init__(self, idema='6156X'):
        root_logger.info(f'OpenData({idema})')  # LOG - INFO
        self.key = Keygen().get_key()
        self.url = 'https://opendata.aemet.es/opendata/api/'
        if idema: self.idema = idema

    def set_idema(self, idema):
        root_logger.info(f'set_idema({idema})')  # LOG - INFO
        self.idema = idema

    def __format_data(self, df):
        df = df[['fecha','tmed','tmin','tmax','dir','velmedia',
                 'racha','presMin','presMax','prec']]
        # fecha = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
        # df.loc[:,'fecha'] = fecha.copy()
        df = df.astype({'fecha':'datetime64[ns]'})
        df.replace({'Ip': '0.09','Acum':'999'}, inplace=True)
        
        df = df.set_index('fecha').astype(str)\
            .applymap(lambda x: x.replace(',','.')).astype(float)
        try:
            df['rain'] = df['prec'].apply(lambda x: 0 if x == 0 else 1)
        except Exception as e:
            root_logger.error(str(e))

        df.loc[df['dir'] == 88, 'dir'] = pd.NA
        df.loc[df['prec'].isna(),'rain'] = pd.NA

        return df



    def read_api(self, url):
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
    @st.cache_resource
    def get_stations(_self):
        url = _self.url + 'valores/climatologicos/inventarioestaciones/todasestaciones'

        response = _self.read_api(url)
        root_logger.debug(f'get_stations() - {response["estado"]}')  # LOG - DEBUG
        if response:
            stations = _self.read_api(response['datos'])
            return pd.DataFrame(stations)
        
    # - Daily data YEAR -
    def get_year(self, year, source=False, legend=False):
        init = f'{year}-01-01T00:00:00UTC'
        end = f'{year}-12-31T23:59:59UTC'
        url = self.url + 'valores/climatologicos/diarios/datos/fechaini/' +\
              f'{init}/fechafin/{end}/estacion/{self.idema}'

        response = self.read_api(url)
        root_logger.debug(f'get_year({year}) - {response["estado"]}')  # LOG - DEBUG
        if response['estado'] == 200:
            data = pd.DataFrame(self.read_api(response['datos']))
            if legend:
                meta = pd.DataFrame(self.read_api(response['metadatos'])['campos'])
                return data , meta
            return data if source else self.__format_data(data)
        else:
            return response['descripcion']
    
    def get_data_range(self, init_year, end_year):
        root_logger.info(f'get_data_range({init_year}, {end_year})')  # LOG - INFO

        col_names = ['fecha','indicativo','nombre','provincia','altitud','tmed',
                     'prec','tmin','horatmin','tmax','horatmax','dir','velmedia',
                     'racha','horaracha','presMax','horaPresMax','presMin',
                     'horaPresMin']
        data = pd.DataFrame(columns=col_names)

        for year in tqdm(range(init_year,end_year+1)):
            data = pd.concat([data, self.get_year(year, source=True)], ignore_index=True)
        
        return data

    #-------------------------------------
    #  READ FILES
    #-------------------------------------

    def read_csv(self, path):
        root_logger.info(f'read_csv({path})')  # LOG - INFO
        data = pd.read_csv(path)

        data['fecha'] = data['fecha'].apply(
            lambda x: datetime.strptime(x, '%Y-%m-%d').date())
        data = data.set_index('fecha')

        return data