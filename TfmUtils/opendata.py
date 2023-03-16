# Dependecies
import sys
from datetime import datetime
from tqdm import tqdm
import requests
import pandas as pd

# Own Modules
from keygen import Keygen

import logging
logging.basicConfig(filename='opendata.log', format='%(asctime)s %(message)s')

class OpenData():
    def __init__(self, idema=None):
        self.key = Keygen().get_key()
        self.url = 'https://opendata.aemet.es/opendata/api/'
        if idema: self.idema = idema

    def set_idema(self, idema):
        self.idema = idema

    def read_api(self, url):
        query = {'api_key': self.key}
        headers = {'cache-control': 'no-cache'}

        response = requests.request("GET", url, headers=headers, params=query)

        if response.status_code == 200:
            logging.info(f'read_api({url}) {response.status_code}')
            return response.json()
    
    def get_stations(self):
        url = self.url + 'valores/climatologicos/inventarioestaciones/todasestaciones'

        response = self.read_api(url)

        if response:
            stations = self.read_api(response['datos'])
            return pd.DataFrame(stations)

    def read_year(self, year, legend=False):
        idema = '6156X'  # self.idema
        init = f'{year}-01-01T00:00:00UTC'
        end = f'{year}-12-31T23:59:59UTC'
        url = self.url + 'valores/climatologicos/diarios/datos/fechaini/' +\
              f'{init}/fechafin/{end}/estacion/{idema}'

        response = self.read_api(url)

        if response:
            data = pd.DataFrame(self.read_api(response['datos']))
            if legend:
                meta = pd.DataFrame(self.read_api(response['metadatos'])['campos'])
                return data , meta
            return data
    
    def get_data_range(self, init_year, end_year):
        col_names = ['fecha','indicativo','nombre','provincia','altitud','tmed',
                     'prec','tmin','horatmin','tmax','horatmax','dir','velmedia',
                     'racha','horaracha','presMax','horaPresMax','presMin',
                     'horaPresMin']
        data = pd.DataFrame(columns=col_names)

        for year in tqdm(range(init_year,end_year+1)):
            data = pd.concat([data, self.read_year(year)]).reset_index(drop=True)
        
        return data
