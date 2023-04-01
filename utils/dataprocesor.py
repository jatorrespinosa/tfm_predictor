# Dependencies
from datetime import datetime
import tqdm
from numpy import NaN
import pandas as pd

from sklearn.impute import KNNImputer

from opendata import OpenData

class Preprocessor():

    def __init__(self):
        self.data = OpenData().read_csv('data-01-22-rain.csv')

    def impute_data(self):
        new_data = self.data.dropna(thresh=6)
        col_names = new_data.columns

        imputer = KNNImputer()
        imputer.fit(new_data)
        notnull = pd.DataFrame(imputer.transform(new_data),
                               columns=col_names, index=new_data.index)
        
        return notnull
        