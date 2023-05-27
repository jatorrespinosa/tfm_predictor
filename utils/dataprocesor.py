# Dependencies
import sys, random, time
import tqdm
from numpy import NaN
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif

from imblearn.under_sampling import RandomUnderSampler

# Own Modules
sys.path.append('../')
from log_handler import root_logger

from utils.opendata import OpenData

class Preprocessor():

    def __init__(self):
        # getters ?!
        # Inplace ?!
        self.dataset = OpenData().read_csv('../data/data-01-22.csv')  # ../
        
        # self.data_prec = OpenData().read_csv('./data/data-01-22-prec.csv')
        # self.data_rain = OpenData().read_csv('./data/data-01-22-rain.csv')
            

    def Xy(self, data='', verbose=False):
        root_logger.info(f'Xy({data}, verbose={verbose})')  # LOG - INFO
        data = self.dataset if isinstance(data, str) else data
        data.dropna(thresh=6, inplace=True)

        # If not exists, Split X and Split y (prec & rain)
        if not (hasattr(self, 'X') and hasattr(self, 'y')):
            # For prec data
            self.X, self.y = {}, {}
            for col in ['prec', 'rain']:
                if col in data.columns:
                    self.X.update({col: data.drop(columns=['prec','rain'],
                                                    errors='ignore')})
                    self.y.update({col: data.loc[:,col]})

            if verbose: return self.X, self.y

    def impute_data(self, data):
        root_logger.debug(f'impute_data({data})')  # LOG - DEBUG
        # Drop rows with more than three missing data
        new_data = data  # self.dataset if isinstance(dataset, str) else data

        # If not exists imputer: instance and fit it
        if not hasattr(self, 'imputer'):
            imputer = KNNImputer()
            imputer.fit(new_data)
            self.imputer = imputer
        # Impute data
        clean_data = pd.DataFrame(self.imputer.transform(new_data),
                               columns=new_data.columns, index=new_data.index)

        return clean_data  # Clean data


    def scale_data(self, data, robust=False):
        # If not exists scaler: instance and fit it
        if not hasattr(self, 'scaler'):
            if robust:  # Select between Robust or MinMax Scaler
                scaler = RobustScaler()
            else:
                scaler = MinMaxScaler()
            scaler.fit(data)  # Fit
            self.scaler = scaler  # Save scaler

        # Save scaled data as X
        data_scaled = self.scaler.transform(data)
        sc_X = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)
        
        return sc_X  # return dataframe

    def select_attr(self, X, attr, **kagrs):
        """attr selects between 'PCA' or 'Kbest'. X will be reduced.
        Will need **kwags:
        - For PCA: n_components.
        - For Kbest: k='all' and score_func (f_classif, chi2, mutual_info_classif)


        :param X: _description_
        :type X: _type_
        :param attr: _description_
        :type attr: _type_
        :return: X_r and ratio of the attributes
        :rtype: tuple
        """
        if attr == 'PCA':
            pca = PCA(**kagrs)  # n_components=
            X_r = pca.fit(X).transform(X)
            ratio = pca.explained_variance_ratio_
        else:
            # y_train
            # score_func: f_classif, chi2, mutual_info_classif,
            # k='all'
            mapper = {'f_classif': f_classif, 'chi2': chi2,
                      'mutual_info_classif':mutual_info_classif}
            y = kagrs['y']
            kagrs.pop('y')
            kagrs['score_func'] = mapper[kagrs['score_func']]
            filt = SelectKBest(**kagrs) 
            X_r = filt.fit_transform(X, y)
            ratio = filt.scores_
        
        return X_r, ratio

    
    def undersampling(self, X, y):
        # For rain data - Undersampling
        rus = RandomUnderSampler()  # random_state=random.seed(time.time())
        X_und, y_und = rus.fit_resample(X, y)

        return X_und, y_und
    

    def split_data(self, X, y, under=False, verbose=False):
        if under:
            X, y = self.undersampling(X, y)

        if not hasattr(self, 'splited'):
            self.splited = {}
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y)
        
        self.splited[y.name].update({'X_train': X_train,
                                   'X_test': X_test,
                                   'y_train': y_train,
                                   'y_test': y_test})

        if verbose: return self.splited


    def preprocess_data(self):
        # X, y
        self.Xy()
        # Imputer
        prec_impute = self.impute_data(self.X['prec'])
        rain_impute = self.impute_data(self.X['rain'])
        # Scaler
        prec_sc = self.scale_data(prec_impute)
        rain_sc = self.scale_data(rain_impute)

        # Select attr Kbest?
            # select_attr(self, X, attr, **kagrs)

        # Splitter
        # self.split_data(prec_sc, self.y['prec'])
        # self.split_data(rain_sc, self.y['rain'])

        return prec_sc, rain_sc