# ----------------------------------------------------------------------------
#   Machine Learning File, process the data for the app.
#   Torres Espinosa, Jose Antonio.
# ----------------------------------------------------------------------------

# Dependencies
import sys, random
from datetime import datetime
from numpy import sqrt
import pandas as pd
import streamlit as st

# For preprocessor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif

from imblearn.under_sampling import RandomUnderSampler

# For processor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, \
                             RandomForestRegressor, AdaBoostRegressor
# from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, \
                            roc_auc_score, mean_squared_error

# Own Modules
sys.path.append('../')
from log_handler import root_logger
from utils.opendata import OpenData

class Preprocessor():
    """Prepares data for models.
    """
    def __init__(self):
        """Preprocessor constructor.
        """
        root_logger.info('Preprocessor()')  # LOG - INFO
        # Dataset
        self.dataset = OpenData().read_csv('data/data-01-22.csv')  # ../
            

    def Xy(self, data='', verbose=False):
        """Split dataset in X and y.

        :param data: Dataset, if empty it'll uses its own data, defaults to ''
        :type data: str, optional
        :param verbose: If True return X and y, defaults to False
        :type verbose: bool, optional
        :return: if verbose is true return X and y.
        :rtype: dict tuple, Dataframes in dicts.
        """
        root_logger.info(f'Xy({data}, verbose={verbose})')  # LOG - INFO
        data = self.dataset if isinstance(data, str) else data
        data.dropna(thresh=6, inplace=True)  # three or fewer nulls

        # If not exists, Split X and Split y (prec & rain)
        if not (hasattr(self, 'X') and hasattr(self, 'y')):
            self.X, self.y = {}, {}
            for col in ['prec', 'rain']:  # Types loop
                if col in data.columns:
                    self.X.update({col: data.drop(columns=['prec','rain'],
                                                    errors='ignore')})
                    self.y.update({col: data.loc[:,col]})

            if verbose: return self.X, self.y


    def impute_data(self, data):
        """Imputes values for rows with three or fewer nulls.

        :param data: X
        :type data: DataFrame
        :return: imputed data
        :rtype: DataFrame
        """
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
        """Scales values with robust or minmax scaler.

        :param data: X
        :type data: DataFrame
        :param robust: If true select RobustScaler else MinMaxScaler, defaults to False
        :type robust: bool, optional
        :return: Scaled data
        :rtype: DataFrame
        """
        root_logger.debug(f'scale_data({data}, robust={robust})')  # LOG - DEBUG
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


    def select_attr(self, X, attr, **kwagrs):
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
            pca = PCA(**kwagrs)  # n_components=
            X_r = pca.fit(X).transform(X)
            ratio = pca.explained_variance_ratio_
        else:
            # y_train
            # score_func: f_classif, chi2, mutual_info_classif,
            # k='all'
            # score_func transforms str to value 
            mapper = {'f_classif': f_classif, 'chi2': chi2,
                      'mutual_info_classif':mutual_info_classif}
            kwagrs['score_func'] = mapper[kwagrs['score_func']]
            # Catch y
            y = kwagrs['y']
            kwagrs.pop('y')
            # - KBest -
            filt = SelectKBest(**kwagrs) 
            X_r = filt.fit_transform(X, y)
            ratio = filt.scores_
        
        return X_r, ratio

    
    def undersampling(self, X, y):
        """For classifications models, undersamplig must be done, Rain class values.

        :param X: X
        :type X: DataFrame
        :param y: rain class series
        :type y: Series
        :return: Returns undersampled data X and y
        :rtype: DataFrame tuple
        """
        root_logger.debug(f'undersampling({X}, {y})')  # LOG - DEBUG
        # For rain data - Undersampling
        rus = RandomUnderSampler(random_state=random.seed(datetime.now()))
        X_und, y_und = rus.fit_resample(X, y)

        return X_und, y_und
    

    def split_data(self, X, y, under=False, verbose=False):
        """Splits data in train and test.

        :param X: X
        :type X: DataFrame
        :param y: y
        :type y: Series
        :param under: Enable undersampling, defaults to False
        :type under: bool, optional
        :param verbose: If true returns splitted data, defaults to False
        :type verbose: bool, optional
        :return: If verbose, splitted data. Dict for each class with every data.
        :rtype: dict of dict.
        """
        root_logger.debug(f'split_data({X}, {y}, under={under}, verbose={verbose})')  # LOG - DEBUG

        if not hasattr(self, 'splited'): self.splited = {}
        if under: X, y = self.undersampling(X, y)  # Undersampling once time only
            
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, random_state=random.seed(datetime.now()))
        # Saves splitted data
        self.splited.update({y.name:{'X_train': X_train,
                                     'X_test': X_test,
                                     'y_train': y_train,
                                     'y_test': y_test}})

        if verbose: return self.splited


    def preprocess_data(self, verbose=False, drop=True):
        """Preprocess a dataset, impute, scale and split in train and test.

        :param verbose: If true, return scaled data, defaults to False
        :type verbose: bool, optional
        :return: scaled prec and rain data
        :rtype: DataFrame tuple
        """
        root_logger.info(f'preprocess_data(verbose={verbose})')  # LOG - INFO
        # X, y
        self.Xy()
        # Imputer
        prec_impute = self.impute_data(self.X['prec'])
        rain_impute = self.impute_data(self.X['rain'])
        # Scaler
        prec_sc = self.scale_data(prec_impute)
        rain_sc = self.scale_data(rain_impute)
        if drop:  # Unable 'dir' and 'presMax' attr
            prec_sc = prec_sc.drop(columns=['dir','presMax'], errors='ignore')
            rain_sc = rain_sc.drop(columns=['dir','presMax'], errors='ignore')
        
        # Splitter
        self.split_data(prec_sc, self.y['prec'])
        self.split_data(rain_sc, self.y['rain'], under=True)

        if verbose: return prec_sc, rain_sc


class Processor(Preprocessor):
    """Contains the machine learning models.

    :param Preprocessor: prepares data.
    :type Preprocessor: class
    """
    def __init__(self):
        """Processor constructor.
        """
        root_logger.info('Processor()')  # LOG - INFO
        super().__init__()
        self.prec_sc, self.rain_sc = self.preprocess_data(verbose=True)
    

    # ------------------------------------
        # Classification Models
    # ------------------------------------
    def knn_clf_model(self):
        """KNeighborsClassifier with n_neighbors=3 and euclidean metric.

        :return: confusion_matrix and roc_auc_score results from predict_clf(model)
        :rtype: dict
        """
        root_logger.info('knn_clf_model()')  # LOG - INFO
        if not hasattr(self, 'knn_clf'):  # If not exists knn_clf: instance and fit it
            self.knn_clf = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
            self.knn_clf.fit(self.splited['rain']['X_train'],
                             self.splited['rain']['y_train'])
        
        return self.predict_clf(self.knn_clf)
    

    def randomforest_clf(self):
        """RandomForestClassifier with max_depth=10, ccp_alpha=.0001 and log_loss criterion.

        :return: confusion_matrix and roc_auc_score results from predict_clf(model)
        :rtype: dict
        """
        root_logger.info('randomforest_clf()')  # LOG - INFO
        if not hasattr(self, 'rf_clf'):  # If not exists rf_clf: instance and fit it
            self.rf_clf = RandomForestClassifier(max_depth=10, ccp_alpha=.0001,
                                                 criterion='log_loss')
            self.rf_clf.fit(self.splited['rain']['X_train'],
                            self.splited['rain']['y_train'])
        
        return self.predict_clf(self.rf_clf)


    def decisiontree_clf(self):
        """DecisionTreeClassifier with max_depth=17, ccp_alpha=.001 and log_loss criterion.

        :return: confusion_matrix and roc_auc_score results from predict_clf(model)
        :rtype: dict
        """
        root_logger.info('decisiontree_clf()')  # LOG - INFO
        if not hasattr(self, 'dt_clf'):  # If not exists dt_clf: instance and fit it
            self.dt_clf = DecisionTreeClassifier(max_depth=17, ccp_alpha=.001,
                                                 criterion='log_loss')
            self.dt_clf.fit(self.splited['rain']['X_train'],
                            self.splited['rain']['y_train'])
        
        return self.predict_clf(self.dt_clf)


    def bagging_clf(self):
        """BaggingClassifier with n_estimators=125, max_samples=.7 and base_estimator from decisiontree_clf().

        :return: confusion_matrix and roc_auc_score results from predict_clf(model)
        :rtype: dict
        """
        root_logger.info('bagging_clf()')  # LOG - INFO
        if not hasattr(self, 'bg_clf'):  # If not exists bg_clf: instance and fit it
            if not hasattr(self, 'dt_clf'): self.decisiontree_clf()  # If not exists dt_clf: creates it
            self.bg_clf = BaggingClassifier(base_estimator=self.dt_clf,
                                            n_estimators=125, max_samples=.7)
            self.bg_clf.fit(self.splited['rain']['X_train'],
                            self.splited['rain']['y_train'])
        
        return self.predict_clf(self.bg_clf)


    # ------------------------------------
        # Regression Models
    # ------------------------------------
    def knn_reg_model(self):
        """KNeighborsRegressor with n_neighbors=3 and euclidean metric.

        :return: rmse results from predict_reg(model)
        :rtype: dict
        """
        root_logger.info('knn_reg_model()')  # LOG - INFO
        if not hasattr(self, 'knn_reg'):  # If not exists knn_reg: instance and fit it
            self.knn_reg = KNeighborsRegressor(n_neighbors=3, metric='euclidean')
            self.knn_reg.fit(self.splited['rain']['X_train'],
                            self.splited['rain']['y_train'])
        
        return self.predict_reg(self.knn_reg)


    def randomforest_reg(self):
        """RandomForestRegressor with max_depth=10, ccp_alpha=.0001 and squared_error criterion.

        :return: rmse results from predict_reg(model)
        :rtype: dict
        """
        root_logger.info('randomforest_reg()')  # LOG - INFO
        if not hasattr(self, 'rf_reg'):  # If not exists rf_reg: instance and fit it
            self.rf_reg = RandomForestRegressor(max_depth=10, ccp_alpha=.0001,
                                                criterion='squared_error')
            self.rf_reg.fit(self.splited['rain']['X_train'],
                            self.splited['rain']['y_train'])

        return self.predict_reg(self.rf_reg)


    def decisiontree_reg(self):
        """DecisionTreeRegressor with max_depth=27, ccp_alpha=.001 and squared_error criterion.

        :return: rmse results from predict_reg(model)
        :rtype: dict
        """
        root_logger.info('decisiontree_reg()')  # LOG - INFO
        if not hasattr(self, 'dt_reg'):  # If not exists dt_reg: instance and fit it
            self.dt_reg = DecisionTreeRegressor(max_depth=27, ccp_alpha=.001,
                                                criterion='squared_error')
            self.dt_reg.fit(self.splited['rain']['X_train'],
                            self.splited['rain']['y_train'])

        return self.predict_reg(self.dt_reg)


    def boosting_reg(self):
        """AdaBoostRegressor with n_estimators=85 and base_estimator from decisiontree_reg().

        :return: rmse results from predict_reg(model)
        :rtype: dict
        """
        root_logger.info('boosting_reg()')  # LOG - INFO
        if not hasattr(self, 'bs_reg'):  # If not exists bs_reg: instance and fit it
            if not hasattr(self, 'dt_reg'): self.decisiontree_reg()  # If not exists dt_reg: creates it
            self.bs_reg = AdaBoostRegressor(base_estimator=self.dt_reg,
                                            n_estimators=85)
            self.bs_reg.fit(self.splited['rain']['X_train'],
                            self.splited['rain']['y_train'])

        return self.predict_reg(self.bs_reg)
    

    # ------------------------------------
        # Predictions
    # ------------------------------------
    def predict_clf(self, model):
        """Makes predictions for classification models and returns confusion matrix and roc_score.
        Predicts with test, train and all data.

        :param model: models from knn_clf_model(), randomforest_clf(), decisiontree_clf() or bagging_clf().
        :type model: Instance of Classification ML model.
        :return: confusion_matrix and roc_auc_score for test, train and all data prediction.
        :rtype: dict
        """
        root_logger.info(f'predict_clf({model})')  # LOG - INFO
        # Predictions
        y_pred = model.predict(self.splited['rain']['X_test'])
        y_pred1 = model.predict(self.splited['rain']['X_train'])
        y_pred2 = model.predict(self.rain_sc)
        # Confusion Matrix and roc_auc_score
        results = {
            'conf_matrix':{
                'test':confusion_matrix(self.splited['rain']['y_test'], y_pred),
                'train':confusion_matrix(self.splited['rain']['y_train'], y_pred1),
                'all':confusion_matrix(self.y['rain'], y_pred2)},
            'roc':{
                'test':roc_auc_score(self.splited['rain']['y_test'], y_pred),
                'train':roc_auc_score(self.splited['rain']['y_train'], y_pred1),
                'all':roc_auc_score(self.y['rain'], y_pred2)
            }}
        
        return results


    def predict_reg(self, model):
        """Makes predictions for regression models and returns rmse.
        Predicts with test, train and all data.

        :param model: models from knn_reg_model(), randomforest_reg(), decisiontree_reg() or bagging_reg().
        :type model: Instance of Regression ML model.
        :return: rmse for test, train and all data prediction.
        :rtype: dict
        """
        root_logger.info(f'predict_reg({model})')  # LOG - INFO
        # Predictions
        y_pred = model.predict(self.splited['rain']['X_test'])
        y_pred1 = model.predict(self.splited['rain']['X_train'])
        y_pred2 = model.predict(self.rain_sc)
        # RMSE
        results = {'rmse':{
            'test':sqrt(mean_squared_error(self.splited['rain']['y_test'], y_pred)),
            'train':sqrt(mean_squared_error(self.splited['rain']['y_train'], y_pred1)),
            'all':sqrt(mean_squared_error(self.y['rain'], y_pred2))
        }}
        
        return results
    

    # ------------------------------------
        # Init Models
    # ------------------------------------
    @st.cache_resource  # Using the cache for model operations
    def init_classification(_self):  # _self due to st.cache_resource
        """Initialize classification models and return evaluation results.

        :return: Results of different classification models.
        :rtype: dict
        """
        root_logger.info('init_classification()')  # LOG - INFO
        kn_results = _self.knn_clf_model()
        rf_results = _self.randomforest_clf()
        dt_results = _self.decisiontree_clf()
        bg_results = _self.bagging_clf()

        return {'knn': kn_results, 'rf':rf_results,
                'dt':dt_results, 'bg':bg_results}
    

    @st.cache_resource  # Using the cache for model operations
    def init_regresion(_self):  # _self due to st.cache_resource
        """Initialize regression models and return evaluation results.

        :return: Results of different regression models.
        :rtype: dict
        """
        root_logger.info('init_regresion()')  # LOG - INFO
        kn_results = _self.knn_reg_model()
        rf_results = _self.randomforest_reg()
        dt_results = _self.decisiontree_reg()
        bs_results = _self.boosting_reg()

        return {'knn': kn_results, 'rf':rf_results,
                'dt':dt_results, 'bs':bs_results}