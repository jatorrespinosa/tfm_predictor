import os
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import PredictionErrorDisplay

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('.')
sys.path.append('..')
sys.path.append('tfm_predictor')

from utils.dataprocesor import Processor
from app.home import App
from log_handler import root_logger


def plot_conf_matrix(values):
    """Creates a confusion matrix plot with the given values.

    :param values: Array result of sklearn.metrics.confusion_matrix
    :type values: Array
    """
    root_logger.info(values)  # LOG - INFO
    ax = ['No rain', 'Rain']
    # Creates figure
    fig = px.imshow(values, text_auto=True, color_continuous_scale='Viridis',
                    x=ax,y=ax, labels={'x':'Predicted','y':'True'})
    fig.update_xaxes(side="top")  # Axes position 
    st.plotly_chart(fig)  # Plot on page for plotly figure

    
def plot_prediction_error(y_true, y_pred):
    """Creates a prediction matrix plot with the given and selected values.

    :param y_true: Selected y values from processor
    :type y_true: pd.Series
    :param y_pred: Predicted array from json response of dataprocesor
    :type y_pred: Array
    """
    root_logger.info(f'plot_prediction_error({len(y_true)}, {len(y_pred)})')
    
    fig = plt.figure()  # Creates figure
    # Plot data on figure
    display = PredictionErrorDisplay.from_predictions(y_true=y_true,
                                                      y_pred=y_pred,
                                                      ax=plt.gca())  # Get Current Axes
    st.pyplot(fig)  # Plot on page for matplotlib figure


def main():
    root_logger.info('Streamlit - ML')  # LOG - INFO

    # Data Preprocessing
    p = Processor()

    with st.sidebar:  # Side menu
        opciones = ('prec','rain')  # Class options

        st.markdown('''# TFM - Torres Espinosa\n ### Data Location:''')
        
        # Selectbox can choose class and graphic to view
        class_type = st.selectbox("Class type: ", opciones,1)
        raw = st.checkbox('View Raw Data')  # To view json response of processor
        st.markdown('#')
        st.markdown('##### Glossary:')
        st.markdown('''
        * knn: KNeighbors
        * rf: RandomForest
        * dt: DecisionTree
        * bg: BaggingClassifier
        * bs: AdaBoostRegressor''')
        
    # Title
    st.title(f"{'Classification' if class_type == 'rain' else 'Regression'}")
    c1, c2 = st.columns([1,2])  # Creates 2 columns on page, second bigger
    t1, t2 = st.columns([2,1])  # Creates 2 columns on page, first bigger

    # Selector by class type
    if class_type == 'rain':  # -- Classification --
        results = p.init_classification()  # Train and predict
        df = pd.DataFrame(results).T  # Results in DataFrame

        with c1:
            st.markdown('#### Area under curve ROC:')
            roc_df = df['roc'].apply(pd.Series)  # Convert to Series
            st.dataframe(roc_df.style.highlight_max(axis=0))  # highlights the maximums
        with c2:
            st.markdown('')  # Positioning spaces
            st.markdown('')
            st.markdown('###### Hyperparameters:')
            st.markdown('''
                * KNeighborsClassifier(n_neighbors=5, metric='euclidean')
                * RandomForestClassifier(max_depth=13, ccp_alpha=0, criterion='log_loss')
                * DecisionTreeClassifier(max_depth=16, ccp_alpha=.0001, criterion='log_loss')
                * BaggingClassifier(base_estimator=DecisionTreeClassifier, n_estimators=60, max_samples=.7)
            ''')
        
        with t2:
            st.markdown('')  # Positioning spaces
            st.markdown('')
            st.markdown('')
            st.markdown('')
            st.markdown('')
            # Selectbox can choose model and data type
            model = st.selectbox('Model:', ('knn','rf','dt','bg'))
            data_type = st.selectbox('Matrix of :', ('test','train','all'))
        with t1:
            st.markdown('#### Confusion matrix:')
            cm_df = df['conf_matrix'].apply(pd.Series)  # Convert to Series
            plot_conf_matrix(cm_df.T.loc[data_type,model])  # Plots confusion matrix

    elif class_type == 'prec':  # -- Regression --
        results = p.init_regresion()  # Train and predict
        
        with c1:
            st.markdown('##### RMSE:')
            roc_df = pd.DataFrame(results).T['rmse'].apply(pd.Series)  # Convert to Series
            st.dataframe(roc_df.style.highlight_min(axis=0))  # highlights the minimums
        with c2:
            st.markdown('')  # Positioning spaces
            st.markdown('')
            st.markdown('###### Hyperparameters:')
            st.markdown('''
                * KNeighborsRegressor(n_neighbors=10, metric='euclidean')
                * RandomForestRegressor(max_depth=30, ccp_alpha=.0001, criterion='squared_error')
                * DecisionTreeRegressor(max_depth=16, ccp_alpha=.0001, criterion='squared_error')
                * AdaBoostRegressor(base_estimator=DecisionTreeRegressor, n_estimators=155)
            ''')
        
        with t2:
            st.markdown('')  # Positioning spaces
            st.markdown('')
            st.markdown('')
            st.markdown('')
            st.markdown('')
            # Selectbox can choose model and data type
            model = st.selectbox('Model:', ('knn','rf','dt','bs'))
            data_type = st.selectbox('Predict data :', ('test','train','all'))
        with t1:
            st.markdown('#### Prediction Error:')
            # Selects y values
            if data_type == 'test': y = p.splited['prec']['y_test']
            elif data_type == 'train': y = p.splited['prec']['y_train']
            elif data_type == 'all': y = p.y['prec']
            plot_prediction_error(y, results[model]['predictions'][data_type])  # Plots prediction_error

    else:  # If not data
        results = 'Choose the correct class type.'
    
    if raw:  # To see json response
        st.markdown('##### Raw:')
        st.write(results)
    

if __name__ == '__main__':
    app = App()
    main()