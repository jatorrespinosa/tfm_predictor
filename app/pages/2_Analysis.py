import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('.')
sys.path.append('..')
sys.path.append('tfm_predictor')
from utils.opendata import OpenData
from utils.dataprocesor import Preprocessor
from app.home import App

# opd = OpenData()
def view_density(X):
    if not isinstance(X, str):
        fig, ax = plt.subplots(ncols=3,nrows=3, figsize=(23,23))
        
        # ax[0].grid()
        X.plot(kind='density', subplots=True, layout=(3,3),sharex=False, ax=[ax[0,0],ax[0,1],ax[0,2],
                                                                             ax[1,0],ax[1,1],ax[1,2],
                                                                             ax[2,0],ax[2,1]])
        st.pyplot(fig)
    else:
        st.write(X)

def view_box(X):
    if not isinstance(X, str):
        fig, ax = plt.subplots(ncols=3,nrows=3, figsize=(23,23))
        
        # ax[0].grid()
        X.plot(kind='box', subplots=True, layout=(3,3),sharex=False, ax=[ax[0,0],ax[0,1],ax[0,2],
                                                                             ax[1,0],ax[1,1],ax[1,2],
                                                                             ax[2,0],ax[2,1]])
        st.pyplot(fig)
    else:
        st.write(X)


def view_pairs(X, y):
    if not isinstance(X, str):
        fig = sns.pairplot(pd.concat([X,y], axis=1), hue=y.name)

        st.pyplot(fig)
    else:
        st.write(X)


def view_correlations(X, y):
    if not isinstance(X, str):
        correlations = pd.concat([X,y], axis=1).corr()

        fig = plt.figure(figsize=(23,23))
        plt.matshow(correlations, fignum=fig.number, ax=plt.gca())
        plt.xticks(range(correlations.shape[1]), correlations.columns, rotation=45)
        plt.yticks(range(correlations.shape[1]), correlations.columns)
        fig.colorbar(plt.gca(), fraction=0.046, pad=0.04)

        st.write(fig)
    else:
        st.write(X)


def view_temp(df):
    if not isinstance(df, str):
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(nrows=4, figsize=(23,23))
        

        st.pyplot(fig)
    else:
        st.write(df)

# ---------------------------
#  MAIN
# ---------------------------

def main():
    plt.rcParams.update({'font.size': 20, 'figure.figsize':(23,23)})

    # -- INIT --
    # Data Preprocessing
    p = Preprocessor()
    prec_sc, rain_sc = p.preprocess_data()

    with st.sidebar:  # Side menu
        opciones = ('prec','rain')  # Class options
        graphics = ('density','box','pairs','correlations','PCA','Kbest')  # Graphics types

        st.markdown('''# TFM - Torres Espinosa\n ### Data Location:''')
        
        # Selectbox can choose class and graphic to view
        class_type = st.selectbox("Class type: ", opciones)
        graphic = st.selectbox("Class type: ", graphics)
        

    # Title
    st.title('Analysis')

    # Function selector by graphic type
    if graphic == 'density':
        view_density(prec_sc if class_type == 'prec' else rain_sc)
    elif graphic == 'box':
        view_box(prec_sc if class_type == 'prec' else rain_sc)
    elif graphic == 'pairs':
        view_pairs(prec_sc if class_type == 'prec' else rain_sc, p.y[class_type])
    elif graphic == 'correlarions':
        view_correlations(prec_sc if class_type == 'prec' else rain_sc, p.y[class_type])
    

if __name__ == '__main__':
    app = App()
    main()

    