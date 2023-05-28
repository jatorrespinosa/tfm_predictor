import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('.')
sys.path.append('..')
sys.path.append('tfm_predictor')

from utils.dataprocesor import Preprocessor
from app.home import App
from log_handler import root_logger

# opd = OpenData()
def view_density(X, y):
    root_logger.info('view_density()')  # LOG - INFO
    if not isinstance(X, str):
        data = pd.concat([X,y], axis=1)
        fig, ax = plt.subplots(ncols=3,nrows=3, figsize=(20,20))
        
        # ax[0].grid()
        data.plot(kind='density', subplots=True, layout=(3,3),sharex=False, 
                  ax=[ax[0,0],ax[0,1],ax[0,2],ax[1,0],ax[1,1],ax[1,2],
                      ax[2,0],ax[2,1],ax[2,2]])
        
        st.write(fig)
    else:
        st.write(X)

def view_box(X, y):
    root_logger.info('view_box()')  # LOG - INFO
    if not isinstance(X, str):
        data = pd.concat([X,y], axis=1)
        fig, ax = plt.subplots(ncols=3,nrows=3, figsize=(23,23))
        
        # ax[0].grid()
        data.plot(kind='box', subplots=True, layout=(3,3),sharex=False,
                  ax=[ax[0,0],ax[0,1],ax[0,2],ax[1,0],ax[1,1],ax[1,2],
                      ax[2,0],ax[2,1],ax[2,2]])
        st.pyplot(fig)
    else:
        st.write(X)


def view_pairs(X, y):
    root_logger.info('view_pairs()')  # LOG - INFO
    if not isinstance(X, str):
        fig = sns.pairplot(pd.concat([X,y], axis=1), hue=y.name)

        st.pyplot(fig)
    else:
        st.write(X)


def view_correlations(X, y):
    if not isinstance(X, str):
        root_logger.info('view_correlations()')  # LOG - INFO
        data = pd.concat([X,y], axis=1)
        correlations = data.corr()
        fig = px.imshow(correlations,color_continuous_scale='viridis',
                        x=data.columns,y=data.columns)
        fig.update_xaxes(side="top")

        st.plotly_chart(fig, theme='streamlit', use_container_width=True)
    else:
        st.write(X)
    


def view_pca(X, p, class_type, columns):
    root_logger.info('view_pca()')  # LOG - INFO
    if not isinstance(X, str):
        
        
        pca3, ratio = p.select_attr(X.drop(columns=columns), 'PCA', n_components=3)

        # ph = .5
        # fig = px.scatter(pd.concat([pd.DataFrame(pca2, columns=[0,1]),
        #                             pd.DataFrame(p.y[class_type])],axis=1),
        #                             x=0, y=1, color=class_type, opacity=ph,
        #                             color_discrete_scale='viridis')
        
        # st.plotly_chart(fig)
        fig = px.scatter_3d(pca3, x=0,y=1,z=2, 
                    color=p.y[class_type], color_continuous_scale='viridis',
                    title=f'total explained variance:{ratio.sum()*100:.2f}',
                    labels={'0':'PCA1','1':'PCA2','2':'PCA3'})
        st.plotly_chart(fig)

    else:
        st.write(X)


def view_kbest(X, p, class_type):
    root_logger.info('view_kbest()')  # LOG - INFO
    if not isinstance(X, str):
        if class_type == 'prec':
            rain_kb_clf, ratio = p.select_attr(X, 'Kbest', y=p.y['prec'], k='all',
                               score_func='f_classif')
            fig = plt.figure(figsize=(10, 4))
            targets = X.columns
            # - Grafica 1 -
            sns.lineplot(x=targets, y=ratio, ax=plt.gca())
            plt.gca().grid()
            plt.title("f_classif")
        else:
            rain_kb_clf, ratio1 = p.select_attr(X, 'Kbest', y=p.y['rain'], k='all',
                               score_func='f_classif')
            rain_kb_chi2, ratio2 = p.select_attr(X, 'Kbest', y=p.y['rain'], k='all',
                                        score_func='chi2')
            rain_kb, ratio3 = p.select_attr(X, 'Kbest', y=p.y['rain'], k='all',
                                        score_func='mutual_info_classif')
            plt.rcParams['figure.figsize'] = [15, 15] # Dimensiona el conjunto de gráficas
            fig, ax = plt.subplots(nrows=3)
            targets = X.columns
            # - Grafica 1 -
            ax[0].plot(targets, ratio1)
            ax[0].grid()
            ax[0].set_title(f"f_classif")
            # - Grafica 2 -
            ax[1].plot(targets, ratio2)
            ax[1].grid()
            ax[1].set_title("chi2")
            # - Grafica 3 -
            ax[2].plot(targets, ratio3)
            ax[2].grid()
            ax[2].set_title("mutual_info_classif")
        st.pyplot(fig)
    else:
        st.write(X)


# ---------------------------
#  MAIN
# ---------------------------

def main():
    root_logger.info('Streamlit - Analysis')  # LOG - INFO
    plt.rcParams.update({'font.size': 20, 'figure.figsize':(23,23)})

    # -- INIT --
    # Data Preprocessing
    p = Preprocessor()
    prec_sc, rain_sc = p.preprocess_data(verbose=True)

    with st.sidebar:  # Side menu
        opciones = ('prec','rain')  # Class options
        graphics = ('density','box','pairs','correlations','PCA','Kbest')  # Graphics types

        st.markdown('''# TFM - Torres Espinosa\n ### Data Location:''')
        
        # Selectbox can choose class and graphic to view
        class_type = st.selectbox("Class type: ", opciones)
        graphic = st.selectbox("Class type: ", graphics)
        

    # Title
    st.title('Analysis')
    # Data
    sc = prec_sc if class_type == 'prec' else rain_sc
    args = (sc, p.y[class_type])
    # Function selector by graphic type
    if graphic == 'density':
        view_density(*args)
    elif graphic == 'box':
        view_box(*args)
    elif graphic == 'pairs':
        view_pairs(*args)
    elif graphic == 'correlations':
        view_correlations(*args)
    elif graphic == 'PCA':
        cols = st.multiselect('Drop attributes:',sc.columns)
        
        view_pca(sc, p, class_type,cols)
    elif graphic == 'Kbest':
        view_kbest(sc, p, class_type)

if __name__ == '__main__':
    app = App()
    main()

    