# ----------------------------------------------------------------------------
#   File to see the analysis graphics.
#   Torres Espinosa, Jose Antonio.
# ----------------------------------------------------------------------------

# Dependencies
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
# import plotly.figure_factory as ff
import plotly.express as px

# Own Modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('.')
sys.path.append('..')
sys.path.append('tfm_predictor')
from utils.dataprocesor import Preprocessor
from app.home import App
from log_handler import root_logger


def view_density(X, y):
    """Creates density graphics.

    :param X: X values from splitted data, see Xy() in Preprocessor.
    :type X: DataFrame
    :param y: y values from splitted data, see Xy() in Preprocessor.
    :type y: Series
    """
    root_logger.info('view_density()')  # LOG - INFO
    if not isinstance(X, str):  # If contains data
        data = pd.concat([X,y], axis=1)
        fig, ax = plt.subplots(ncols=3,nrows=3, figsize=(20,20))  # Create fig and axes
        ax_flat = [item for row in ax for item in row]  # Flatten list of axes
        data.plot(kind='density', subplots=True, layout=(3,3), sharex=False,
                  ax=ax_flat[:data.shape[1]])  # Plot densities
        st.write(fig)
    else:  # If not contains data
        st.write(X)


def view_box(X, y):
    """Creates box graphics.

    :param X: X values from splitted data, see Xy() in Preprocessor.
    :type X: DataFrame
    :param y: y values from splitted data, see Xy() in Preprocessor.
    :type y: Series
    """
    root_logger.info('view_box()')  # LOG - INFO
    if not isinstance(X, str):  # If contains data
        data = pd.concat([X,y], axis=1)
        fig, ax = plt.subplots(ncols=3,nrows=3, figsize=(23,23))  # Create fig and axes
        ax_flat = [item for row in ax for item in row]  # Flatten list of axes
        data.plot(kind='box', subplots=True, layout=(3,3),sharex=False,
                  ax=ax_flat[:data.shape[1]])  # Plot boxes
        st.pyplot(fig)
    else:  # If not contains data
        st.write(X)


def view_pairs(X, y):
    """Creates pairs comparation graphics and density in diagonal.

    :param X: X values from splitted data, see Xy() in Preprocessor.
    :type X: DataFrame
    :param y: y values from splitted data, see Xy() in Preprocessor.
    :type y: Series
    """
    root_logger.info('view_pairs()')  # LOG - INFO
    if not isinstance(X, str):  # If contains data
        fig = sns.pairplot(pd.concat([X,y], axis=1), hue=y.name)
        st.pyplot(fig)
    else:  # If not contains data
        st.write(X)


def view_correlations(X, y):
    """Creates correlation graph.

    :param X: X values from splitted data, see Xy() in Preprocessor.
    :type X: DataFrame
    :param y: y values from splitted data, see Xy() in Preprocessor.
    :type y: Series
    """
    if not isinstance(X, str):  # If contains data
        root_logger.info('view_correlations()')  # LOG - INFO
        data = pd.concat([X,y], axis=1)
        correlations = data.corr()  # gets correlations
        fig = px.imshow(correlations,color_continuous_scale='viridis',
                        x=data.columns,y=data.columns)  # Creates graph
        fig.update_xaxes(side="top")
        st.plotly_chart(fig, theme='streamlit', use_container_width=True)
    else:  # If not contains data
        st.write(X)
    

def view_pca(X, p, class_type, columns):
    """Creates PCA 3 components graph

    :param X: X values from splitted data, see Xy() in Preprocessor.
    :type X: DataFrame
    :param p: Preprocessor instance
    :type p: Preprocessor
    :param class_type: 'prec' or 'rain' to choose class
    :type class_type: str
    :param columns: columns from multiselect for drop attr columns
    :type columns: list
    """
    root_logger.info('view_pca()')  # LOG - INFO
    if not isinstance(X, str):  # If contains data
        # Drop columns from multiselect and calculates ratios PCA
        pca3, ratio = p.select_attr(X.drop(columns=columns), 'PCA', n_components=3)
        # Creates graph
        fig = px.scatter_3d(pca3, x=0,y=1,z=2, 
                    color=p.y[class_type], color_continuous_scale='viridis',
                    title=f'total explained variance:{ratio.sum()*100:.2f}',
                    labels={'0':'PCA1','1':'PCA2','2':'PCA3'})
        st.plotly_chart(fig)
    else:  # If not contains data
        st.write(X)


def view_kbest(X, p, class_type):
    """Show the kbest graph for attr data.

    :param X: X values from splitted data, see Xy() in Preprocessor.
    :type X: DataFrame
    :param p: Preprocessor instance
    :type p: Preprocessor
    :param class_type: 'prec' or 'rain' to choose class
    :type class_type: str
    """
    root_logger.info('view_kbest()')  # LOG - INFO
    if not isinstance(X, str):  # If contains data
        targets = X.columns
        if class_type == 'prec':  # For regression
            rain_kb_clf, ratio = p.select_attr(X, 'Kbest', y=p.y['prec'], k='all',
                               score_func='f_classif')  # Ratios
            fig = plt.figure(figsize=(10, 4))
            # - Graph -
            sns.lineplot(x=targets, y=ratio, ax=plt.gca())
            plt.gca().grid()
            plt.title("f_classif")

        else:  # For classification
            # Ratios
            rain_kb_clf, ratio1 = p.select_attr(X, 'Kbest', y=p.y['rain'], k='all',
                               score_func='f_classif')
            rain_kb_chi2, ratio2 = p.select_attr(X, 'Kbest', y=p.y['rain'], k='all',
                                        score_func='chi2')
            rain_kb, ratio3 = p.select_attr(X, 'Kbest', y=p.y['rain'], k='all',
                                        score_func='mutual_info_classif')
            # Figure
            plt.rcParams['figure.figsize'] = [15, 15]
            fig, ax = plt.subplots(nrows=3)
            
            # - Graph 1 -
            ax[0].plot(targets, ratio1)
            ax[0].grid()
            ax[0].set_title("f_classif")
            # - Graph 2 -
            ax[1].plot(targets, ratio2)
            ax[1].grid()
            ax[1].set_title("chi2")
            # - Graph 3 -
            ax[2].plot(targets, ratio3)
            ax[2].grid()
            ax[2].set_title("mutual_info_classif")
        st.pyplot(fig)
    else:  # If not contains data
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
    prec_sc, rain_sc = p.preprocess_data(verbose=True, drop=False)

    with st.sidebar:  # Side menu
        st.markdown('''# TFM - Torres Espinosa\n ### Data Location:''')
        
        # Selectbox to choose class and graphic to view
        options = ('prec','rain')  # Class options
        class_type = st.selectbox("Class type: ", options)  # choose class type
        graphics = ('density','box','pairs','correlations','PCA','Kbest')  # Graphics types
        graphic = st.selectbox("Class type: ", graphics)  # choose graph
        
    # Title
    st.title('Analysis')
    # Data
    sc = prec_sc if class_type == 'prec' else rain_sc
    args = (sc, p.y[class_type])

    # Function selector by graphic type
    if graphic == 'density': view_density(*args)
    elif graphic == 'box': view_box(*args)
    elif graphic == 'pairs': view_pairs(*args)
    elif graphic == 'correlations': view_correlations(*args)
    elif graphic == 'PCA':
        cols = st.multiselect('Drop attributes:',sc.columns)  # For drop attrs
        view_pca(sc, p, class_type,cols)
    elif graphic == 'Kbest': view_kbest(sc, p, class_type)


if __name__ == '__main__':  # Run the application
    app = App()
    main()