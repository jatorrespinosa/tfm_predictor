# tfm_predictor

Its a Machine learning App to predict wheather in Andalusia. Uses Aemet 
data through its Open Data Api. Requests the stations data and processes it with
internal code. This app shows data graphs, analysis graps and accuracy 
results of Machine Learning models.

This app runs with streamlit and the machine learning section uses scikit-learn.
Developed by Torres Espinosa, Jose Antonio.

Deploy with the sentence: **streamlit run app/home.py**
Needs the 'utils/keygen.py' file to make the correct requests.


## Pages
### Home
Contains a brief description, source data and formated data.
### Graphics
Displays the evolution of temperature, pressure and wind attributes.
### Analysis
Shows a several graphs to research the attributes, such as density, box plots,
correlations, pca and kbest.
### ML Results
For classification and regression, shows the different results of the proposed models.
