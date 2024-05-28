import streamlit as st
import pandas as pd
import pickle
from io import StringIO

from Ml_modules.decision_tree_model import decision_tree_model
from Ml_modules.random_forest_model import random_forest_model
from Ml_modules.linear_regression_model import linear_model

primaryColor="#FFC0CB"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"
bytes_data = pd.DataFrame()
uploaded_files = st.file_uploader("Choose a CSV file", type = "csv" , accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)

prompt = st.chat_input("How do you feel my nigger")
if prompt:
    st.write(f"My beloved Nigger has spoken : {prompt}")
st.title('Predicting House Price')
@st.cache_data
def load_data(filename:str,columns:list = pd.DataFrame().columns):
    df = pd.read_csv('source_files\\melb_data.csv')
    df = df[columns]
    return df
data = load_data(bytes_data,['Rooms','Bathroom','Distance','Landsize','Price'])
#data['Ratio'] = data['Landsize'] % data['Bedroom2']
model_name = st.selectbox(
    label = "Select Machine Learning Model",
    options = ['Decision Tree','Random Forest','Linear Regression'],
)
entered_rooms = st.slider(
    label = 'Enter Number of Rooms',min_value = 0,max_value = 10,value = 3
)
entered_Bathroom = st.slider(
    label = 'Enter Number of Bathroom',min_value = 0,max_value = 8,value = 3
)
entered_Distance = st.slider(
    label = 'Enter Number of Distance',min_value = 0.0,max_value = 50.0,value = 8.0
)
entered_Lansize = st.slider(
    label = 'Enter Number of Lansize',min_value = 0,max_value = 30000,value = 300
)
#def model_predict(input_columns:list , model_name):
#    model_name = model_name.lower(),replace('','_')
#    with.open(f'source_files/(model_name).py','rb') as f :
#        model = pickle.load(f)
#        predicting_value = model.predict(input_columns)
btn_clicked = st.button(label = 'Predict',type = 'primary')
if btn_clicked:
    predicting_value = [0]
    input_columns = ['Rooms','Bathroom','Distance','Landsize']
    match model_name:        
        case "Decision Tree":
            predicting_value = decision_tree_model(
                data,'Price',['Rooms','Bathroom','Distance','Landsize'],[entered_rooms,entered_Lansize,entered_Distance,entered_Bathroom],
            )
        case "Random Forest":
            predicting_value = random_forest_model(
                data,'Price',['Rooms','Bathroom','Distance','Landsize'],[entered_rooms,entered_Lansize,entered_Distance,entered_Bathroom],
            )
        case "Linear Regression":
            predicting_value = linear_model(
                data,'Price',['Rooms','Bathroom','Distance','Landsize'],[entered_rooms,entered_Lansize,entered_Distance,entered_Bathroom],
            )
    predicting_value = f"${predicting_value[0]: .2f}"
    st.header(predicting_value,divider = "rainbow")