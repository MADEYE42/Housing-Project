import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
def linear_model(
  df:pd.DataFrame(),predicting_column:str,input_column:list,input_values:list      
):
    X = df[input_column]
    y = df[predicting_column]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
    model = LinearRegression()
    model.fit(X_train,y_train)
    predicted_value = model.predict([input_values])
    return predicted_value