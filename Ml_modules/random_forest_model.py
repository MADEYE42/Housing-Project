import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def random_forest_model(
  df:pd.DataFrame(),predicting_column:str,input_column:list,input_values:list      
):
    x = df[input_column]
    y = df[predicting_column]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
    model = RandomForestRegressor()
    model.fit(x_train,y_train)
    predicted_value = model.predict([input_values])
    return predicted_value