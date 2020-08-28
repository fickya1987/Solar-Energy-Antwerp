import streamlit as st
import pickle
from pycaret.regression import load_model, predict_model
import pandas as pd
import numpy as np

model = load_model('deployment_1')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('panels.jpeg')

    st.image(image,use_column_width=True)

    add_selectbox = st.sidebar.selectbox(
    "Would you like to predict a single day or upload a .csv?",
    ("Single Day", "Upload .csv"))

    st.sidebar.info('Using a Machine Learning model to predict the kW production of Solar Panels in Antwerp, Belgium')
    st.sidebar.info('Please refer to the GitHub repo to view the Weather Mapping for the "Weather Condition" input')
    st.sidebar.success('https://tmplayground.com')
    st.sidebar.success('https://github.com/thabied')
    st.sidebar.image('https://media.giphy.com/media/l1J9Nd2okdiIq7K9O/giphy.gif',use_column_width=True)

    st.title("Solar Power Prediction Application")

    if add_selectbox == 'Single Day':

        year = st.number_input('Year', min_value=1990, max_value=2550, value=2014)
        month = st.number_input('Month', min_value=1, max_value=12, value=5)
        day = st.number_input('Day', min_value=1, max_value=31, value=16)
        temp = st.slider('Temperature (celsius)', min_value=-0, max_value=50, value=18, steps=1)
        weather = st.number_input('Weather Condition', min_value=1, max_value=130, value=50)
        wind = st.number_input('Wind (km/h)', min_value=1, max_value=50, value=14)
        humidity = st.slider('Humidity (%)', min_value=0, max_value=100, value=0, step=1)
        barometer = st.number_input('Atmospheric Pressure (milibar)', min_value=1, max_value=2000, value=25)
        visibility = st.number_input('Visibility (km)', min_value=0, max_value=40, value=25)

        output=""

        input_dict = {'year' : year, 'month' : month, 'day' : day, 'temp' : temp, 'weather' : weather, 'wind' : wind, 'humidity' : humidity, 'barometer' : barometer, 'visibility' : visibility}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):

            output = predict(model=model, input_df=input_df)
            output = str(output)+'kWs'

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Upload .csv':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
