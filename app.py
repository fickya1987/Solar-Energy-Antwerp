import streamlit as st
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

model = pickle.load(open('./xgbmodel.json', "rb"))

def run():

    from PIL import Image
    image = Image.open('panels.jpeg')
    image_playground = Image.open('playground.jpeg')

    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "Would you like to predict a single day or upload a .csv?",
    ("Single Day", "Upload .csv"))

    st.sidebar.info('Using a Machine Learning model to predict the kW production of Solar Panels in Antwerp, Belgium')
    st.sidebar.info('Please refer to the GitHub repo to view the Weather Mapping for the "Weather Condition" input')
    st.sidebar.success('https://tmplayground.com')
    st.sidebar.success('https://github.com/thabied')
    st.sidebar.image(image_playground)

    st.title("Solar Power Prediction Application")

    if add_selectbox == 'Single Day':

        month = st.number_input('Month', min_value=1, max_value=12, value=5)
        day = st.number_input('Day', min_value=1, max_value=31, value=16)
        temp = st.number_input('Temperature (celsius)', min_value=-60, max_value=50, value=25)
        weather = st.number_input('Weather Condition', min_value=1, max_value=130, value=50)
        wind = st.number_input('Wind (km/h)', min_value=1, max_value=50, value=14)
        humidity = st.number_input('Humidity (%)', min_value=1, max_value=100, value=60)
        barometer = st.number_input('Atmospheric Pressure (milibar)', min_value=1, max_value=2000, value=25)
        visibility = st.number_input('Visibility (km)', min_value=0, max_value=40, value=25)

        output=""

        input_dict = {'month' : month, 'day' : day, 'temp' : temp, 'weather' : weather, 'wind' : wind, 'humidity' : humidity, 'barometer' : barometer, 'visibility' : visibility}
        input_df = pd.DataFrame([input_dict])
        input_df = scaler.fit_transform(input_df)

        if st.button("Predict"):

            output = model.predict(input_df)
            output = str(output)+'kWs'

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Upload .csv':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = model.predict(data)
            st.write(predictions)

if __name__ == '__main__':
    run()
