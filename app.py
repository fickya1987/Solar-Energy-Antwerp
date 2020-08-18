import pickle
import streamlit as st
import pandas as pd
import numpy as np

model = pickle.load(open('./xgbmodel.pkl', 'rb'))

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('panels.jpeg')
    image_playground = Image.open('playground.jpeg')

    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "Would you like to predict a single day or upload a .csv?",
    ("Single Day", "Upload .csv"))

    st.sidebar.info('Using a Machine Learning model to predict the kW production of Solar Panels in Antwerp, Belgium')
    st.sidebar.success('https://tmplayground.com')
    st.sidebar.success('https://github.com/thabied')
    st.sidebar.image(image_ground)

    st.title("Insurance Charges Prediction App")

    if add_selectbox == 'Single Day':

        age = st.number_input('Age', min_value=1, max_value=100, value=25)
        sex = st.selectbox('Sex', ['male', 'female'])
        bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
        children = st.selectbox('Children', [0,1,2,3,4,5,6,7,8,9,10])
        if st.checkbox('Smoker'):
            smoker = 'yes'
        else:
            smoker = 'no'
        region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

        output=""

        input_dict = {'age' : age, 'sex' : sex, 'bmi' : bmi, 'children' : children, 'smoker' : smoker, 'region' : region}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = '$' + str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Upload .csv':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
