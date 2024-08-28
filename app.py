import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

st.title('Mushroom Edibility Predictor')

# Create input fields for each feature
cap_diameter = st.number_input('Cap Diameter (cm)', min_value=0.0, max_value=30.0, value=5.0)
cap_shape = st.selectbox('Cap Shape', ['b', 'c', 'x', 'f', 's', 'p', 'o'])
cap_surface = st.selectbox('Cap Surface', ['i', 'g', 'y', 's', 'h', 'l', 'k', 't', 'w', 'e'])
cap_color = st.selectbox('Cap Color', ['n', 'b', 'g', 'r', 'p', 'u', 'e', 'w', 'y', 'l', 'o', 'k'])
does_bruise_bleed = st.selectbox('Does Bruise/Bleed', ['t', 'f'])
gill_attachment = st.selectbox('Gill Attachment', ['a', 'x', 'd', 'e', 's', 'p', 'f', '?'])
gill_spacing = st.selectbox('Gill Spacing', ['c', 'd', 'f'])
gill_color = st.selectbox('Gill Color', ['n', 'b', 'g', 'r', 'p', 'u', 'e', 'w', 'y', 'l', 'o', 'k', 'f'])
stem_height = st.number_input('Stem Height (cm)', min_value=0.0, max_value=35.0, value=5.0)
stem_width = st.number_input('Stem Width (mm)', min_value=0.0, max_value=100.0, value=10.0)
stem_surface = st.selectbox('Stem Surface', ['i', 'g', 'y', 's', 'h',  'l', 'k', 't', 'w', 'e', 'f'])
stem_color = st.selectbox('Stem Color', ['n', 'b', 'g', 'r', 'p', 'u', 'e', 'w', 'y', 'l', 'o', 'k', 'f'])
has_ring = st.selectbox('Has Ring', ['t', 'f'])
ring_type = st.selectbox('Ring Type', ['c', 'e', 'r', 'g', 'l', 'p', 's', 'z', 'y', 'm', 'f', '?'])
habitat = st.selectbox('Habitat', ['g', 'l', 'm', 'p', 'h', 'u', 'w', 'd'])

if st.button('Predict'):
    try:
        # Load the trained model
        model = tf.keras.models.load_model('mushroom_model.h5')
        
        # Prepare the input data
        input_data = pd.DataFrame({
            'cap-diameter': [cap_diameter],
            'cap-shape': [cap_shape],
            'cap-surface': [cap_surface],
            'cap-color': [cap_color],
            'does-bruise-bleed': [does_bruise_bleed],
            'gill-attachment': [gill_attachment],
            'gill-spacing': [gill_spacing],
            'gill-color': [gill_color],
            'stem-height': [stem_height],
            'stem-width': [stem_width],
            'stem-surface': [stem_surface],
            'stem-color': [stem_color],
            'has-ring': [has_ring],
            'ring-type': [ring_type],
            'habitat': [habitat]
        })

        # Handle missing values
        imputer = SimpleImputer(strategy='most_frequent')
        input_data_imputed = pd.DataFrame(imputer.fit_transform(input_data), columns=input_data.columns)

        # Encode categorical variables
        label_encoders = {}
        for column in input_data_imputed.columns:
            if input_data_imputed[column].dtype == 'object':
                label_encoders[column] = LabelEncoder()
                input_data_imputed[column] = label_encoders[column].fit_transform(input_data_imputed[column])

        # Scale the input data
        scaler = StandardScaler()
        input_data_scaled = scaler.fit_transform(input_data_imputed)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Display result
        if prediction[0][0] > 0.5:
            st.write('This mushroom is predicted to be: POISONOUS')
        else:
            st.write('This mushroom is predicted to be: EDIBLE')

        st.write(f'Prediction probability: {prediction[0][0]:.2f}')
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Model structure:")
        try:
            model = tf.keras.models.load_model('mushroom_model.h5')
            st.write(model.summary())
        except Exception as model_error:
            st.error(f"Could not load model: {str(model_error)}") 