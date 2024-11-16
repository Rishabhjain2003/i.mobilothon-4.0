import streamlit as st
from tensorflow import keras
import numpy as np
import tensorflow as tf  # Import TensorFlow
import base64
import math


# Define the MultiHeadAttentionLayer class 
@tf.keras.utils.register_keras_serializable()
class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        self.W_query = self.add_weight(name='query_weight', shape=(input_shape[-1], input_shape[-1]), initializer='he_normal')
        self.W_key = self.add_weight(name='key_weight', shape=(input_shape[-1], input_shape[-1]), initializer='he_normal')
        self.W_value = self.add_weight(name='value_weight', shape=(input_shape[-1], input_shape[-1]), initializer='he_normal')
        super(MultiHeadAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        queries = tf.matmul(inputs, self.W_query)
        keys = tf.matmul(inputs, self.W_key)
        values = tf.matmul(inputs, self.W_value)

        # Compute attention scores and weights
        scores = tf.matmul(queries, keys, transpose_b=True) / tf.sqrt(float(inputs.shape[-1]))
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Compute the context vector as a weighted sum of values
        context_vector = tf.matmul(attention_weights, values)
        return tf.reduce_sum(context_vector, axis=1)
    def get_config(self):
        config = super(MultiHeadAttentionLayer, self).get_config()
        config.update({"num_heads": self.num_heads})
        return config


# Load the model
model = keras.models.load_model("predictive_maintenance_ann.keras", custom_objects={'MultiHeadAttentionLayer': MultiHeadAttentionLayer})

# Define the customized ranges for each feature based on dataset statistics
custom_ranges = {
    'Engine rpm': (61.0, 2239.0),
    'Lub oil pressure': (0.003384, 7.265566),
    'Fuel pressure': (0.003187, 21.138326),
    'Coolant pressure': (0.002483, 7.478505),
    'lub oil temp': (71.321974, 89.580796),
    'Coolant temp': (61.673325, 195.527912),
}

# Feature Descriptions
feature_descriptions = {
    'Engine RPM': 'Revolution per minute of the engine.',
    'Lube Oil Pressure': 'Pressure of the lubricating oil.',
    'Fuel Pressure': 'Pressure of the fuel.',
    'Coolant Pressure': 'Pressure of the coolant.',
    'Lube Oil Temp': 'Temperature of the lubricating oil.',
    'Coolant Temp': 'Temperature of the coolant.',
}



def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background('background.jpg')

# Engine Condition Prediction App
def main():
    st.title("Predictive Maintenance System")

    # Display feature descriptions
    st.sidebar.title("Feature Descriptions")
    for feature, description in feature_descriptions.items():
        st.sidebar.markdown(f"**{feature}:** {description}")

    # Input widgets with customized ranges
    engine_rpm = st.slider("Engine RPM", min_value=float(custom_ranges['Engine rpm'][0]), 
                           max_value=float(custom_ranges['Engine rpm'][1]), 
                           value=float(custom_ranges['Engine rpm'][1] / 2))
    lub_oil_pressure = st.slider("Lub Oil Pressure", min_value=custom_ranges['Lub oil pressure'][0], 
                                 max_value=custom_ranges['Lub oil pressure'][1], 
                                 value=(custom_ranges['Lub oil pressure'][0] + custom_ranges['Lub oil pressure'][1]) / 2)
    fuel_pressure = st.slider("Fuel Pressure", min_value=custom_ranges['Fuel pressure'][0], 
                              max_value=custom_ranges['Fuel pressure'][1], 
                              value=(custom_ranges['Fuel pressure'][0] + custom_ranges['Fuel pressure'][1]) / 2)
    coolant_pressure = st.slider("Coolant Pressure", min_value=custom_ranges['Coolant pressure'][0], 
                                 max_value=custom_ranges['Coolant pressure'][1], 
                                 value=(custom_ranges['Coolant pressure'][0] + custom_ranges['Coolant pressure'][1]) / 2)
    lub_oil_temp = st.slider("Lub Oil Temperature", min_value=custom_ranges['lub oil temp'][0], 
                             max_value=custom_ranges['lub oil temp'][1], 
                             value=(custom_ranges['lub oil temp'][0] + custom_ranges['lub oil temp'][1]) / 2)
    coolant_temp = st.slider("Coolant Temperature", min_value=custom_ranges['Coolant temp'][0], 
                             max_value=custom_ranges['Coolant temp'][1], 
                             value=(custom_ranges['Coolant temp'][0] + custom_ranges['Coolant temp'][1]) / 2)

    # Predict button
    if st.button("Predict Engine Condition"):
        temp_difference = coolant_temp - lub_oil_temp
        result = predict_condition(engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference)
        result = math.ceil(abs(result[0][0]))
        print(result)
        # Explanation
        if result <= 50:
            st.info(f"The engine is in good condition. As the  Confidence level of the machine is : {100.0 - result}%")
        else:
            st.warning(f"Warning! Please investigate further, the  Confidence level of the machine is : {100.0 - result}%")

    # Reset button
    if st.button("Reset Values"):
        st.experimental_rerun()

# Function to predict engine condition
def predict_condition(engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference):
    input_data = np.array([engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference]).reshape(1, -1)
    input_data_reshaped = input_data.reshape(input_data.shape[0], input_data.shape[1], 1)
    prediction = model.predict(input_data_reshaped)
    print(prediction)
    return prediction

if __name__ == "__main__":
    main()
