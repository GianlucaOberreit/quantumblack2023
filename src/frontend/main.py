import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.layers import GaussianNoise
import tensorflow as tf
from tensorflow import keras
from keras import models, layers, regularizers
from keras.metrics import Recall, Precision
import math
from keras import Model
import matplotlib as plt
from scipy.ndimage import zoom

def set_up(hex_color, max_width=1200, padding_top=1, padding_right=1, padding_left=1, padding_bottom=1, text_color="#FFF", background_color="#0A100D"):
    st.set_page_config(layout="wide")
    st.markdown(
        f"""    
        <style>
            .reportview-container .main .block-container {{
                max-width: {max_width}px;
                padding-top: {padding_top}rem;
                padding-right: {padding_right}rem;
                padding-left: {padding_left}rem;
                padding-bottom: {padding_bottom}rem;
            }}
            .reportview-container .main {{
                color: {text_color};
                background-color: {background_color};
            }}
            .stApp {{
                background-color: {hex_color};
            }}
            .centered {{
                text-align: center;
            }}
            .text {{
                font-family: 'Montserrat', sans-serif;  
                font-weight: bold;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_up("#0A100D") # Set background color and other properties

def define_model(name, input_shape, seed=31415):
    # Setting seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Getting the parameters
    filters = [8, 16, 64]
    filters_dim = [3, 3, 5, 5]
    max_pool_dim = [1, 2, 1, 2]
    dropout = 0.2
    classification_layers = [16, 8]
    localization_layers = [64, 16]

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        # layers.RandomRotation(0.2),
    ])

    # Backbone model
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    # model.add(data_augmentation)
    model.add(GaussianNoise(0.01))
    model.add(layers.Lambda(lambda x: tf.abs(x)))
    model.add(layers.Conv2D(filters[0], filters_dim[0], activation='relu', input_shape=input_shape,
                            kernel_regularizer=regularizers.l1_l2(0.01, 0.03)))
    model.add(layers.MaxPooling2D(max_pool_dim[0]))
    # if dropout:
    #     model.add(layers.Dropout(dropout))

    for num_filter, conv_dim, max_pool in zip(filters[1:], filters_dim[1:], max_pool_dim[1:]):
        model.add(layers.Conv2D(num_filter, conv_dim, activation='relu'))
        model.add(layers.MaxPooling2D(max_pool))
        # if dropout:
        #     model.add(layers.Dropout(dropout))

    model.add(layers.GlobalAveragePooling2D())
    flattened_output = model.output

    # Final layers for classification
    class_prediction = layers.Dense(classification_layers[0], activation="relu",
                                    kernel_regularizer=regularizers.l1_l2(0.01, 0.03))(flattened_output)
    class_prediction = layers.Dropout(dropout)(class_prediction)
    for classification_layer in classification_layers[1:]:
        class_prediction = layers.Dense(classification_layer, activation="relu")(class_prediction)
        # class_prediction = layers.Dropout(dropout)(class_prediction)
    class_prediction = layers.Dense(1, activation='sigmoid', name="class_output")(class_prediction)

    # Final layers for localization
    box_output = layers.Dense(localization_layers[0], activation="relu")(flattened_output)
    box_output = layers.Dropout(dropout)(box_output)
    for localization_layer in localization_layers[1:]:
        box_output = layers.Dense(localization_layer, activation="relu")(box_output)
        box_output = layers.Dropout(dropout)(box_output)
    box_predictions = layers.Dense(2, activation='sigmoid', name="box_output")(box_output)

    complete_model = keras.Model(name=name, inputs=model.input, outputs=[box_predictions, class_prediction])
    return complete_model


# Load your trained model (make sure to provide the correct path)
# @st.cache(allow_output_mutation=True)
def load_model(model_path):
    # model = tf.keras.models.load_model(model_path)
    model = define_model(name='production', input_shape=(64, 64, 1))
    model.load_weights(model_path)
    # model = tf.keras.model.load_weights(model_path)
    return model

def convert_I_to_L(img):
    array = np.uint8(np.array(img) / 256)
    return Image.fromarray(array)

def process_image(image):
    image = image.resize((64, 64))  # Replace with the model's expected input size
    image = np.array(image) / 2 ** 16
    image = np.expand_dims(image, axis=0)  # Model expects a batch of images
    return image


def create_heatmap(img, model):
    conv_output = model.get_layer("max_pooling2d_2").output
    pred_ouptut = model.get_layer("dense").output
    heatmap_model = Model(model.input, outputs=[conv_output, pred_ouptut])

    conv, pred = heatmap_model.predict(img.reshape([*img.shape, 1]), verbose=0)
    target = np.argmax(pred, axis=1).squeeze()
    w, b = heatmap_model.get_layer("dense").weights
    scaleh = model.input.shape[1] / conv_output.shape[1]
    scalew = model.input.shape[2] / conv_output.shape[2]
    weights = w[:, target].numpy()
    heatmap = conv.squeeze() @ weights
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(zoom(heatmap, zoom=(scaleh, scalew)), cmap='jet', alpha=0.5)
    plt.savefig('../data/test.png', bbox_inches='tight', pad_inches=0)

    return heatmap[:-20, 25:]

def Home():
    # Layout with columns
    col1, col2, col3, col4 = st.columns([1,14,1.2,1])

    with col1:
        # Display small logo
        st.image("./data/small_logo.png", width=100)

    with col3:
        if st.button('Sign Up'):
            pass  # Perform sign-up action
    with col4:
        if st.button('Log In'):
            pass  # Perform log-in action


    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        empty_space(4)
        st.image("./data/big_logo.jpg", width=780)
        # Display and center the fixed string using markdown
    empty_space(4)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        fixed_string = "Atmospheric intelligence to \ninform the energy transition"
        st.markdown(f"<div class='centered'><h1 class='text'>{fixed_string}</h1></div>", unsafe_allow_html=True)
        model = load_model('./data/87.2.h5')

        st.title('Try it yourself')
        uploaded_file = st.file_uploader("Choose a TIFF image...", type="tiff")
        # Upload TIFF image
    if uploaded_file is not None:
        # Open the image using PIL
        image = Image.open(uploaded_file)

        # Convert image to 'RGB' if it's not already
        rgbimage = convert_I_to_L(image).convert('RGB')

        # Display the uploaded (and possibly converted) image
        col1, col2, col3 = st.columns([3, 1, 3])
        with col2:
            st.image(rgbimage, caption='Uploaded Image', width=200)

        # Process the image for your model (adjust according to your model's needs)
        # Example: resize image, scale pixel values, etc.
        processed_image = process_image(image)
        create_heatmap(processed_image, model)
        col1, col2, col3 = st.columns([3, 1, 3])
        with col2:
            st.image('./data/tmp_heatmap.jpg', caption='Uploaded Image', width=200)
        # Predict using the model
        prediction = model.predict(processed_image)
        box_pred, class_pred = prediction
        col1, col2, col3 = st.columns([1, 2, 1])
        value = class_pred[0][0]*100
        if value > 50:
            value = math.trunc(value * 100) / 100.0
            result = f'AtmospheR detected a plume in the image with a confidence of {value}%'
        else:
            value = 100 - value
            value = math.trunc(value * 100) / 100.0
            result = f'AtmospheR detected no plumes in the image with a confidence of {value}%'
        with col2:
            st.write(result)
    empty_space(4)
    st.markdown(f"<div class='centered'><h1 class='text'>Learn about our brand new products</h1></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='centered'>Fast, accurate, AI-powered pollutant detection for industry leaders, policymakers, and the citizens of tomorrow</div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([1,2,2,1])
    with col2:
        st.image('./data/TrackR.png', width = 400)

    with col3:
        st.image('./data/ForecastR.png', width=400)



def empty_space(i):
    for _ in range(i):
        st.markdown('#')
if __name__ == '__main__':

    Home()