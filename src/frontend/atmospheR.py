import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from keras.layers import GaussianNoise
import tensorflow as tf
from tensorflow import keras
from keras import models, layers, regularizers
from keras.metrics import Recall, Precision


def AtmospheR_page():
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

    model = load_model('./data/87.2.h5')

    # Streamlit app starts here
    st.title('TIFF Image Classifier and Localization')
    print('hello')
    # Upload TIFF image
    uploaded_file = st.file_uploader("Choose a TIFF image...", type="tiff")
    if uploaded_file is not None:
        # Open the image using PIL
        image = Image.open(uploaded_file)

        def convert_I_to_L(img):
            array = np.uint8(np.array(img) / 256)
            return Image.fromarray(array)

        # Convert image to 'RGB' if it's not already
        rgbimage = convert_I_to_L(image).convert('RGB')

        # Display the uploaded (and possibly converted) image
        st.image(rgbimage, caption='Uploaded Image', use_column_width=True)

        # Process the image for your model (adjust according to your model's needs)
        # Example: resize image, scale pixel values, etc.
        def process_image(image):
            image = image.resize((64, 64))  # Replace with your model's expected input size
            image = np.array(image) / 2 ** 16
            image = np.expand_dims(image, axis=0)  # Model expects a batch of images
            return image

        processed_image = process_image(image)

        # Predict using the model
        prediction = model.predict(processed_image)
        print(prediction)
        box_pred, class_pred = prediction

        st.write(f'Classification Result: {class_pred}')