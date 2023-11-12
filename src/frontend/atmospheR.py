import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io

def AtmospheR_page():
    # Load your trained model (make sure to provide the correct path)
    # @st.cache(allow_output_mutation=True)
    def load_model(model_path):
        model = tf.keras.models.load_model(model_path)
        return model

    model = load_model('./data/95.h5')

    # Streamlit app starts here
    st.title('TIFF Image Classifier and Localization')

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
            image = np.array(image) / 2**16
            image = np.expand_dims(image, axis=0)  # Model expects a batch of images
            return image

        processed_image = process_image(image)

        # Predict using the model
        prediction = model.predict(processed_image)
        box_pred, class_pred = prediction

        # Display the results
        st.write(f'Classification Result: {class_pred}')
        st.write(f'Bounding Box Coordinates: {box_pred}')