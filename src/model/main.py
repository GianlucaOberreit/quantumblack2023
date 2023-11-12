# -*- coding: utf-8 -*-
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.metrics import Recall, Precision

import numpy as np
import random as rn
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Recall, Precision
import time
import rasterio

DATA_PATH = "./data/"
IMAGE_SIZE = 64


@tf.function
def giou_loss_with_center(y_true_center, y_pred_center):
    """
    Computes the Generalized Intersection over Union (GIoU) loss for a given pair of true and predicted bounding box centers.
    This function assumes a fixed height and width for the bounding boxes and calculates GIoU based on these dimensions.

    The GIoU is a metric that measures the similarity between two bounding boxes, considering both their area of overlap
    and the area of the smallest enclosing box that contains both.

    Parameters:
    y_true_center (Tensor): A tensor containing the true center coordinates of the bounding boxes.
    y_pred_center (Tensor): A tensor containing the predicted center coordinates of the bounding boxes.

    Returns:
    Tensor: A tensor containing the GIoU loss for each pair of bounding boxes.
    """
    eps = 1e-7
    h = 0.2 / 2  # Half of the bounding box height
    w = 0.2 / 2  # Half of the bounding box width

    xc1 = y_true_center[:, 0] - w
    yc1 = y_true_center[:, 1] + h
    xc2 = y_true_center[:, 0] + w
    yc2 = y_true_center[:, 1] - h

    xp1 = y_pred_center[:, 0] - w
    yp1 = y_pred_center[:, 1] + h
    xp2 = y_pred_center[:, 0] + w
    yp2 = y_pred_center[:, 1] - h

    ix1 = tf.maximum(xc1, xp1)
    iy1 = tf.minimum(yc1, yp1)
    ix2 = tf.minimum(xc2, xp2)
    iy2 = tf.maximum(yc2, yp2)

    zero = tf.zeros(tf.shape(ix1))
    aux = 2 * h * w * tf.ones(tf.shape(ix1))
    intersection = tf.multiply(
        tf.maximum((ix2 - ix1), zero), tf.maximum((iy1 - iy2), zero)
    )
    union = aux - intersection
    iou = tf.math.divide(intersection, union)

    # smallest enclosing box width
    cw = tf.maximum(xc2, xp2) - tf.minimum(xc1, xp1)
    # smallest enclosing box height
    ch = tf.maximum(yc1, yp1) - tf.minimum(yc2, yp2)
    c_area = tf.multiply(cw, ch) + eps  # smallest enclosing box area
    giou = iou - tf.divide(c_area - union, c_area)  # GIoU
    return 1 - giou


def giou_loss(y_true, y_pred):
    """
    Calculates the Generalized Intersection over Union (GIoU) loss between the predicted and true bounding boxes.
    This function computes the GIoU based on the coordinates of the bounding boxes, assuming a fixed height and width.

    GIoU is an enhancement over the traditional Intersection over Union (IoU) metric, accounting for the size of the
    smallest enclosing box covering both the predicted and true bounding boxes.

    Parameters:
    y_true (Tensor): A tensor containing the true coordinates (x, y) of the bounding boxes.
    y_pred (Tensor): A tensor containing the predicted coordinates (x, y) of the bounding boxes.

    Returns:
    Tensor: A tensor representing the GIoU loss between the predicted and true bounding boxes.
    """
    # Extracting center coordinates
    x1, y1 = tf.split(y_true, 2, axis=-1)
    x2, y2 = tf.split(y_pred, 2, axis=-1)

    # Assuming fixed width and height of 0.2
    w1 = h1 = w2 = h2 = 0.2

    # Calculating coordinates of the bounding boxes
    xt1 = x1 - w1 / 2
    yt1 = y1 - h1 / 2
    xt2 = x1 + w1 / 2
    yt2 = y1 + h1 / 2

    xp1 = x2 - w2 / 2
    yp1 = y2 - h2 / 2
    xp2 = x2 + w2 / 2
    yp2 = y2 + h2 / 2

    # Calculating intersection and union
    xi = tf.maximum(xt1, xp1)
    yi = tf.maximum(yt1, yp1)
    xi2 = tf.minimum(xt2, xp2)
    yi2 = tf.minimum(yt2, yp2)

    inter_area = tf.maximum(xi2 - xi, 0) * tf.maximum(yi2 - yi, 0)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    # Calculating GIoU
    iou = tf.math.divide_no_nan(inter_area, union_area)
    cw = tf.maximum(xt2, xp2) - tf.minimum(xt1, xp1)
    ch = tf.maximum(yt2, yp2) - tf.minimum(yt1, yp1)
    c_area = cw * ch + 1e-7
    giou = iou - tf.math.divide_no_nan(c_area - union_area, c_area)

    return 1 - giou


def get_datasets(datasets_folder):
    """
    Loads training and validation datasets from the specified folder.

    This function reads .npy files containing image data, labels, and bounding boxes for training and validation sets.
    It also converts grayscale images to RGB by repeating the grayscale channel three times.

    Parameters:
    ----------
    datasets_folder : str
        The path to the folder containing the .npy files for the datasets.

    Returns:
    -------
    dict
        A dictionary containing the training and validation datasets, both in original and RGB formats,
        along with their corresponding labels and bounding box data.
    """
    # Getting train-validation data
    x_train = np.load(os.path.join(datasets_folder, "x_train.npy"))
    y_labels_train = np.load(os.path.join(datasets_folder, "y_labels_train.npy"))
    y_boxes_train = np.load(os.path.join(datasets_folder, "y_boxes_train.npy"))

    x_valid = np.load(os.path.join(datasets_folder, "x_valid.npy"))
    y_labels_valid = np.load(os.path.join(datasets_folder, "y_labels_valid.npy"))
    y_boxes_valid = np.load(os.path.join(datasets_folder, "y_boxes_valid.npy"))

    rgb_x_train = np.squeeze(x_train)
    rgb_x_train = np.repeat(rgb_x_train[..., np.newaxis], 3, -1)
    rgb_x_valid = np.squeeze(x_valid)
    rgb_x_valid = np.repeat(rgb_x_valid[..., np.newaxis], 3, -1)

    train_valid_data = {
        "x_train": x_train,
        "rgb_x_train": rgb_x_train,
        "y_labels_train": y_labels_train,
        "y_boxes_train": y_boxes_train,
        "x_valid": x_valid,
        "rgb_x_valid": rgb_x_valid,
        "y_labels_valid": y_labels_valid,
        "y_boxes_valid": y_boxes_valid,
    }

    return train_valid_data


def organize_data(datasets_folder):
    """
    Organizes the dataset into .npy files.

    This function reads a metadata CSV file from the specified dataset folder,
    extracts the path to grayscale image files along with their corresponding plume detection flag
    and coordinates. It reads each image using rasterio, processes it, and saves the arrays
    into separate .npy files for training and validation datasets.

    Parameters:
    ----------
    datasets_folder : str
        The path to the folder containing the dataset and metadata.csv.

    Returns:
    -------
    None
        This function does not return any value. It saves .npy files to disk.

    Saves Files:
    -----------
    - X_train.npy : NumPy array file containing the training image data.
    - X_valid.npy : NumPy array file containing the validation image data.
    - y_labels_train.npy : NumPy array file containing the training labels.
    - y_labels_valid.npy : NumPy array file containing the validation labels.
    - y_boxes_train.npy : NumPy array file containing the training bounding box coordinates.
    - y_boxes_valid.npy : NumPy array file containing the validation bounding box coordinates.
    """
    csv_file_path = datasets_folder + "metadata.csv"
    selected_columns = ["path", "plume", "coord_x", "coord_y"]
    df = pd.read_csv(csv_file_path, usecols=selected_columns)
    size = df.shape[0]
    x = np.empty((size, IMAGE_SIZE, IMAGE_SIZE))
    y = np.empty(size)
    y_boxes = np.empty((size, 2))
    for index, row in df.iterrows():
        photo = rasterio.open(datasets_folder + row["path"] + ".tif")
        data = photo.read(1)
        scaled_data = data / 2**16
        x[index] = scaled_data
        plume = 1 if row["plume"] == "yes" else 0
        for i in range(1):
            y[index + i] = int(plume)
        y_boxes[index] = np.array(
            [row["coord_x"] / image_width, row["coord_y"] / image_heigth]
        )
    datanames = [
        "x_train",
        "x_valid",
        "y_labels_train",
        "y_labels_valid",
        "y_boxes_train",
        "y_boxes_valid",
    ]
    data = train_test_split(x, y, y_boxes, test_size=0.2)
    x_train_len, y_labels_len, y_boxes_len = len(data[0]), len(data[2]), len(data[4])
    print(data[0].shape, data[2].shape, data[4].shape)
    x_train_aug = np.resize(data[0], (x_train_len * 5, IMAGE_SIZE, IMAGE_SIZE))
    y_labels_aug = np.resize(data[2], (y_labels_len * 5,))
    y_boxes_aug = np.resize(data[4], (y_boxes_len * 5, 2))
    print(x_train_aug.shape)
    for i in range(x_train_len):
        for j in range(1, 4):
            x_train_aug[i + j * x_train_len], y_boxes_aug[i + j * x_train_len] = rotate(
                x_train_aug[i + (j - 1) * x_train_len],
                y_boxes_aug[i + (j - 1) * x_train_len],
            )

        x_train_aug[i + 4 * x_train_len], y_boxes_aug[i + 4 * x_train_len] = flip(
            x_train_aug[i], y_boxes_aug[i]
        )

    data[0], data[2], data[4] = x_train_aug, y_labels_aug, y_boxes_aug
    for index, name in enumerate(datanames):
        print(len(data[index]))
        np.save(datasets_folder + f"processed_data/{name}.npy", data[index])


def rotate(x, y_boxes):
    """
    Rotates an image by 90 degrees and adjusts the bounding box coordinates accordingly.

    Parameters:
    ----------
    x : ndarray
        The image data to be rotated.
    y_boxes : ndarray
        The bounding box coordinates associated with the image.

    Returns:
    -------
    tuple
        A tuple containing the rotated image and the updated bounding box coordinates.
    """
    width, height = x.shape
    x_rotated = np.rot90(x, k=1)
    y_boxes_rotated = np.array([width - y_boxes[1], y_boxes[0]])
    return x_rotated, y_boxes_rotated


def flip(x, y_boxes):
    """
    Flips an image horizontally (left to right) and adjusts the bounding box coordinates accordingly.

    Parameters:
    ----------
    x : ndarray
        The image data to be flipped horizontally.
    y_boxes : ndarray
        The bounding box coordinates associated with the image.

    Returns:
    -------
    tuple
        A tuple containing the horizontally flipped image and the updated bounding box coordinates.
    """
    width, height = x.shape
    x_flipped = np.fliplr(x)
    y_boxes_flipped = np.array([width - y_boxes[0], y_boxes[1]])
    return x_flipped, y_boxes_flipped


organize_data(DATA_PATH)
train_valid_data = get_datasets(DATA_PATH + "processed_data/")


def define_model(name, input_shape, seed=31415, existing_model_path=None):
    """
    Defines a convolutional neural network (CNN) model for both classification and localization tasks.

    If an existing model path is provided, the function loads and returns the saved model. Otherwise, it creates a new
    model using specified parameters. The model comprises convolutional, pooling, and dense layers, with dropout and
    regularization applied. The model outputs both class predictions and bounding box coordinates.

    Parameters:
    ----------
    name : str
        Name of the model.
    input_shape : tuple
        The shape of the input data (height, width, channels).
    seed : int, optional
        Random seed for reproducibility. Defaults to 31415.
    existing_model_path : str, optional
        Path to a pre-trained model. If provided, this model is loaded instead of creating a new one.

    Returns:
    -------
    keras.Model
        The compiled model, ready for training.
    """
    if existing_model_path:
        return tf.keras.models.load_model(existing_model_path)

    # Setting seed
    rn.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Getting the parameters
    filters = [8, 16, 64]
    filters_dim = [3, 3, 5, 5]
    max_pool_dim = [1, 2, 1, 2]
    dropout = 0.1
    classification_layers = [16, 8]
    localization_layers = [64, 16]

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            # layers.RandomRotation(0.2),
        ]
    )

    # Backbone model
    model = models.Sequential()
    # model.add(layers.InputLayer(input_shape=input_shape))
    # model.add(data_augmentation)
    # model.add(GaussianNoise(0.1))
    model.add(
        layers.Conv2D(
            filters[0],
            filters_dim[0],
            activation="relu",
            input_shape=input_shape,
            kernel_regularizer=regularizers.l2(0.03),
        )
    )
    model.add(layers.MaxPooling2D(max_pool_dim[0]))
    # if dropout:
    #     model.add(layers.Dropout(dropout))

    for num_filter, conv_dim, max_pool in zip(
        filters[1:], filters_dim[1:], max_pool_dim[1:]
    ):
        model.add(layers.Conv2D(num_filter, conv_dim, activation="relu"))
        model.add(layers.MaxPooling2D(max_pool))
        # if dropout:
        #     model.add(layers.Dropout(dropout))

    model.add(layers.GlobalAveragePooling2D())
    flattened_output = model.output

    # Final layers for classification
    class_prediction = layers.Dense(
        classification_layers[0],
        activation="relu",
        kernel_regularizer=regularizers.l2(0.01),
    )(flattened_output)
    class_prediction = layers.Dropout(dropout)(class_prediction)
    for classification_layer in classification_layers[1:]:
        class_prediction = layers.Dense(classification_layer, activation="relu")(
            class_prediction
        )
        class_prediction = layers.Dropout(dropout)(class_prediction)
    class_prediction = layers.Dense(1, activation="sigmoid", name="class_output")(
        class_prediction
    )

    # Final layers for localization
    box_output = layers.Dense(localization_layers[0], activation="relu")(
        flattened_output
    )
    box_output = layers.Dropout(dropout)(box_output)
    for localization_layer in localization_layers[1:]:
        box_output = layers.Dense(localization_layer, activation="relu")(box_output)
        box_output = layers.Dropout(dropout)(box_output)
    box_predictions = layers.Dense(2, activation="sigmoid", name="box_output")(
        box_output
    )

    complete_model = keras.Model(
        name=name, inputs=model.input, outputs=[box_predictions, class_prediction]
    )
    return complete_model


def compile_model(
    name, input_shape, box_loss="mean_squared_error", existing_model_path=None
):
    """
    Compiles a CNN model for simultaneous classification and bounding box regression.

    The function defines the model using 'define_model' and then compiles it with specific loss functions and metrics
    for each output. The model is optimized for binary classification (class_output) and mean squared error (box_output).

    Parameters:
    ----------
    name : str
        Name of the model.
    input_shape : tuple
        The shape of the input data.
    box_loss : str, optional
        The loss function to be used for the bounding box regression task. Defaults to 'mean_squared_error'.
    existing_model_path : str, optional
        Path to a pre-trained model. If provided, this model is loaded and compiled.

    Returns:
    -------
    keras.Model
        The compiled model.
    """
    # Creating model
    model = define_model(
        name=name, input_shape=input_shape, existing_model_path=existing_model_path
    )

    # Compiling the model
    losses = {"box_output": box_loss, "class_output": "binary_crossentropy"}
    loss_weights = {"box_output": 0.0, "class_output": 1.0}
    metrics = {"box_output": "mse", "class_output": ["accuracy", Precision(), Recall()]}

    learning_rate = 1e-5
    # opt = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = 0.9)
    opt = tf.keras.optimizers.Adam()

    model.compile(
        optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=metrics
    )
    return model


def train_model(model, batch_size, epochs, train_valid_data):
    """
    Trains the given model on the specified training and validation datasets.

    The function takes a compiled model and trains it using the provided training and validation data.
    It supports early stopping to prevent overfitting and plots the training and validation loss at the end of training.

    Parameters:
    ----------
    model : keras.Model
        The compiled model to be trained.
    batch_size : int
        The number of samples per batch of computation.
    epochs : int
        The number of epochs to train the model.
    train_valid_data : dict
        A dictionary containing the training and validation data, including inputs and labels for both tasks.

    Returns:
    -------
    keras.Model
        The trained model.
    """
    # Getting the train-validation data
    x_train = train_valid_data["x_train"]
    y_labels_train = train_valid_data["y_labels_train"]
    y_boxes_train = train_valid_data["y_boxes_train"]
    x_valid = train_valid_data["x_valid"]
    y_labels_valid = train_valid_data["y_labels_valid"]
    y_boxes_valid = train_valid_data["y_boxes_valid"]
    early_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20)
    # Training the model
    history = model.fit(
        x=x_train,
        verbose=2,
        callbacks=[],
        y={"box_output": y_boxes_train, "class_output": y_labels_train},
        validation_data=(
            x_valid,
            {"box_output": y_boxes_valid, "class_output": y_labels_valid},
        ),
        batch_size=batch_size,
        epochs=epochs,
        sample_weight={
            "box_output": y_labels_train,
            "class_output": np.ones(y_labels_train.shape),
        },
    )
    training_loss = history.history["loss"]
    validation_loss = history.history["val_loss"]

    # Create a plot of training and validation loss
    plt.plot(range(1, len(training_loss) + 1), training_loss, label="Training Loss")
    plt.plot(
        range(1, len(validation_loss) + 1), validation_loss, label="Validation Loss"
    )
    plt.xlabel("Epochs")
    plt.yscale("log")
    plt.xlim(0, epochs)
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    return model


def test_model(model, train_valid_data):
    """
    Evaluates the trained model on the validation dataset using various metrics.

    This function uses the validation data to test the performance of the model. It calculates metrics such as
    accuracy, F1 score, precision, recall, inverse precision, inverse recall, average Intersection over Union (IoU) for bounding boxes,
    and the model's execution time. It also includes the total number of parameters in the model.

    Parameters:
    ----------
    model : keras.Model
        The trained model to be evaluated.
    train_valid_data : dict
        A dictionary containing the validation data.

    Returns:
    -------
    dict
        A dictionary containing various performance metrics of the model and the model itself.
    """
    x_valid = train_valid_data["x_valid"]
    y_labels_valid = train_valid_data["y_labels_valid"]
    y_boxes_valid = train_valid_data["y_boxes_valid"]

    # Testing the model on different metrics
    y_valid_pred = model.predict(x_valid, verbose=0)
    y_valid_pred_corner = y_valid_pred[0]
    y_valid_pred_label = np.round(y_valid_pred[1])

    run_results = {
        "accuracy": accuracy_score(y_true=y_labels_valid, y_pred=y_valid_pred_label),
        "f1": f1_score(y_true=y_labels_valid, y_pred=y_valid_pred_label),
        "precision": precision_score(y_true=y_labels_valid, y_pred=y_valid_pred_label),
        "recall": recall_score(y_true=y_labels_valid, y_pred=y_valid_pred_label),
        "f1_inv": f1_score(
            y_true=y_labels_valid, y_pred=y_valid_pred_label, pos_label=0
        ),
        "precision_inv": precision_score(
            y_true=y_labels_valid, y_pred=y_valid_pred_label, pos_label=0
        ),
        "recall_inv": recall_score(
            y_true=y_labels_valid, y_pred=y_valid_pred_label, pos_label=0
        ),
        "avg_iou": avg_iou(y_true=y_boxes_valid, y_pred=y_valid_pred_corner),
        "execution_time": execution_time(model, x_valid),
        "num_parameters": model.count_params(),
        "model": model,
    }
    return run_results


def compile_and_train(
    name,
    input_shape,
    batch_size,
    epochs,
    train_valid_data,
    box_loss,
    existing_model_path=None,
):
    """
    Compiles and trains a CNN model using specified parameters.

    This function compiles a new model or loads an existing one, then trains it using the provided training and
    validation data. The model is compiled with specific input shape, batch size, number of epochs, and loss function
    for the bounding box regression task.

    Parameters:
    ----------
    name : str
        Name of the model.
    input_shape : tuple
        The shape of the input data.
    batch_size : int
        The number of samples per batch of computation.
    epochs : int
        The number of epochs to train the model.
    train_valid_data : dict
        A dictionary containing the training and validation data.
    box_loss : tf.keras.losses
        The loss function to be used for the bounding box regression task.
    existing_model_path : str, optional
        Path to a pre-trained model. If provided, this model is loaded and trained.

    Returns:
    -------
    keras.Model
        The trained model.
    """
    model = compile_model(name, input_shape, box_loss, existing_model_path)
    trained_model = train_model(model, batch_size, epochs, train_valid_data)
    return trained_model


input_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)
batch_size = 32
epochs = 1000
my_loss = tf.keras.losses.MeanSquaredError()
folder = DATA_PATH + "models"

trained_model = compile_and_train(
    f"Ipop8", input_shape, batch_size, epochs, train_valid_data, my_loss
)

# print(trained_model.summary())
trained_model.save(os.path.join(folder, trained_model.name + ".h5"))


def bb(model, test_data, height_image=64, width_image=64, height_box=20, width_box=20):
    """
    Visualizes bounding box predictions on sample images from the test dataset.

    This function selects one sample with a plume and another without from the test dataset,
    displays the images, and overlays the predicted and actual bounding boxes.
    It also prints the model's confidence in the presence of a plume.

    Parameters:
    ----------
    model : keras.Model
        The trained model for prediction.
    test_data : dict
        Dictionary containing test dataset with image and label data.
    height_image : int, optional
        Height of the test images. Defaults to 64.
    width_image : int, optional
        Width of the test images. Defaults to 64.
    height_box : int, optional
        Height of the bounding box. Defaults to 20.
    width_box : int, optional
        Width of the bounding box. Defaults to 20.

    Returns:
    -------
    None
        The function does not return a value; it visualizes the results.
    """
    x_test = test_data["x_valid"]
    y_labels_test = test_data["y_labels_valid"]
    y_boxes_test = test_data["y_boxes_valid"]

    im_plume_index = np.where(y_labels_test == 1)[0][1]
    im_plume = x_test[im_plume_index, :, :]
    y_true_plume_corner = y_boxes_test[im_plume_index]

    im_no_plume_index = np.where(y_labels_test == 0)[0][2]
    im_no_plume = x_test[im_no_plume_index, :, :]
    y_true_no_plume_corner = y_boxes_test[im_no_plume_index]

    y_pred_plume = model.predict(im_plume.reshape([1, *im_plume.shape]))
    y_pred_no_plume = model.predict(im_no_plume.reshape([1, *im_no_plume.shape]))

    y_pred_plume_corner = y_pred_plume[0][0]
    y_pred_no_plume_corner = y_pred_no_plume[0][0]

    print(f"Confidence in image with plume: {y_pred_plume[1][0][0]}")
    print(f"Confidence in image without plume: {y_pred_no_plume[1][0][0]}")
    print(f"Predicted plume coordinates: {y_pred_plume_corner}")

    plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    ax1.set_title("With Plume")
    ax1.tick_params("y", labelleft=False)
    ax1.tick_params("x", labelbottom=False)
    ax1.imshow(im_plume, cmap="gray")
    rect_pred = plt.Rectangle(
        (
            y_pred_plume_corner[0] * width_image - width_box / 2,
            (y_pred_plume_corner[1] * height_image) - height_box / 2,
        ),
        width_box,
        height_box,
        edgecolor="orange",
        facecolor="none",
        linewidth=2,
    )
    rect_true = plt.Rectangle(
        (
            y_true_plume_corner[0] * width_image - width_box / 2,
            (y_true_plume_corner[1] * height_image) - height_box / 2,
        ),
        width_box,
        height_box,
        edgecolor="red",
        facecolor="none",
        linewidth=2,
    )
    ax1.add_patch(rect_pred)
    ax1.add_patch(rect_true)

    ax2.set_title("No Plume")
    ax2.tick_params("y", labelleft=False)
    ax2.tick_params("x", labelbottom=False)
    ax2.imshow(im_no_plume, cmap="gray")
    """
    rect_pred = plt.Rectangle((y_pred_no_plume_corner[0]*width_image, (y_pred_no_plume_corner[1])*height_image), width_box*width_image/100, height_box*height_image/100, edgecolor='orange',
   facecolor='none', linewidth=2)
    rect_true = plt.Rectangle((y_true_no_plume_corner[0]*width_image, (y_true_no_plume_corner[1])*height_image), width_box*width_image/100, height_box*height_image/100, edgecolor='red',
   facecolor='none', linewidth=2)
    ax2.add_patch(rect_pred)
    ax2.add_patch(rect_true)
    """


bb(trained_model, train_valid_data)


def iou(y_true_center, y_pred_center, h=0.2, w=0.2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    ----------
    y_true_center : array_like
        Center coordinates of the true bounding box.
    y_pred_center : array_like
        Center coordinates of the predicted bounding box.
    h : float, optional
        Height of the bounding box. Defaults to 0.2.
    w : float, optional
        Width of the bounding box. Defaults to 0.2.

    Returns:
    -------
    float
        The IoU value between the two bounding boxes.
    """
    xt1 = y_true_center[0] - w / 2
    yt1 = y_true_center[1] + h / 2
    xt2 = xt1 + w
    yt2 = yt1 - h

    xp1 = y_pred_center[0] - w / 2
    yp1 = y_pred_center[1] + h / 2
    xp2 = xp1 + w
    yp2 = yp1 - h

    ix1 = max(xt1, xp1)
    iy1 = min(yt1, yp1)
    ix2 = min(xt2, xp2)
    iy2 = max(yt2, yp2)

    aux1 = max((ix2 - ix1), 0)
    aux2 = max((iy1 - iy2), 0)
    intersection = aux1 * aux2
    union = h * w * 2 - intersection

    return intersection / union


def avg_iou(y_true, y_pred):
    """
    Computes the average Intersection over Union (IoU) for a set of bounding boxes.

    Parameters:
    ----------
    y_true : array_like
        Array of center coordinates of the true bounding boxes.
    y_pred : array_like
        Array of center coordinates of the predicted bounding boxes.

    Returns:
    -------
    float
        The average IoU over all provided bounding boxes.
    """
    total = 0
    for entry_y_true, entry_y_pred in zip(y_true, y_pred):
        total += iou(entry_y_true, entry_y_pred)
    return total / y_true.shape[0]


def one_execution_time(model, x_valid):
    """
    Measures the execution time for one prediction using the model.

    Parameters:
    ----------
    model : keras.Model
        The trained model to be evaluated.
    x_valid : ndarray
        The validation dataset.

    Returns:
    -------
    float
        The execution time for a single prediction, in milliseconds.
    """
    random_index = np.random.randint(0, x_valid.shape[0])
    x = x_valid[random_index, :, :]
    x = x.reshape([1, *x.shape])
    start = time.time()
    y = model.predict(x, verbose=0)
    end = time.time()
    return np.round_((end - start) * 1000, decimals=2)


def execution_time(model, x_valid, iterations=100):
    """
    Calculates the average execution time for predictions over multiple iterations.

    Parameters:
    ----------
    model : keras.Model
        The trained model to be evaluated.
    x_valid : ndarray
        The validation dataset.
    iterations : int, optional
        The number of iterations over which to average the execution time. Defaults to 100.

    Returns:
    -------
    float
        The average execution time for the given number of iterations, in milliseconds.
    """
    avg = 0
    for _ in range(iterations):
        avg += one_execution_time(model, x_valid)
    return np.round_(avg / iterations, decimals=2)


test_model(trained_model, train_valid_data)
