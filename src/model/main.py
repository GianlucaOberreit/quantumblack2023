# -*- coding: utf-8 -*-
from tensorflow import keras
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import Recall, Precision
from keras.layers import concatenate
from keras import Input

import numpy as np
import random as rn
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Recall, Precision
import time
from PIL import Image
from collections import namedtuple
import rasterio
import cv2

@tf.function
def giou_loss_with_center(y_true_center, y_pred_center):
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
    intersection = tf.multiply(tf.maximum((ix2 - ix1), zero), tf.maximum((iy1 - iy2), zero))
    union = aux - intersection
    iou = tf.math.divide(intersection, union)

    cw = tf.maximum(xc2, xp2) - tf.minimum(xc1, xp1)  # smallest enclosing box width
    ch = tf.maximum(yc1, yp1) - tf.minimum(yc2, yp2)  # smallest enclosing box height
    c_area = tf.multiply(cw, ch) + eps  # smallest enclosing box area
    giou = iou - tf.divide(c_area - union, c_area)  # GIoU
    return 1 - giou

def giou_loss(y_true, y_pred):
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

from google.colab import drive
!rm -rf '/content/McKinsey_Hackathon'
!ls
drive.mount('/content/drive/', force_remount=True)

!cp -r '/content/drive/MyDrive/McKinsey_Hackathon/' '/content/'
!ls

def get_datasets(datasets_folder):

    # Getting train-validation data
    x_train = np.load(os.path.join(datasets_folder, 'x_train.npy'))
    y_labels_train = np.load(os.path.join(datasets_folder, 'y_labels_train.npy'))
    y_boxes_train = np.load(os.path.join(datasets_folder, 'y_boxes_train.npy'))

    x_valid= np.load(os.path.join(datasets_folder, 'x_valid.npy'))
    y_labels_valid = np.load(os.path.join(datasets_folder, 'y_labels_valid.npy'))
    y_boxes_valid = np.load(os.path.join(datasets_folder, 'y_boxes_valid.npy'))

    rgb_x_train = np.squeeze(x_train)
    rgb_x_train = np.repeat(rgb_x_train[..., np.newaxis], 3, -1)
    rgb_x_valid = np.squeeze(x_valid)
    rgb_x_valid = np.repeat(rgb_x_valid[..., np.newaxis], 3, -1)

    train_valid_data = {
    'x_train': x_train,
    'rgb_x_train': rgb_x_train,
    'y_labels_train': y_labels_train,
    'y_boxes_train': y_boxes_train,
    'x_valid': x_valid,
    'rgb_x_valid': rgb_x_valid,
    'y_labels_valid': y_labels_valid,
    'y_boxes_valid': y_boxes_valid,
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
  image_width = 64
  image_heigth = 64
  csv_file_path = datasets_folder + 'metadata.csv'
  selected_columns = ['path', 'plume', 'coord_x', 'coord_y']
  df = pd.read_csv(csv_file_path, usecols=selected_columns)
  size = df.shape[0]
  x = np.empty((8*size, image_width, image_heigth))
  y = np.empty(8*size)
  y_boxes = np.empty((8*size, 2))
  for index, row in df.iterrows():
      photo = rasterio.open(datasets_folder + row['path']+'.tif')
      data = photo.read(1)
      scaled_data = data / 2**16
      print(scaled_data)
      x[8*index]=scaled_data
      plume = 1 if row['plume'] == 'yes' else 0
      for i in range(8):
        y[8*index+i]=int(plume)
      y_boxes[8*index]=np.array([row['coord_x']/image_width, row['coord_y']/image_heigth])

      x_tmp = x[8*index]
      y_boxes_tmp = y_boxes[8*index]
      x_f, y_boxes_f = flip(x_tmp, y_boxes_tmp)
      x[8*index+1] = x_f
      y_boxes[8*index+1] = y_boxes_f

      for i in range(1,4):
        x_tmp, y_boxes_tmp = rotate(x_tmp, y_boxes_tmp)
        x[8*index+2*i] = x_tmp
        y_boxes[8*index+2*i] = y_boxes_tmp
        x_f, y_boxes_f = flip(x_tmp, y_boxes_tmp)
        x[8*index+2*i+1] = x_f
        y_boxes[8*index+2*i+1] = y_boxes_f
  datanames = ["x_train", "x_valid", "y_labels_train", "y_labels_valid", "y_boxes_train", "y_boxes_valid"]
  data = train_test_split(x, y, y_boxes, test_size=0.2)
  for index, name in enumerate(datanames):
      np.save(datasets_folder + f'processed_data/{name}.npy', data[index])


def rotate(x, y_boxes):
  width, height = x.shape
  x_rotated = np.rot90(x, k=1)
  y_boxes_rotated = np.array([width - y_boxes[1], y_boxes[0]])
  return x_rotated, y_boxes_rotated

def flip(x, y_boxes):
  width, height = x.shape
  x_flipped = np.fliplr(x)
  y_boxes_flipped = np.array([width - y_boxes[0], y_boxes[1]])
  return x_flipped, y_boxes_flipped

organize_data('/content/McKinsey_Hackathon/')
train_valid_data = get_datasets('/content/McKinsey_Hackathon/processed_data')

, callbacks
from tensorflow.keras import regularizers

def define_model(name, input_shape, seed = 31415):

    # Setting seed
    rn.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Getting the parameters
    filters = [8,16,32]
    filters_dim = [3,3,5]
    max_pool_dim = [2,2,2]
    dropout = 0.1
    classification_layers = [16,8]
    localization_layers = [32,16]

    # Backbone model
    model = models.Sequential()
    model.add(layers.Conv2D(filters[0], filters_dim[0], activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.MaxPooling2D(max_pool_dim[0]))
    if dropout:
        model.add(layers.Dropout(dropout))

    for num_filter, conv_dim, max_pool in zip(filters[1:], filters_dim[1:], max_pool_dim[1:]):
        model.add(layers.Conv2D(num_filter, conv_dim, activation='relu'))
        model.add(layers.MaxPooling2D(max_pool))
        if dropout:
            model.add(layers.Dropout(dropout))

    model.add(layers.GlobalAveragePooling2D())
    flattened_output = model.output

    # Final layers for classification
    class_prediction = layers.Dense(classification_layers[0], activation="relu", kernel_regularizer=regularizers.l2(0.01))(flattened_output)
    class_prediction = layers.Dropout(dropout)(class_prediction)
    for classification_layer in classification_layers[1:]:
        class_prediction = layers.Dense(classification_layer, activation="relu")(class_prediction)
        class_prediction = layers.Dropout(dropout)(class_prediction)
    class_prediction = layers.Dense(1, activation='sigmoid', name="class_output")(class_prediction)

    # Final layers for localization
    box_output = layers.Dense(localization_layers[0], activation="relu")(flattened_output)
    box_output = layers.Dropout(dropout)(box_output)
    for localization_layer in localization_layers[1:]:
        box_output = layers.Dense(localization_layer, activation="relu")(box_output)
        box_output = layers.Dropout(dropout)(box_output)
    box_predictions = layers.Dense(2, activation='sigmoid', name= "box_output")(box_output)

    complete_model = keras.Model(name = name, inputs = model.input, outputs= [box_predictions, class_prediction])
    return complete_model

def compile_model (name, input_shape, box_loss = "mean_squared_error"):

    # Creating model
    model = define_model(name = name, input_shape = input_shape)

    # Compiling the model
    losses = { "box_output": box_loss,
               "class_output": "binary_crossentropy" }
    loss_weights = { "box_output": 0.0,
                     "class_output": 1.0 }
    metrics = {'box_output':  'mse',
               'class_output': ['accuracy', Precision(), Recall()]}

    learning_rate = 1e-4
    # opt = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = 0.9)
    opt = tf.keras.optimizers.Adam()

    model.compile(optimizer = opt, loss = losses, loss_weights = loss_weights, metrics = metrics)
    return model

def train_model (model, batch_size, epochs, train_valid_data):


    # Getting the train-validation data
    x_train = train_valid_data['x_train']
    y_labels_train = train_valid_data['y_labels_train']
    y_boxes_train = train_valid_data['y_boxes_train']
    x_valid = train_valid_data['x_valid']
    y_labels_valid = train_valid_data['y_labels_valid']
    y_boxes_valid = train_valid_data['y_boxes_valid']
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='', patience=7)
    # Training the model
    history = model.fit(x = x_train, verbose = 2,
                    y= {
                        "box_output": y_boxes_train,
                        "class_output": y_labels_train
                        },
                    validation_data=(
                        x_valid,
                        {
                          "box_output": y_boxes_valid,
                          "class_output": y_labels_valid
                          }),
                    batch_size = batch_size, epochs = epochs,  sample_weight={
                              "box_output": y_labels_train,
                              "class_output": np.ones(y_labels_train.shape)
                            },
                    callbacks=[early_stopping])
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Create a plot of training and validation loss
    plt.plot(range(1, len(training_loss) + 1), training_loss, label='Training Loss')
    plt.plot(range(1, len(validation_loss) + 1), validation_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.yscale('log')
    plt.xlim(0,epochs)
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return model
    return model

def test_model(model, train_valid_data):
    x_valid = train_valid_data['x_valid']
    y_labels_valid = train_valid_data['y_labels_valid']
    y_boxes_valid = train_valid_data['y_boxes_valid']

    # Testing the model on different metrics
    y_valid_pred = model.predict(x_valid, verbose = 0)
    y_valid_pred_corner = y_valid_pred[0]
    y_valid_pred_label = np.round(y_valid_pred[1])


    run_results = {'accuracy': accuracy_score(y_true=y_labels_valid, y_pred=y_valid_pred_label),
                  'f1': f1_score(y_true=y_labels_valid, y_pred=y_valid_pred_label),
                  'precision': precision_score(y_true=y_labels_valid, y_pred=y_valid_pred_label),
                  'recall': recall_score(y_true=y_labels_valid, y_pred=y_valid_pred_label),
                  'f1_inv': f1_score(y_true=y_labels_valid, y_pred=y_valid_pred_label, pos_label=0),
                  'precision_inv': precision_score(y_true=y_labels_valid, y_pred=y_valid_pred_label, pos_label=0),
                  'recall_inv': recall_score(y_true=y_labels_valid, y_pred=y_valid_pred_label, pos_label=0),
                  'avg_iou': avg_iou(y_true=y_boxes_valid, y_pred=y_valid_pred_corner),
                  'execution_time': execution_time(model, x_valid),
                  'num_parameters': model.count_params(),
                  'model': model}
    return run_results

def compile_and_train (name,  input_shape,  batch_size, epochs, train_valid_data, box_loss):
    model = compile_model (name, input_shape, box_loss = box_loss)
    trained_model = train_model (model,  batch_size, epochs, train_valid_data)
    return trained_model

input_shape = (64,64,1)
batch_size = 32
epochs = 300
my_loss = tf.keras.losses.MeanSquaredError()
folder = '/content/McKinsey_Hackathon/models'

trained_model = compile_and_train (f"Ipop3", input_shape, batch_size, epochs, train_valid_data, my_loss)

# print(trained_model.summary())
trained_model.save(os.path.join(folder, trained_model.name + '.h5'))

def bb(model, test_data, height_image = 64, width_image = 64, height_box = 20, width_box = 20):

    x_test = test_data['x_valid']
    y_labels_test = test_data['y_labels_valid']
    y_boxes_test = test_data['y_boxes_valid']

    im_plume_index = np.where(y_labels_test==1)[0][1]
    im_plume = x_test[im_plume_index, :, :]
    y_true_plume_corner = y_boxes_test[im_plume_index]

    im_no_plume_index = np.where(y_labels_test==0)[0][2]
    im_no_plume = x_test[im_no_plume_index, :, :]
    y_true_no_plume_corner = y_boxes_test[im_no_plume_index]

    y_pred_plume = model.predict(im_plume.reshape([1, *im_plume.shape]))
    y_pred_no_plume = model.predict(im_no_plume.reshape([1, *im_no_plume.shape]))

    y_pred_plume_corner = y_pred_plume[0][0]
    y_pred_no_plume_corner = y_pred_no_plume[0][0]

    print(f'Confidence in image with plume: {y_pred_plume[1][0][0]}')
    print(f'Confidence in image without plume: {y_pred_no_plume[1][0][0]}')
    print(f'Predicted plume coordinates: {y_pred_plume_corner}')

    plt.figure(figsize=(6,6))
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)

    ax1.set_title('With Plume')
    ax1.tick_params('y', labelleft=False)
    ax1.tick_params('x', labelbottom=False)
    ax1.imshow(im_plume, cmap = 'gray')
    rect_pred = plt.Rectangle((y_pred_plume_corner[0]*width_image - width_box/2, (y_pred_plume_corner[1]*height_image)- height_box/2) , width_box, height_box, edgecolor='orange',
   facecolor='none', linewidth=2)
    rect_true = plt.Rectangle((y_true_plume_corner[0]*width_image - width_box/2, (y_true_plume_corner[1]*height_image)- height_box/2), width_box, height_box, edgecolor='red',
   facecolor='none', linewidth=2)
    ax1.add_patch(rect_pred)
    ax1.add_patch(rect_true)

    ax2.set_title('No Plume')
    ax2.tick_params('y', labelleft=False)
    ax2.tick_params('x', labelbottom=False)
    ax2.imshow(im_no_plume, cmap = 'gray')
    """
    rect_pred = plt.Rectangle((y_pred_no_plume_corner[0]*width_image, (y_pred_no_plume_corner[1])*height_image), width_box*width_image/100, height_box*height_image/100, edgecolor='orange',
   facecolor='none', linewidth=2)
    rect_true = plt.Rectangle((y_true_no_plume_corner[0]*width_image, (y_true_no_plume_corner[1])*height_image), width_box*width_image/100, height_box*height_image/100, edgecolor='red',
   facecolor='none', linewidth=2)
    ax2.add_patch(rect_pred)
    ax2.add_patch(rect_true)
    """

bb(trained_model,train_valid_data)

def iou(y_true_center, y_pred_center, h=0.2, w=0.2):
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
    total = 0
    for entry_y_true, entry_y_pred in zip(y_true, y_pred):
        total += iou(entry_y_true, entry_y_pred)
    return total/y_true.shape[0]

def one_execution_time(model, x_valid):
    random_index = np.random.randint(0, x_valid.shape[0])
    x = x_valid[random_index, :, :]
    x = x.reshape([1, *x.shape])
    start = time.time()
    y = model.predict(x, verbose = 0)
    end = time.time()
    return  np.round_((end-start)*1000, decimals = 2)

def execution_time(model, x_valid, iterations = 100):
    avg = 0
    for _ in range(iterations):
        avg += one_execution_time(model, x_valid)
    return np.round_(avg/iterations, decimals = 2)

test_model(trained_model, train_valid_data)

def testing(epoch_values, iterations):
  scores = {epoch: 0 for epoch in epoch_values}
  for _ in range(iterations):
    organize_data('/content/drive/MyDrive/McKinsey_Hackathon/')
    train_data = get_datasets('/content/drive/MyDrive/McKinsey_Hackathon/processed_data')
    for epoch in epoch_values:
      trained_model = compile_and_train_int_resnet (f"{epoch}epochs", input_shape, batch_size, epoch, train_data, my_loss)
      results = test_resnet_model(trained_model, train_data)
      score = (results['accuracy'] + results['f1'] + results['precision'] + results['recall'] + results['f1_inv'] + results['precision_inv'] + results['recall_inv'])/7
      scores[epoch] += score/iterations
  return scores

testing([10,20,30,40,50, 70], 5)

def giou_loss_CL(y_true_corner, y_pred_corner):
    eps=1e-7
    h = 25/100
    w = 45/100

    xt1 = y_true_corner[:, 0]
    yt1 = y_true_corner[:, 1]
    xt2 = xt1 + w
    yt2 = yt1 - h

    xp1 = y_pred_corner[:, 0]
    yp1 = y_pred_corner[:, 1]
    xp2 = xp1 + w
    yp2 = yp1 - h

    ix1 = tf.maximum(xt1,xp1)
    iy1 = tf.minimum(yt1,yp1)
    ix2 = tf.minimum(xt2,xp2)
    iy2 = tf.maximum(yt2,yp2)

    zero = tf.zeros(tf.shape(ix1))
    aux = 2*h*w*tf.ones(tf.shape(ix1))
    intersection = tf.multiply(tf.maximum((ix2 - ix1), zero), tf.maximum((iy1-iy2), zero))
    union = aux - intersection
    iou = tf.math.divide(intersection, union)

    cw = tf.maximum(xt2, xp2) - tf.minimum(xt1, xp1)  # smallest enclosing box width
    ch = tf.maximum(yt1, yp1) - tf.minimum(yt2, yp2)  # smallest enclosing box height
    c_area = tf.multiply(cw, ch) + eps  # smallest enclosing box area
    giou = iou - tf.divide(c_area - union, c_area)  # GIoU
    return 1 - giou

tf.keras.utils.get_custom_objects()['giou_loss_CL'] = giou_loss_CL

model_to_evaluate = tf.keras.models.load_model("/content/drive/MyDrive/McKinsey_Hackathon/models/1000epochs.keras",custom_objects={'giou_loss_CL': giou_loss_CL })
