# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 11:27:43 2021

@author: Dilumika
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image
import os

#####################################################


device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

style_transfer_url = os.path.join('style_images', '#style image')
base_url = os.path.join('base_images', '#base image')

a = plt.imread(base_url)
b = plt.imread(style_transfer_url)

f, axarr = plt.subplots(1,2, figsize=(15,15))
axarr[0].imshow(a)
axarr[1].imshow(b)
plt.show()


width, height = image.load_img(base_url).size
img_rows = 400
img_cols = int(width * img_rows / height)

model = vgg19.VGG19(weights="imagenet", include_top=False)
model.summary()

outputs_dict= dict([(layer.name, layer.output) for layer in model.layers])
feature_extractor = Model(inputs=model.inputs, outputs=outputs_dict)


style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

content_layer = "block5_conv2"

content_weight = 2.5e-8
style_weight = 1e-6

#####################################################

def gram_matrix(x):
    x = tf.transpose(x, (2 ,0, 1))
    feat = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(feat, tf.transpose(feat))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_rows * img_cols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))


def loss_function(combination_image, base_image, style_reference_image):

    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )

    features = feature_extractor(input_tensor)
    loss = tf.zeros(shape=())

    layer_features = features[content_layer]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]

    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )

    for layer_name in style_layers:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layers)) * sl

    return loss


@tf.function
def compute_loss(combination_image, base_image, style_image):
    with tf.GradientTape() as tape:
        loss = loss_function(combination_image, base_image, style_image)
    
    grads = tape.gradient(loss, combination_image)
    return loss, grads



def preprocess_image(image_path, img_rows, img_cols):
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_rows, img_cols))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    
    return tf.convert_to_tensor(img)

def deprocess_image(x, img_rows, img_cols):

    x = x.reshape((img_rows, img_cols, 3))

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")

    return x




def TransferImage():
    
    optimizer = SGD(
        tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
        )
    )
    
    base_image = preprocess_image(base_url, img_rows, img_cols)
    style_image = preprocess_image(style_transfer_url, img_rows, img_cols)
    combination_image = tf.Variable(preprocess_image(base_url, img_rows, img_cols))
    
    iterations = 1000
    
    for i in range(1, iterations + 1):
        loss, grads = compute_loss(
            combination_image, base_image, style_image
        )
        optimizer.apply_gradients([(grads, combination_image)])
        if i % 10 == 0:
            print("Iteration %d: loss=%.2f" % (i, loss))
        img = deprocess_image(combination_image.numpy(), img_rows, img_cols)
    image.save_img(os.path.join("combined_images","img2.png"), img)
        
        
TransferImage()