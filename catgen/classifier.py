#!python

import cv2
import os
import pathlib
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers


class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


def build_training_data(r_w=256, r_h=256):
    """build training data for our stuff"""
    if (os.path.exists('training.pickle')):
        print("load training data from pickled output")
        images, labels = pickle.load(open('training.pickle', 'rb'))
        print(images.shape, labels.shape)
        return images, labels
    
    catpath = pathlib.Path('./dataset/cats')
    otherpath = pathlib.Path('./dataset/others')

    dirs = (catpath, otherpath)
    scores = (1.0, 0.0)
    images = []
    labels = []
    for i, impath in enumerate(dirs):
        score = scores[i]
        files = os.listdir(impath)
        for image in files:
            if not image.endswith('jpg'):
                continue
            totalpath = impath.joinpath(image)
            img = cv2.imread(str(impath.joinpath(image)))
            if img is None:
                continue
            img = cv2.resize(img, (r_w, r_h)).astype(np.float32)
            img /= 255.0
            images.append(img)
            labels.append(score)
    output = (np.array(images), np.array(labels))
    with open('training.pickle', 'wb') as fh:
        pickle.dump(output, fh)
    return output

def build_discriminator_network(r_w=256, r_h=256, f_1=32, k_1=(3,3), a_h='relu', a_d='relu', n_layers=3, dropout_base=0.1, dropout_factor=1.125) -> tf.keras.Sequential:
    """build a discriminator network.
    
    hyperparameters:
        r_w: resize width
        r_h: resize height
        f_1: number of filters for first hidden layer
        k_1: kernel size for activation function
        a_h: activation function for hidden layers
        a_d: activation function for dense coalescing layer
        n_layers: number of hidden layers, each doubling the number of filters
    """
    model = tf.keras.Sequential()
    # input 
    model.add(layers.Input(shape=(r_w,r_h,3)))
    dropout = dropout_base
    num_neurons = 1
    for m in range(1, n_layers+1):
        num_neurons = f_1 * (2**(m-1))
        model.add(layers.Conv2D(num_neurons, k_1, activation=a_h))
        model.add(layers.Dropout(dropout))
        dropout *= dropout_factor
    model.add(layers.Flatten())
    # now a dense layer with as many neurons as the last hidden layer to recombine the features
    model.add(layers.Dense(num_neurons, activation=a_d))
    print(f"first dense layer has {num_neurons} neurons")
    # now our answer layer, this one is a sigmoid
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def split_data_for_tests(images, labels):
    assert len(images) == len(labels)
    permutation = np.random.permutation(len(images))
    return images[permutation], labels[permutation]

def get_sets(images, labels):
    images, labels = split_data_for_tests(images, labels)
    total_pop = len(images)
    validation_ratio = 0.3
    breakpoin = int(total_pop * validation_ratio)
    val_images, val_labels = images[0:breakpoin], labels[0:breakpoin]
    train_images, train_labels = images[breakpoin:], labels[breakpoin:]
    return train_images, train_labels, (val_images, val_labels)

_the_model = None
def train_model(epochs=20, batch_size=8, layers=3, dropout_base=0.1, dropout_factor=1.125, filters_base=16, retrain=False):
    global _the_model
    tgt_file = f'catgen_d{dropout_base}f{dropout_factor}_F{filters_base}_l{layers}_e{epochs}.h5'
    if _the_model is None and os.path.exists(tgt_file) and not retrain:
        _the_model = tf.keras.models.load_model(tgt_file)
    if _the_model is not None and not retrain:
        return _the_model
    with tf.device('/gpu:0'):
        dn = build_discriminator_network(n_layers=layers, dropout_base=dropout_base, dropout_factor=dropout_factor, f_1=filters_base)
    x_train, y_train, validation_data = get_sets(*build_training_data())
    train_gen = DataGenerator(x_train, y_train, batch_size)
    test_gen = DataGenerator(*validation_data, batch_size)
    dn.fit(train_gen, epochs=epochs, validation_data=test_gen, shuffle=True)
    dn.save(tgt_file)
    _the_model = dn
    return dn

def test_image(imgpath, r_w=256, r_h=256):
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (r_w, r_h))
    image = np.array(img.astype(np.float32))
    image = tf.expand_dims(image, 0)
    image /= 255.0
    model = train_model()
    prediction = model.predict(image)
    return prediction


def cert_to_str(c):
    if (c < 0.1):
        return f"not a cat {c}"
    elif (c > 0.95):
        return "a cat (> 0.95)"
    elif (c > 0.9):
        return "a cat (> 0.9)"
    elif (c > 0.8):
        return "a cat? (> 0.8)"
    else:
        return f"unsure [{c}]"

if __name__ == '__main__':
    model = train_model()
    import sys
    print("Model architecture:")
    model.summary()
    if len(sys.argv) > 1:
        for item in sys.argv[1:]:
            print(f"prediction:{item}:",cert_to_str(test_image(item)))
    