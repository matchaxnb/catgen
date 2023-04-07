from keras.utils import Sequence
import cv2
import os
import pathlib
import pickle
import numpy as np



class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        return batch_x, batch_y


def build_training_data(r_w=256, r_h=256):
    """build training data for our stuff"""
    tgt_path = f"training-{r_w}-{r_h}.pickle"
    if os.path.exists(tgt_path):
        print("load training data from pickled output")
        images, labels = pickle.load(open(tgt_path, "rb"))
        print(images.shape, labels.shape)
        return images, labels

    catpath = pathlib.Path("./dataset/cats")
    otherpath = pathlib.Path("./dataset/others")

    dirs = (catpath, otherpath)
    scores = (1.0, 0.0)
    images = []
    labels = []
    for i, impath in enumerate(dirs):
        score = scores[i]

        files = os.listdir(impath)
        for image in files:
            if not image.endswith("jpg"):
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
    with open(tgt_path, "wb") as fh:
        pickle.dump(output, fh)
    return output


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