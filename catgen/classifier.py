#!python

import cv2
import numpy as np
import tensorflow as tf
from .efnet.model import train_efficientnetv2_discriminator_model
from .scratch.model import train_from_scratch_model
from keras.preprocessing.image import ImageDataGenerator




def test_image(model, imgpath, r_w=256, r_h=256):
    img = cv2.imread(imgpath)
    try:
        img = cv2.resize(img, (r_w, r_h))
    except:
        return -1
    image = np.array(img.astype(np.float32))
    image = tf.expand_dims(image, 0)
    image /= 255.0
    prediction = model.predict(image)
    return prediction


def cert_to_str(c):
    if c < 0:
        return "error"
    elif c < 0.1:
        return f"not a cat {c}"
    elif c > 0.95:
        return "a cat (> 0.95)"
    elif c > 0.9:
        return "a cat (> 0.9)"
    elif c > 0.8:
        return "a cat? (> 0.8)"
    else:
        return f"unsure [{c}]"


if __name__ == "__main__":
    import sys
    HP_RW = 256
    HP_RH = 256
    choice = sys.argv[1]
    if choice not in ('scratch', 'efnet'):
        print("wrong model choice, not one of [scratch|efnet]", file=sys.stderr)
        sys.exit(1)
    model = None
    if choice == 'scratch':
        model = train_from_scratch_model(r_w=HP_RW, r_h=HP_RH)
    else:
        model = train_efficientnetv2_discriminator_model(input_shape=(HP_RW, HP_RH, 3))
    

    if len(sys.argv) > 2:
        for item in sys.argv[2:]:
            print(
                f"prediction:{item}:",
                cert_to_str(test_image(model, item, r_w=HP_RW, r_h=HP_RH)),
            )







