import os
import tensorflow as tf

from catgen.utils import DataGenerator, build_training_data, get_sets
from keras_cv_attention_models import efficientnet
from tensorflow.keras import layers

_the_efnet_model = None


def build_efficientnetv2_discriminator_model(
    input_shape=(256,256,3)
):
    model = efficientnet.EfficientNetV2S(pretrained="imagenet", input_shape=input_shape, num_classes=0)
    # freeze the weights of the base model. we don't want to alter them.
    model.trainable = False
    # now add the output layers.
    # this flattens each dimension to 1 number
    x = layers.GlobalAveragePooling2D(name="output_pooling2d")(model.output)
    x = layers.BatchNormalization()(x)
    # now a dense layer takes the output of the 2D GAP and reduces it to 1 sigmoid scalar
    outputs = layers.Dense(1, activation="sigmoid", name="is_a_cat")(x)
    # now we build our model that embeds the other one and specializes
    our_model = tf.keras.Model(model.inputs, outputs, name="ENCatDiscriminator")
    our_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # and boom we're done
    return our_model


def train_efficientnetv2_discriminator_model(
    epochs=3,
    batch_size=16,
    input_shape=(256,256,3),
    retrain=False
):
    global _the_efnet_model
    tgt_file = f"catgen_efnet_E{epochs}_{input_shape[0]}x{input_shape[1]}.h5"
    if _the_efnet_model is None and os.path.exists(tgt_file) and not retrain:
        _the_efnet_model = tf.keras.models.load_model(tgt_file)
    if _the_efnet_model is not None and not retrain:
        _the_efnet_model.summary()
        return _the_efnet_model
    with tf.device("/gpu:0"):
        dn = build_efficientnetv2_discriminator_model(
            input_shape=input_shape,
        )
    dn.summary()
    x_train, y_train, validation_data = get_sets(*build_training_data(r_w=input_shape[0], r_h=input_shape[1]))
    train_gen = DataGenerator(x_train, y_train, batch_size)
    test_gen = DataGenerator(*validation_data, batch_size)
    dn.fit(train_gen, epochs=epochs, validation_data=test_gen, shuffle=True)
    dn.save(tgt_file)
    _the_efnet_model = dn
    return _the_efnet_model