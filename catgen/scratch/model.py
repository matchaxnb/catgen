import os
import tensorflow as tf


from math import ceil

from catgen.utils import get_sets
from catgen.utils import DataGenerator, build_training_data
from keras import layers

_the_model = None

def build_from_scratch_model(
    r_w=256,
    r_h=256,
    f_1=32,
    k_1=(3, 3),
    a_h="relu",
    a_d="relu",
    n_hidden_processors=3,
    layer_growth_factor=2,
    extra_dense_layers=2,
) -> tf.keras.Sequential:
    """build a discriminator network.
    hyperparameters:
        r_w: resize width
        r_h: resize height
        f_1: number of filters for first hidden layer
        k_1: kernel size for activation function
        a_h: activation function for hidden layers
        a_d: activation function for dense coalescing layer
        n_hidden_processors: number of hidden layer sandwiches, each doubling the number of filters
    """
    model = tf.keras.Sequential()
    # input
    model.add(layers.Input(shape=(r_w, r_h, 3)))
    # now let's build our vision hidden layers
    # a common arch is a sandwich of
    # (conv{1,2} maxpooling) repeated
    # then 2/3 dense layers
    num_neurons = f_1
    image_width = r_w
    for m in range(1, n_hidden_processors + 1):
        # the top convolution layer in the sandwich starts with f_1 filters (default 32)
        # then we do a 2D max pooling to reduce the spatial dimensions 2-fold.
        # our input tensor was (1, 256, 256), after the convolutional layer is applied it
        # is now (32, 256, 256) ( x 32 ), and after we MaxPool it is (32, 128, 128) ( / 4 ).
        # The size of data is multiplied by 8 at this stage.
        # therefore: we will multiply the number of neurons by 2 in the convolutional layer of
        # each new sandwich, to be able to sample more of the output.
        print(f"Adding convolution layer with {num_neurons} filters and kernel {k_1}")
        model.add(layers.Conv2D(num_neurons, k_1, activation=a_h, padding="same"))
        model.add(layers.BatchNormalization())
        # let's add another here
        model.add(layers.Conv2D(num_neurons, k_1, activation=a_h, padding="same"))
        model.add(layers.BatchNormalization())
        # now let's reduce the spatial dimensions with some maxpooling

        if image_width > 32:
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.BatchNormalization())
            image_width /= 2
        # we now keep only the most significant data over windows of (2,2)
        # let's grow the number of neurons for next layer
        num_neurons = ceil(num_neurons * layer_growth_factor)
    # let's get num_neurons back to where it should be
    num_neurons = ceil(num_neurons / layer_growth_factor)
    # we flatten the model, it becomes 1D so the neurons can abstract away from spatial considerations now
    # as we are doing categorization now that we've seen things
    model.add(layers.Flatten())
    # now a dense layer with as many neurons as the last hidden layer to recombine the features
    model.add(layers.Dense(num_neurons, activation=a_d))
    model.add(layers.BatchNormalization())
    print(f"first dense layer has {num_neurons} neurons")
    # let's add more to help seeing things better
    for i in range(0, extra_dense_layers):
        model.add(layers.Dense(num_neurons, activation=a_d))
        model.add(layers.BatchNormalization())

    # now our answer layer, this one is a sigmoid
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train_from_scratch_model(
    epochs=20,
    batch_size=8,
    filter_sandwiches=3,
    filters_base=64,
    r_w=256,
    r_h=256,
    extra_dense_layers=2,
    layer_growth_factor=2,
    retrain=False,
):
    global _the_model
    tgt_file = f"catgen_F{filters_base}_l{filter_sandwiches}g{layer_growth_factor}_d{extra_dense_layers}_e{epochs}_{r_w}x{r_h}.h5"
    if _the_model is None and os.path.exists(tgt_file) and not retrain:
        _the_model = tf.keras.models.load_model(tgt_file)
    if _the_model is not None and not retrain:
        _the_model.summary()
        return _the_model
    with tf.device("/gpu:0"):
        dn = build_from_scratch_model(
            r_w=r_w,
            r_h=r_h,
            n_hidden_processors=filter_sandwiches,
            f_1=filters_base,
            layer_growth_factor=layer_growth_factor,
        )
        dn.summary()

    x_train, y_train, validation_data = get_sets(*build_training_data(r_w=r_w, r_h=r_h))
    train_gen = DataGenerator(x_train, y_train, batch_size)
    test_gen = DataGenerator(*validation_data, batch_size)
    dn.fit(train_gen, epochs=epochs, validation_data=test_gen, shuffle=True)
    dn.save(tgt_file)
    _the_model = dn
    return dn