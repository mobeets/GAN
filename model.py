import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

def sample_noise(shape, noise_type):
    if noise_type == 'uniform':
        return np.random.uniform(-1, 1, shape)
    elif noise_type == 'normal':
        return np.random.normal(0, 1, shape)

def build_generator(batch_size, latent_dim, intermediate_dims, original_dim, optimizer):
    """
    given some random noise (Z), return a generated sample (Yh)
    """
    Z = Input(batch_shape=(batch_size, latent_dim), name='Z')
    H = Z
    intermediate_dims = intermediate_dims if intermediate_dims is not None else []
    for i, intermediate_dim in enumerate(intermediate_dims):
        H = Dense(intermediate_dim, activation='relu', name='hg'+str(i))(H)
    Yh = Dense(original_dim, activation='tanh', name='Yh')(H)

    mdl = Model(Z, Yh)
    mdl.compile(optimizer=optimizer, loss='binary_crossentropy')
    return mdl

def build_discriminator(batch_size, original_dim, intermediate_dims, optimizer):
    """
    given an image (Y), return the probability that this sample is fake (P)
    """
    Y = Input(batch_shape=(batch_size, original_dim), name='Y')
    H = Y
    intermediate_dims = intermediate_dims if intermediate_dims is not None else []
    for i, intermediate_dim in enumerate(intermediate_dims):
        H = Dense(intermediate_dim, activation='relu', name='hg'+str(i))(H)
    P = Dense(1, activation='sigmoid', name='P')(H)

    mdl = Model(Y, P)
    mdl.compile(optimizer=optimizer, loss='binary_crossentropy')
    return mdl

def build_combined(batch_size, latent_dim, gen_model, disc_model, optimizer):
    Z = Input(batch_shape=(batch_size, latent_dim), name='Z')
    Yh = gen_model(Z)
    disc_model.trainable = False
    P = disc_model(Yh)
    mdl = Model(Z, P)
    mdl.compile(optimizer=optimizer, loss='binary_crossentropy')
    return mdl
