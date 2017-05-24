from __future__ import print_function
import os.path
import pickle
import argparse
import numpy as np
from keras.utils.generic_utils import Progbar
from model import build_generator, build_discriminator, build_combined, sample_noise

from keras.datasets import mnist
from PIL import Image

def load_data(digit=None):
    # normalize between -1 and 1
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5

    # reshape from (n, 28,28) to (n, 28*28)
    nd = X_train.shape[1]*X_train.shape[2]
    X_train = X_train.reshape((len(X_train), nd))
    X_test = X_test.reshape((len(X_test), nd))
    
    # only fit a single digit
    if digit is not None:
        ix_tr = y_train == 2
        ix_te = y_test == 2
        return X_train[ix_tr], X_test[ix_te], y_train[ix_tr], y_test[ix_te]
    else:
        return X_train, X_test, y_train, y_test

def train_on_batch(Xtr, batch_index, batch_size, latent_size, noise_type, k, generator, discriminator, combined):

    # generate samples
    Zba = sample_noise((batch_size, latent_size), noise_type)
    Xba_hat = generator.predict(Zba, verbose=False)

    # train discriminator to tell apart true and fake samples
    if batch_index % k == 0:
        # only train discriminator every k batches
        Xba = Xtr[batch_index * batch_size:(batch_index + 1) * batch_size]
        X = np.concatenate((Xba, Xba_hat))
        y = np.array([1] * batch_size + [0] * batch_size)
        batch_disc_loss = discriminator.train_on_batch(X, y)
    else:
        batch_disc_loss = 0.0

    # train generator to fool discriminator
    #   n.b. want generator to get same # of samples as discriminator
    Zba = sample_noise((2*batch_size, latent_size), noise_type)
    yhope = np.ones(2 * batch_size) # hope they all look like true samples
    batch_gen_loss = combined.train_on_batch(Zba, yhope)

    return batch_disc_loss, batch_gen_loss

def save_image(X, nrows, fnm):
    img = (np.concatenate([r.reshape(-1, 28)for r in np.split(X, nrows)], axis=-1) * 127.5 + 127.5).astype(np.uint8)
    Image.fromarray(img).save(fnm)

def evaluate(Xte, batch_size, latent_size, noise_type, generator, discriminator, combined, fnm_samp):
    """
    same as train_on_batch only now we evaluate
    """
    ntest = Xte.shape[0]

    Zte = sample_noise((ntest, latent_size), noise_type)
    Xte_hat = generator.predict(Zte, verbose=False, batch_size=batch_size)

    X = np.concatenate((Xte, Xte_hat))
    y = np.array([1]*ntest + [0]*ntest)
    test_disc_loss = discriminator.evaluate(X, y, verbose=False, batch_size=2*batch_size)

    Zte = sample_noise((2*ntest, latent_size), noise_type)
    yhope = np.ones(2*ntest)
    test_gen_loss = combined.evaluate(Zte, yhope, verbose=False, batch_size=2*batch_size)

    save_image(Xte_hat[:100], 10, fnm_samp)

    return test_disc_loss, test_gen_loss

def get_filenames(model_dir, run_name):
    fnm_gen = os.path.join(model_dir, run_name + '_gen_{0:03d}.h5')
    fnm_disc = os.path.join(model_dir, run_name + '_disc_{0:03d}.h5')
    fnm_hist = os.path.join(model_dir, run_name + '_history.pickle')
    fnm_samp = os.path.join(model_dir, run_name + '_samp_{0:03d}.png')
    return fnm_gen, fnm_disc, fnm_hist, fnm_samp

def print_history(history):
    ROW_FMT = '{0:<22s} | {1:<4.2f}'# | {2:<15.2f} | {3:<5.2f}'
    print(ROW_FMT.format('generator (train)',
        history['train']['generator'][-1]))
    print(ROW_FMT.format('generator (test)',
        history['test']['generator'][-1]))
    print(ROW_FMT.format('discriminator (train)',
        history['train']['discriminator'][-1]))
    print(ROW_FMT.format('discriminator (test)',
        history['test']['discriminator'][-1]))
    return history

def update_history(history, train_loss, test_loss, do_print=True):
    history['train']['discriminator'].append(train_loss[0])
    history['train']['generator'].append(train_loss[1])
    history['test']['discriminator'].append(test_loss[0])
    history['test']['generator'].append(test_loss[1])
    if do_print:
        print_history(history)
    return history

def init_history():
    history = {}
    history['train'] = {'discriminator': [], 'generator': []}
    history['test'] = {'discriminator': [], 'generator': []}
    return history

def save_history(fnm, history):
    with open(fnm, 'w') as f:
        pickle.dump(history, f)

def train(run_name, digit, nepochs, batch_size, latent_dim, noise_type, k,
    optimizer, model_dir, gen_intermediate_dims=None,
    disc_intermediate_dims=None):

    # load data and params
    Xtr, Xte, _, _ = load_data(digit)
    Xte = Xte[:(Xte.shape[0]/batch_size)*batch_size] # correct for batch_size
    original_dim = Xtr.shape[-1]
    nbatches = int(Xtr.shape[0]/batch_size)
    fnm_gen, fnm_disc, fnm_hist, fnm_samp = get_filenames(model_dir, run_name)

    # load models
    generator = build_generator(batch_size, latent_dim,
        gen_intermediate_dims, original_dim, optimizer)
    discriminator = build_discriminator(2*batch_size, original_dim,
        disc_intermediate_dims, optimizer)
    combined = build_combined(2*batch_size, latent_dim,
        generator, discriminator, optimizer)

    # train and test
    history = init_history()
    for i in xrange(nepochs):
        print('Epoch {} of {}'.format(i+1, nepochs))
        progress_bar = Progbar(target=nbatches)

        # train on mini-batches of train data
        epoch_losses = []
        for j in xrange(nbatches):
            progress_bar.update(j)
            batch_loss = train_on_batch(Xtr, j, batch_size, latent_dim, noise_type, k, generator, discriminator, combined)
            epoch_losses.append(batch_loss)

        # evaluate on test data
        print('\nTesting epoch {}:'.format(i + 1))
        test_loss = evaluate(Xte, batch_size, latent_dim, noise_type, generator, discriminator, combined, fnm_samp.format(i+1))
        train_loss = np.mean(np.array(epoch_losses), axis=0)
        history = update_history(history, train_loss, test_loss, do_print=True)

        # save weights
        generator.save_weights(fnm_gen.format(i), True)
        discriminator.save_weights(fnm_disc.format(i), True)        

    # save history    
    save_history(fnm_hist, history)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
                help='tag for current run')
    parser.add_argument('--num_epochs', type=int, default=200,
                help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=10,
                help='batch size')
    parser.add_argument('--latent_dim', type=int, default=2,
                help='latent dim')
    parser.add_argument('--digit', type=int, default=2,
                help='which digit to train on (0-9)')
    parser.add_argument('-k', type=int, default=2,
                help='train discriminator every k batches')
    parser.add_argument('--gen_intermediate_dim', type=int, default=4,
                help='generator intemediate dim')
    parser.add_argument('--disc_intermediate_dim', type=int, default=4,
                help='discriminator intemediate dim')
    parser.add_argument('--optimizer', type=str, default='adam',
                help='optimizer name') # 'rmsprop'
    parser.add_argument('--noise', type=str, default='normal',
                choices=['normal', 'uniform'],
                help='noise distribution') # 'rmsprop'
    parser.add_argument('--model_dir', type=str,
                default='data/models',
                help='basedir for saving model weights')
    args = parser.parse_args()
    train(args.run_name, args.digit, args.num_epochs, args.batch_size, args.latent_dim, args.noise, args.k, args.optimizer, args.model_dir, [args.gen_intermediate_dim], [args.disc_intermediate_dim])
