from __future__ import print_function
import os.path
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from keras.utils.generic_utils import Progbar
from model import build_generator, build_discriminator, build_combined, sample_noise

# def ldiv(A, b):
#     return np.linalg.lstsq(A,b)[0]

# def cursor_mats(A, B, c):
#     """
#     BCI: v_t = A v_{t-1} + B y + c
#     This function assumes for simplicity that v_{t-1} = v_t

#     returns matrices D, d such that v_t = Dy + d
#     """
#     D = ldiv(np.eye(A.shape[0]) - A, B)
#     d = ldiv(np.eye(A.shape[0]) - A, c)
#     return D, d

# def cursor_loss(v, vpred):
#     pass

def load_data(fnm):
    D = loadmat(fnm, struct_as_record=False)
    B = D['F'][0,0].train[0,0]
    X = B.latents
    Y = B.thetas
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.1)
    return Xtr, Xte, Ytr, Yte

def train_on_batch(Xtr, batch_index, batch_size, latent_size):

    # generate samples
    Zba = sample_noise((batch_size, latent_size))
    Xba_hat = generator.predict(Zba, verbose=False)

    # train discriminator to tell apart true and fake samples
    Xba = Xtr[batch_index * batch_size:(batch_index + 1) * batch_size]
    X = np.concatenate((Xba, Xba_hat))
    y = np.array([1] * batch_size + [0] * batch_size)
    batch_disc_loss = discriminator.train_on_batch(X, y)

    # train generator to fool discriminator
    #   n.b. want generator to get same # of samples as discriminator
    Zba = sample_noise((2*batch_size, latent_size))
    yhope = np.ones(2 * batch_size) # hope they all look like true samples
    batch_gen_loss = combined.train_on_batch(Zba, yhope)

    return batch_disc_loss, batch_gen_loss

def evaluate(Xte, latent_size):
    """
    same as train_on_batch only now we evaluate
    """
    ntest = Xte.shape[0]

    Zte = sample_noise((ntest, latent_size))
    Xte_hat = generator.predict(Zte, verbose=False)

    X = np.concatenate((Xte, Xte_hat))
    y = np.array([1]*ntest + [0]*ntest)
    test_disc_loss = discriminator.evaluate(X, y, verbose=False)

    Zte = sample_noise((2*ntest, latent_size))
    yhope = np.ones(2*ntest)
    test_gen_loss = combined.evaluate(Zte, yhope, verbose=False)

    return test_disc_loss, test_gen_loss

def get_filenames(model_dir, run_name):
    fnm_gen = os.path.join(model_dir, run_name + '_gen_{0:03d}.h5')
    fnm_disc = os.path.join(model_dir, run_name + '_disc_{0:03d}.h5')
    fnm_hist = os.path.join(model_dir, run_name + '_history.pickle')
    return fnm_gen, fnm_disc, fnm_hist

def print_history(history):
    ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
    print(ROW_FMT.format('generator (train)',
        *history['train']['generator'][-1]))
    print(ROW_FMT.format('generator (test)',
        *history['test']['generator'][-1]))
    print(ROW_FMT.format('discriminator (train)',
        *history['train']['discriminator'][-1]))
    print(ROW_FMT.format('discriminator (test)',
        *history['test']['discriminator'][-1]))
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

def train(run_name, nepochs, batch_size, latent_dim,
    optimizer, data_file, gen_intermediate_dims=None,
    disc_intermediate_dims=None):

    # load data and params
    Xtr, Xte, _, _ = load_data(data_file)
    original_dim = Xtr.shape[-1]
    nbatches = int(Xtr.shape[0]/batch_size)
    fnm_gen, fnm_disc, fnm_hist = get_filenames(model_dir, run_name)

    # load models
    generator = build_generator(batch_size, latent_dim,
        gen_intermediate_dims, original_dim, optimizer)
    discriminator = build_discriminator(batch_size, original_dim,
        disc_intermediate_dims, optimizer)
    combined = build_combined(batch_size, latent_dim,
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
            batch_loss = train_on_batch(Xtr, batch_index, batch_size, latent_size)
            epoch_losses.append(batch_loss)

        # evaluate on test data
        print('\nTesting epoch {}:'.format(i + 1))
        test_loss = evaluate(Xte, latent_size)
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
    parser.add_argument('--batch_size', type=int, default=100,
                help='batch size')
    parser.add_argument('--latent_dim', type=int, default=4,
                help='latent dim')
    parser.add_argument('--gen_intermediate_dim', type=int, default=4,
                help='generator intemediate dim')
    parser.add_argument('--disc_intermediate_dim', type=int, default=4,
                help='discriminator intemediate dim')
    parser.add_argument('--optimizer', type=str, default='adam',
                help='optimizer name') # 'rmsprop'
    parser.add_argument('--model_dir', type=str,
                default='data/models',
                help='basedir for saving model weights')
    parser.add_argument('--train_file', type=str,
                default='data/input/20120525.mat',
                help='file of training data (.mat)')
    args = parser.parse_args()
    train(args.run_name, args.num_epochs, args.batch_size, args.latent_dim, args.optimizer, args.train_file, [args.gen_intermediate_dim], [args.disc_intermediate_dim])
