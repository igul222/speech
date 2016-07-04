"""
RNN Speech Generation Model
Ishaan Gulrajani
"""
import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(high_priority=False)
except ImportError:
    pass

import numpy
numpy.random.seed(123)
import random
random.seed(123)

import dataset

import theano
import theano.tensor as T
import theano.ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lib
import lasagne
import scipy.io.wavfile

import time
import functools
import itertools

theano_srng = RandomStreams(seed=234)

# Hyperparams
BATCH_SIZE = 128
FRAME_SIZE = 16
N_FRAMES = (32*16)/FRAME_SIZE
SEQ_LEN = FRAME_SIZE*N_FRAMES # How many audio samples to include in each truncated BPTT pass
DIM = 512 # Model dimensionality. 512 is sufficient for model development; 1024 if you want good samples.
LATENT_DIM = 128
N_GRUS = 2
Q_LEVELS = 256 # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
GRAD_CLIP = 1 # Elementwise grad clip threshold

VANILLA = False

ALPHA_ITERS = 10000

# Dataset
DATA_PATH = '/media/seagate/blizzard/parts'
N_FILES = 141703
# DATA_PATH = '/PersimmonData/kiwi_parts'
# N_FILES = 516
BITRATE = 16000

# Other constants
TRAIN_MODE = 'iters' # 'iters' to use PRINT_ITERS and STOP_ITERS, 'time' to use PRINT_TIME and STOP_TIME
PRINT_ITERS = 1000 # Print cost, generate samples, save model checkpoint every N iterations.
STOP_ITERS = 100000 # Stop after this many iterations
GENERATE_SAMPLES_AND_SAVE_PARAMS = True
PRINT_TIME = 60*60 # Print cost, generate samples, save model checkpoint every N seconds.
STOP_TIME = 60*60*12 # Stop after this many seconds of actual training (not including time req'd to generate samples etc.)
TEST_SET_SIZE = 128 # How many audio files to use for the test set
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude
SAMPLE_LEN = 5*BITRATE
# SAMPLE_LEN = 1024

print "Model settings:"
all_vars = [(k,v) for (k,v) in locals().items() if (k.isupper() and k != 'T')]
all_vars = sorted(all_vars, key=lambda x: x[0])
for var_name, var_value in all_vars:
    print "\t{}: {}".format(var_name, var_value)

def Layer(name, n_in, n_out, inputs):
    output = lib.ops.Linear(name, n_in, n_out, inputs, initialization='he')
    output = T.nnet.relu(output)
    return output

def MLP(name, n_in, n_out, inputs):
    output = Layer(name+'.1', n_in, DIM, inputs)
    output = Layer(name+'.2', DIM, DIM, output)
    output = Layer(name+'.3', DIM, DIM, output)
    output = lib.ops.Linear(name+'.Output', DIM, n_out, output)
    return output

def FrameProcessor(frames):
    """
    frames.shape: (batch size, n frames, FRAME_SIZE)
    output.shape: (batch size, n frames, DIM)
    """

    embedded = lib.ops.Embedding('FrameEmbedding', Q_LEVELS, Q_LEVELS, frames)
    embedded = embedded.reshape((frames.shape[0], frames.shape[1], Q_LEVELS * FRAME_SIZE))
    output = MLP('FrameProcessor', FRAME_SIZE*Q_LEVELS, DIM, embedded)
    return output

    # frames = (frames.astype('float32') / lib.floatX(Q_LEVELS/2)) - lib.floatX(1)
    # frames *= lib.floatX(2)
    # output = MLP('FrameProcessor', FRAME_SIZE, DIM, frames)
    # return output

def LatentsProcessor(latents):
    """
    latents.shape: (batch size, n frames, LATENT_DIM)
    output.shape: (batch size, n frames, DIM)
    """
    return MLP('LatentsProcessor', LATENT_DIM, DIM, latents)

def Prior(contexts):
    """
    contexts.shape: (batch size, n frames, DIM)
    outputs: (mu, log_sigma), each with shape (batch size, n frames, LATENT_DIM)
    """
    mu_and_log_sigma = MLP('Prior', DIM, 2*LATENT_DIM, contexts)
    return mu_and_log_sigma[:,:,:LATENT_DIM], mu_and_log_sigma[:,:,LATENT_DIM:]

def Encoder(processed_frames, contexts):
    """
    processed_frames.shape: (batch size, n frames, DIM)
    contexts.shape: (batch size, n frames, DIM)
    outputs: (mu, log_sigma), each with shape (batch size, n frames, LATENT_DIM)
    """
    inputs = T.concatenate([
        processed_frames,
        contexts
    ], axis=2)
    mu_and_log_sigma = MLP('Encoder', 2*DIM, 2*LATENT_DIM, inputs)
    return mu_and_log_sigma[:,:,:LATENT_DIM], mu_and_log_sigma[:,:,LATENT_DIM:]

def Decoder(latents, contexts, prevs):
    """
    latents.shape: (batch size, n frames, LATENT_DIM)
    contexts.shape: (batch size, n frames, DIM)
    prevs.shape: (batch size, n frames * FRAME_SIZE)
    outputs: (batch size, n frames, FRAME_SIZE, Q_LEVELS)
    """
    inputs = T.concatenate([
        LatentsProcessor(latents),
        contexts
    ], axis=2)
    output = MLP('Decoder', 2*DIM, FRAME_SIZE*Q_LEVELS, inputs)
    return output.reshape((output.shape[0], output.shape[1], FRAME_SIZE, Q_LEVELS))

def Recurrence(processed_frames, h0, reset):
    """
    processed_frames.shape: (batch size, n frames, DIM)
    h0.shape: (batch size, N_GRUS, DIM)
    reset.shape: ()
    output.shape: (batch size, n frames, DIM)
    """

    # print "warning no recurrence"
    # return T.zeros_like(processed_frames), h0

    learned_h0 = lib.param(
        'Recurrence.h0',
        numpy.zeros((N_GRUS, DIM), dtype=theano.config.floatX)
    )
    learned_h0 = T.alloc(learned_h0, h0.shape[0], N_GRUS, DIM)
    learned_h0 = T.patternbroadcast(learned_h0, [False] * learned_h0.ndim)
    h0 = theano.ifelse.ifelse(reset, learned_h0, h0)

    gru0 = lib.ops.LowMemGRU('Recurrence.GRU0', DIM, DIM, processed_frames, h0=h0[:, 0])
    grus = [gru0]
    for i in xrange(1, N_GRUS):
        gru = lib.ops.LowMemGRU('Recurrence.GRU'+str(i), DIM, DIM, grus[-1], h0=h0[:, i])
        grus.append(gru)

    last_hidden = T.stack([gru[:,-1] for gru in grus], axis=1)

    return (grus[-1], last_hidden)


sequences = T.imatrix('sequences')
h0 = T.tensor3('h0')
reset = T.iscalar('reset')

frames = sequences.reshape((sequences.shape[0], -1, FRAME_SIZE))
processed_frames = FrameProcessor(frames)

contexts, new_h0 = Recurrence(processed_frames[:,:-1], h0, reset)

mu_prior, log_sigma_prior = Prior(contexts)
mu_post, log_sigma_post = Encoder(processed_frames[:,1:], contexts)

# log_sigma_prior = T.log(T.nnet.softplus(log_sigma_prior))
# log_sigma_post = T.log(T.nnet.softplus(log_sigma_post))

eps = theano_srng.normal(mu_post.shape).astype('float32')
latents = mu_post
if not VANILLA:
    latents += (T.exp(log_sigma_post) * eps)
else:
    print "warning no latent noise"

reconstructions = Decoder(latents, contexts, sequences[:, FRAME_SIZE-1:-1])

reconst_cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(reconstructions.reshape((-1, Q_LEVELS))),
    frames[:,1:].flatten()
).mean()
reconst_cost.name = 'reconst_cost'


def KLGaussianGaussian(mu1, sig1, mu2, sig2):
    """
    (adapted from https://github.com/jych/cle)
    mu1, sig1 = posterior mu and *log* sigma
    mu2, sig2 = prior mu and *log* sigma
    """
    #    0.5 * (1 + 2*log_sigma - mu**2 - T.exp(2*log_sigma)).mean(axis=0).sum()
    kl = 0.5 * (2*sig2 - 2*sig1 + (T.exp(2*sig1) + (mu1 - mu2)**2) / T.exp(2*sig2) - 1)
    return kl

reg_cost = KLGaussianGaussian(
    mu_post,
    log_sigma_post,
    mu_prior, 
    log_sigma_prior
)
reg_cost = reg_cost.sum() / T.cast(frames[:,1:].flatten().shape[0], 'float32')

# By default we report cross-entropy cost in bits. 
# Switch to nats by commenting out this line:
reg_cost = reg_cost * lib.floatX(1.44269504089)
reconst_cost = reconst_cost * lib.floatX(1.44269504089)

alpha = T.scalar('alpha')
cost = reconst_cost
if not VANILLA:
    cost += (alpha * reg_cost)

params = lib.search(cost, lambda x: hasattr(x, 'param'))
lib._train.print_params_info(cost, params)

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

updates = lasagne.updates.adam(grads, params)

train_fn = theano.function(
    [sequences, h0, reset, alpha],
    [reg_cost, reconst_cost, cost, new_h0],
    updates=updates,
    on_unused_input='warn'
)

gen_fn_contexts, gen_fn_new_h0 = Recurrence(processed_frames, h0, reset)
gen_recurrence_fn = theano.function(
    [sequences, h0, reset],
    [gen_fn_contexts, gen_fn_new_h0],
    on_unused_input='warn'
)

gen_vae_fn = theano.function(
    [contexts],
    lib.ops.softmax_and_sample(
        Decoder(
            mu_prior + theano_srng.normal(mu_prior.shape).astype('float32') * T.exp(log_sigma_prior), 
            contexts
        )
    ),
    on_unused_input='warn'
)

def generate_and_save_samples(tag):

    def write_audio_file(name, data):
        data = data.astype('float32')
        data -= data.min()
        data /= data.max()
        data -= 0.5
        data *= 0.95
        scipy.io.wavfile.write(name+'.wav', BITRATE, data)

    # Generate 5 sample files, each 5 seconds long
    N_SEQS = 10
    LENGTH = SAMPLE_LEN - (SAMPLE_LEN%FRAME_SIZE)

    samples = numpy.zeros((N_SEQS, LENGTH/FRAME_SIZE, FRAME_SIZE), dtype='int32')
    samples[:, 0] = Q_ZERO

    h0 = numpy.zeros((N_SEQS, N_GRUS, DIM), dtype='float32')
    contexts, h0 = gen_recurrence_fn(samples[:,0], h0, numpy.int32(1))

    for frame_i in xrange(1, LENGTH/FRAME_SIZE):
        samples[:,frame_i:frame_i+1] = gen_vae_fn(contexts)
        contexts, h0 = gen_recurrence_fn(samples[:,frame_i], h0, numpy.int32(0))

    for i in xrange(N_SEQS):
        write_audio_file("sample_{}_{}".format(tag, i), samples[i].reshape((-1)))

print "Training!"
total_iters = 0
total_time = 0.
last_print_time = 0.
last_print_iters = 0
reg_costs = []
reconst_costs = []
costs = []
for epoch in itertools.count():

    h0 = numpy.zeros((BATCH_SIZE, N_GRUS, DIM), dtype='float32')
    data_feeder = dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, Q_LEVELS, Q_ZERO)

    def sigmoid(x):
      return 1 / (1 + numpy.exp(-x))

    for seqs, reset in data_feeder:

        # alpha = lib.floatX(sigmoid((total_iters - ALPHA_B)/float(ALPHA_A)))
        # if alpha > 0.99:
        #     alpha = lib.floatX(1)
        # if alpha < 1e-5:
        #     alpha = lib.floatX(1e-5)

        # alpha = lib.floatX(0)

        alpha = lib.floatX(float(total_iters) / ALPHA_ITERS)
        if alpha > 1:
            alpha = lib.floatX(1)

        start_time = time.time()
        reg_cost, reconst_cost, cost, h0 = train_fn(seqs, h0, reset, alpha)
        total_time += time.time() - start_time
        total_iters += 1

        reg_costs.append(reg_cost)
        reconst_costs.append(reconst_cost)
        costs.append(cost)

        if (TRAIN_MODE=='iters' and total_iters-last_print_iters == PRINT_ITERS) or \
            (TRAIN_MODE=='time' and total_time-last_print_time >= PRINT_TIME):
            
            print "epoch:{}\ttotal iters:{}\talpha:{}\treg:{}\treconst:{}\tfull:{}\ttotal time:{}\ttime per iter:{}".format(
                epoch,
                total_iters,
                alpha,
                numpy.mean(reg_costs),
                numpy.mean(reconst_costs),
                numpy.mean(costs),
                total_time,
                total_time / total_iters
            )
            tag = "iters{}_time{}".format(total_iters, total_time)

            if GENERATE_SAMPLES_AND_SAVE_PARAMS:
                generate_and_save_samples(tag)
                lib.save_params('params_{}.pkl'.format(tag))

            reg_costs = []
            reconst_costs = []
            costs = []
            last_print_time += PRINT_TIME
            last_print_iters += PRINT_ITERS

        if (TRAIN_MODE=='iters' and total_iters == STOP_ITERS) or \
            (TRAIN_MODE=='time' and total_time >= STOP_TIME):

            print "Done!"

            try: # This only matters on Ishaan's computer
                import experiment_tools
                experiment_tools.send_sms("done!")
            except ImportError:
                pass

            sys.exit()