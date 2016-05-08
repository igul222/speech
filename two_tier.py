"""
RNN Speech Generation Model
Ishaan Gulrajani
"""

import os, sys
sys.path.append(os.getcwd())

import dataset

import numpy
import theano
import theano.tensor as T
import lib
import lasagne
import scipy.io.wavfile
import scikits.audiolab

import random
import time
import functools

# Hyperparams
BATCH_SIZE = 128
N_FRAMES = 64 # How many 'frames' to include in each truncated BPTT pass
FRAME_SIZE = 4 # How many samples per frame
DIM = 512 # Model dimensionality. 512 is sufficient for model development; 1024 if you want good samples.
N_GRUS = 3 # How many GRUs to stack in the frame-level model
Q_LEVELS = 256 # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
GRAD_CLIP = 1 # Elementwise grad clip threshold

# Dataset
DATA_PATH = '/media/seagate/blizzard/parts'
N_FILES = 141703
BITRATE = 16000

# Other constants
PRINT_EVERY = 10000 # Print cost, generate samples, save model checkpoint every N iterations.
TEST_SET_SIZE = 128 # How many audio files to use for the test set
SEQ_LEN = N_FRAMES * FRAME_SIZE # Total length (# of samples) of each truncated BPTT sequence
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude

def frame_level_rnn(input_sequences, h0):
    """
    input_sequences.shape: (batch size, n frames * FRAME_SIZE)
    h0.shape:              (batch size, N_GRUS, DIM)
    output.shape:          (batch size, n frames * FRAME_SIZE, DIM)
    """

    frames = input_sequences.reshape((
        input_sequences.shape[0],
        input_sequences.shape[1] / FRAME_SIZE,
        FRAME_SIZE
    ))

    # Rescale prev_frames from ints in [0, Q_LEVELS) to floats in [-2, 2]
    # (a reasonable range to pass as inputs to the RNN)
    frames = (frames.astype('float32') / lib.floatX(Q_LEVELS/2.)) - lib.floatX(1)
    frames *= lib.floatX(2)

    if N_GRUS != 3:
        raise Exception('N_GRUS must be 3, at least for now')

    gru1 = lib.ops.LowMemGRU('FrameLevel.GRU1', FRAME_SIZE, DIM, frames, h0=h0[:, 0])
    gru2 = lib.ops.LowMemGRU('FrameLevel.GRU2', DIM, DIM, gru1, h0=h0[:, 1])
    gru3 = lib.ops.LowMemGRU('FrameLevel.GRU3', DIM, DIM, gru2, h0=h0[:, 2])

    output = lib.ops.Linear(
        'FrameLevel.Output', 
        DIM,
        FRAME_SIZE * DIM,
        gru3,
        initialization='he'
    )
    output = output.reshape((output.shape[0], output.shape[1] * FRAME_SIZE, DIM))

    last_hidden = T.stack([gru1[:, -1], gru2[:, -1], gru3[:, -1]], axis=1)

    return (output, last_hidden)

def sample_level_predictor(frame_level_outputs, prev_samples):
    """
    frame_level_outputs.shape: (batch size, DIM)
    prev_samples.shape:        (batch size, FRAME_SIZE)
    output.shape:              (batch size, Q_LEVELS)
    """

    prev_samples = lib.ops.Embedding(
        'SampleLevel.Embedding',
        Q_LEVELS,
        Q_LEVELS,
        prev_samples
    ).reshape((-1, FRAME_SIZE * Q_LEVELS))

    out = lib.ops.Linear(
        'SampleLevel.L1_PrevSamples', 
        FRAME_SIZE * Q_LEVELS,
        DIM,
        prev_samples,
        biases=False,
        initialization='he'
    )
    out += frame_level_outputs
    out = T.nnet.relu(out)

    out = lib.ops.Linear('SampleLevel.L2', DIM, DIM, out, initialization='he')
    out = T.nnet.relu(out)
    out = lib.ops.Linear('SampleLevel.L3', DIM, DIM, out, initialization='he')
    out = T.nnet.relu(out)

    # We apply the softmax later
    return lib.ops.Linear('SampleLevel.Output', DIM, Q_LEVELS, out)

sequences   = T.imatrix('sequences')
h0          = T.tensor3('h0')

input_sequences = sequences[:, :-FRAME_SIZE]
target_sequences = sequences[:, FRAME_SIZE:]

frame_level_outputs, new_h0 = frame_level_rnn(input_sequences, h0)

prev_samples = sequences[:, :-1]
prev_samples = prev_samples.reshape((1, BATCH_SIZE, -1, 1))
prev_samples = T.nnet.neighbours.images2neibs(prev_samples, (FRAME_SIZE, 1), neib_step=(1, 1), mode='valid')
prev_samples = prev_samples.reshape((BATCH_SIZE * SEQ_LEN, FRAME_SIZE))

sample_level_outputs = sample_level_predictor(
    frame_level_outputs.reshape((BATCH_SIZE * SEQ_LEN, DIM)),
    prev_samples
)

cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(sample_level_outputs),
    target_sequences.flatten()
).mean()

# By default we report cross-entropy cost in bits. 
# Switch to nats by commenting out this line:
cost = cost * lib.floatX(1.44269504089)

params = lib.search(cost, lambda x: hasattr(x, 'param'))
lib._train.print_params_info(cost, params)

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

updates = lasagne.updates.adam(grads, params)

train_fn = theano.function(
    [sequences, h0],
    [cost, new_h0],
    updates=updates,
    on_unused_input='warn'
)

frame_level_generate_fn = theano.function(
    [sequences, h0],
    frame_level_rnn(sequences, h0),
    on_unused_input='warn'
)

frame_level_outputs = T.matrix('frame_level_outputs')
prev_samples = T.imatrix('prev_samples')
sample_level_generate_fn = theano.function(
    [frame_level_outputs, prev_samples],
    lib.ops.softmax_and_sample(
        sample_level_predictor(
            frame_level_outputs, 
            prev_samples
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
    N_SEQS = 5
    LENGTH = 5*BITRATE

    samples = numpy.zeros((N_SEQS, LENGTH), dtype='int32')
    samples[:, :FRAME_SIZE] = Q_ZERO

    h0 = numpy.zeros((N_SEQS, N_GRUS, DIM), dtype='float32')
    frame_level_outputs = None

    for t in xrange(FRAME_SIZE, LENGTH):

        if t % FRAME_SIZE == 0:
            frame_level_outputs, h0 = frame_level_generate_fn(
                samples[:, t-FRAME_SIZE:t], 
                h0
            )

        samples[:, t] = sample_level_generate_fn(
            frame_level_outputs[:, t % FRAME_SIZE], 
            samples[:, t-FRAME_SIZE:t]
        )

    for i in xrange(N_SEQS):
        write_audio_file("sample_{}_{}".format(tag, i), samples[i])

print "Training!"

for epoch in xrange(1000):

    h0 = numpy.zeros((BATCH_SIZE, N_GRUS, DIM), dtype='float32')
    costs = []
    times = []
    data_feeder = dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, Q_LEVELS, Q_ZERO)

    t0 = time.time()

    for i, (seqs, reset) in enumerate(data_feeder):

        if reset:
            h0.fill(0)

        cost, h0 = train_fn(seqs, h0)

        costs.append(cost)
        times.append(time.time() - t0)

        if len(costs) == PRINT_EVERY:
            print "epoch:{}\titer:{}\ttrain cost:{}\titer time:{}".format(
                epoch, 
                i+1, 
                numpy.mean(costs),
                numpy.mean(times)
            )
            costs = []
            times = []
            tag = "epoch{}_iter{}".format(epoch, i+1)
            generate_and_save_samples(tag)
            lib.save_params('params_{}.pkl'.format(tag))

        t0 = time.time()