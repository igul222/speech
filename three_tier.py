"""
RNN Speech Generation Model
Ishaan Gulrajani
"""

import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.register_crash_notifier()
    experiment_tools.wait_for_gpu(high_priority=False, debug=True)
except ImportError:
    pass

import numpy
numpy.random.seed(123)
import random
random.seed(123)

import dataset

import theano
import theano.tensor as T
import theano.tensor.nnet.neighbours
import theano.ifelse
import lib
import lasagne
import scipy.io.wavfile

import time
import functools
import itertools

# Hyperparams
BATCH_SIZE = 128
SEQ_LEN = 512 # How many samples to include in each truncated BPTT pass
PRE_SEQ_LEN = 1024
FRAME_SIZE = 2 # How many samples per frame
N_GRUS = 1 # How many GRUs to stack in the frame-level model
BIG_FRAME_SIZE = 8 # how many samples per big frame
N_BIG_GRUS = 4 # how many GRUs to stack in the big-frame-level model
assert(SEQ_LEN % BIG_FRAME_SIZE == 0)
assert(BIG_FRAME_SIZE % FRAME_SIZE == 0)
DIM = 1024 # Model dimensionality. 512 is sufficient for model development; 1024 if you want good samples.
BIG_DIM = 1024 # dimensionality for the slowest level
Q_LEVELS = 256 # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
GRAD_CLIP = 1 # Elementwise grad clip threshold

# Dataset
DATA_PATH = '/media/seagate/blizzard/parts'
N_FILES = 141703
# DATA_PATH = '/PersimmonData/kiwi_parts'
# N_FILES = 516
BITRATE = 16000

# Other constants
TEST_SET_SIZE = 128 # How many audio files to use for the test set
N_FRAMES = SEQ_LEN / FRAME_SIZE # Number of frames in each truncated BPTT pass
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude

# # Pretrain loop
PRE_TRAIN_MODE = 'time' # only time supported right now
PRE_PRINT_TIME = 60*60 # Print cost, generate samples, save model checkpoint every N seconds.
PRE_STOP_TIME = 60*60*4 # Stop after this many seconds of actual training (not including time req'd to generate samples etc.)
PRE_PRINT_ITERS = 0
PRE_STOP_ITERS = 0

# in between "pretraining" and "fine tuning" (i.e. end-to-end) there's a period 
# where we only train the bottom levels, so that when we train end-to-end we 
# don't screw up the top levels with gradients from the random bottom levels
# if PRE_STOP_TIME > 0:
# TIME_BEFORE_FINETUNE = 60*60*1
TIME_BEFORE_FINETUNE = 0

# Train loop
TRAIN_MODE = 'iters' # 'iters' to use PRINT_ITERS and STOP_ITERS, 'time' to use PRINT_TIME and STOP_TIME
PRINT_ITERS = 1 # Print cost, generate samples, save model checkpoint every N iterations.
STOP_ITERS = 100000 # Stop after this many iterations
PRINT_TIME = 60*60 # Print cost, generate samples, save model checkpoint every N seconds.
STOP_TIME = 60*60*12 # Stop after this many seconds of actual training (not including time req'd to generate samples etc.)

STOP_TIME -= PRE_STOP_TIME

print "Model settings:"
all_vars = [(k,v) for (k,v) in locals().items() if (k.isupper() and k != 'T')]
all_vars = sorted(all_vars, key=lambda x: x[0])
for var_name, var_value in all_vars:
    print "\t{}: {}".format(var_name, var_value)

def big_frame_level_rnn(input_sequences, h0, reset):
    """
    input_sequences.shape: (batch size, n big frames * BIG_FRAME_SIZE)
    h0.shape:              (batch size, N_BIG_GRUS, BIG_DIM)
    reset.shape:           ()
    output[0].shape:       (batch size, n frames, DIM)
    output[1].shape:       same as h0.shape
    output[2].shape:       (batch size, seq len, Q_LEVELS)
    """

    learned_h0 = lib.param(
        'BigFrameLevel.h0',
        numpy.zeros((N_BIG_GRUS, BIG_DIM), dtype=theano.config.floatX)
    )
    learned_h0 = T.alloc(learned_h0, h0.shape[0], N_BIG_GRUS, BIG_DIM)
    learned_h0 = T.patternbroadcast(learned_h0, [False] * learned_h0.ndim)
    h0 = theano.ifelse.ifelse(reset, learned_h0, h0)

    frames = input_sequences.reshape((
        input_sequences.shape[0],
        input_sequences.shape[1] / BIG_FRAME_SIZE,
        BIG_FRAME_SIZE
    ))

    # Rescale frames from ints in [0, Q_LEVELS) to floats in [-2, 2]
    # (a reasonable range to pass as inputs to the RNN)
    frames = (frames.astype('float32') / lib.floatX(Q_LEVELS/2)) - lib.floatX(1)
    frames *= lib.floatX(2)

    gru0 = lib.ops.LowMemGRU('BigFrameLevel.GRU0', BIG_FRAME_SIZE, BIG_DIM, frames, h0=h0[:, 0])
    grus = [gru0]
    for i in xrange(1, N_BIG_GRUS):
        gru = lib.ops.LowMemGRU('BigFrameLevel.GRU'+str(i), BIG_DIM, BIG_DIM, grus[-1], h0=h0[:, i])
        grus.append(gru)

    output = lib.ops.Linear(
        'BigFrameLevel.Output', 
        BIG_DIM,
        DIM * BIG_FRAME_SIZE / FRAME_SIZE,
        grus[-1]
    )
    output = output.reshape((output.shape[0], output.shape[1] * BIG_FRAME_SIZE / FRAME_SIZE, DIM))

    last_hidden = T.stack([gru[:,-1] for gru in grus], axis=1)

    independent_preds = lib.ops.Linear(
        'BigFrameLevel.IndependentPreds', 
        BIG_DIM,
        Q_LEVELS * BIG_FRAME_SIZE,
        grus[-1]
    )
    independent_preds = independent_preds.reshape((independent_preds.shape[0], independent_preds.shape[1] * BIG_FRAME_SIZE, Q_LEVELS))

    return (output, last_hidden, independent_preds)

def frame_level_rnn(input_sequences, other_input, h0, reset):
    """
    input_sequences.shape: (batch size, n frames * FRAME_SIZE)
    other_input.shape:     (batch size, n frames, DIM)
    h0.shape:              (batch size, N_GRUS, DIM)
    reset.shape:           ()
    output.shape:          (batch size, n frames * FRAME_SIZE, DIM)
    """

    learned_h0 = lib.param(
        'FrameLevel.h0',
        numpy.zeros((N_GRUS, DIM), dtype=theano.config.floatX)
    )
    learned_h0 = T.alloc(learned_h0, h0.shape[0], N_GRUS, DIM)
    learned_h0 = T.patternbroadcast(learned_h0, [False] * learned_h0.ndim)
    h0 = theano.ifelse.ifelse(reset, learned_h0, h0)

    frames = input_sequences.reshape((
        input_sequences.shape[0],
        input_sequences.shape[1] / FRAME_SIZE,
        FRAME_SIZE
    ))

    # Rescale frames from ints in [0, Q_LEVELS) to floats in [-2, 2]
    # (a reasonable range to pass as inputs to the RNN)
    frames = (frames.astype('float32') / lib.floatX(Q_LEVELS/2)) - lib.floatX(1)
    frames *= lib.floatX(2)

    gru_input = lib.ops.Linear('FrameLevel.InputExpand', FRAME_SIZE, DIM, frames) + other_input

    gru0 = lib.ops.LowMemGRU('FrameLevel.GRU0', DIM, DIM, gru_input, h0=h0[:, 0])
    grus = [gru0]
    for i in xrange(1, N_GRUS):
        gru = lib.ops.LowMemGRU('FrameLevel.GRU'+str(i), DIM, DIM, grus[-1], h0=h0[:, i])
        grus.append(gru)

    output = lib.ops.Linear(
        'FrameLevel.Output', 
        DIM,
        FRAME_SIZE * DIM,
        grus[-1],
        initialization='he'
    )
    output = output.reshape((output.shape[0], output.shape[1] * FRAME_SIZE, DIM))

    last_hidden = T.stack([gru[:,-1] for gru in grus], axis=1)

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
big_h0      = T.tensor3('big_h0')
reset       = T.iscalar('reset')

big_input_sequences = sequences[:, :-BIG_FRAME_SIZE]
input_sequences = sequences[:, BIG_FRAME_SIZE-FRAME_SIZE:-FRAME_SIZE]
target_sequences = sequences[:, BIG_FRAME_SIZE:]

big_frame_level_outputs, new_big_h0, big_frame_independent_preds = big_frame_level_rnn(big_input_sequences, big_h0, reset)

frame_level_outputs, new_h0 = frame_level_rnn(input_sequences, big_frame_level_outputs, h0, reset)

prev_samples = sequences[:, BIG_FRAME_SIZE-FRAME_SIZE:-1]
prev_samples = prev_samples.reshape((1, BATCH_SIZE, 1, -1))
prev_samples = T.nnet.neighbours.images2neibs(prev_samples, (1, FRAME_SIZE), neib_step=(1, 1), mode='valid')
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

ip_cost = lib.floatX(1.44269504089) * T.nnet.categorical_crossentropy(
    T.nnet.softmax(big_frame_independent_preds.reshape((-1, Q_LEVELS))),
    target_sequences.flatten()
).mean()

all_params = lib.search(cost, lambda x: hasattr(x, 'param'))
ip_params = lib.search(ip_cost, lambda x: hasattr(x, 'param') and 'BigFrameLevel' in x.name)
other_params = [p for p in all_params if p not in ip_params]
all_params = ip_params + other_params
lib._train.print_params_info(ip_cost, ip_params)
lib._train.print_params_info(cost, other_params)
lib._train.print_params_info(cost, all_params)

ip_grads = T.grad(ip_cost, wrt=ip_params, disconnected_inputs='warn')
ip_grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in ip_grads]

other_grads = T.grad(cost, wrt=other_params, disconnected_inputs='warn')
other_grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in other_grads]

grads = T.grad(cost, wrt=all_params, disconnected_inputs='warn')
grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]


ip_updates = lasagne.updates.adam(ip_grads, ip_params)
other_updates = lasagne.updates.adam(other_grads, other_params)
updates = lasagne.updates.adam(grads, all_params)

ip_train_fn = theano.function(
    [sequences, big_h0, reset],
    [ip_cost, new_big_h0],
    updates=ip_updates,
    on_unused_input='warn'
)

other_train_fn = theano.function(
    [sequences, big_h0, h0, reset],
    [cost, new_big_h0, new_h0],
    updates=other_updates,
    on_unused_input='warn'
)

train_fn = theano.function(
    [sequences, big_h0, h0, reset],
    [cost, new_big_h0, new_h0],
    updates=updates,
    on_unused_input='warn'
)

big_frame_level_generate_fn = theano.function(
    [sequences, big_h0, reset],
    big_frame_level_rnn(sequences, big_h0, reset)[0:2],
    on_unused_input='warn'
)

big_frame_level_outputs = T.matrix('big_frame_level_outputs')
frame_level_generate_fn = theano.function(
    [sequences, big_frame_level_outputs, h0, reset],
    frame_level_rnn(sequences, big_frame_level_outputs.dimshuffle(0,'x',1), h0, reset),
    on_unused_input='warn'
)

frame_level_outputs = T.matrix('frame_level_outputs')
prev_samples        = T.imatrix('prev_samples')
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
    N_SEQS = 10
    LENGTH = 5*BITRATE

    samples = numpy.zeros((N_SEQS, LENGTH), dtype='int32')
    samples[:, :BIG_FRAME_SIZE] = Q_ZERO

    big_h0 = numpy.zeros((N_SEQS, N_BIG_GRUS, BIG_DIM), dtype='float32')
    h0 = numpy.zeros((N_SEQS, N_GRUS, DIM), dtype='float32')
    big_frame_level_outputs = None
    frame_level_outputs = None

    for t in xrange(BIG_FRAME_SIZE, LENGTH):

        if t % BIG_FRAME_SIZE == 0:
            big_frame_level_outputs, big_h0 = big_frame_level_generate_fn(
                samples[:, t-BIG_FRAME_SIZE:t],
                big_h0,
                numpy.int32(t == BIG_FRAME_SIZE)
            )

        if t % FRAME_SIZE == 0:
            frame_level_outputs, h0 = frame_level_generate_fn(
                samples[:, t-FRAME_SIZE:t], 
                big_frame_level_outputs[:, (t / FRAME_SIZE) % (BIG_FRAME_SIZE / FRAME_SIZE)],
                h0,
                numpy.int32(t == BIG_FRAME_SIZE)
            )

        samples[:, t] = sample_level_generate_fn(
            frame_level_outputs[:, t % FRAME_SIZE], 
            samples[:, t-FRAME_SIZE:t]
        )

    for i in xrange(N_SEQS):
        write_audio_file("sample_{}_{}".format(tag, i), samples[i])

if PRE_STOP_TIME > 0:
    print "Pretraining!"
    total_iters = 0
    total_time = 0.
    last_print_time = 0.
    last_print_iters = 0
    pretrain_finished = False

    for epoch in itertools.count():
        if pretrain_finished:
            break

        big_h0 = numpy.zeros((BATCH_SIZE, N_BIG_GRUS, BIG_DIM), dtype='float32')
        costs = []
        data_feeder = dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, PRE_SEQ_LEN, BIG_FRAME_SIZE, Q_LEVELS, Q_ZERO)

        for seqs, reset in data_feeder:
            if pretrain_finished:
                break

            start_time = time.time()
            cost, big_h0 = ip_train_fn(seqs, big_h0, reset)
            total_time += time.time() - start_time
            total_iters += 1

            costs.append(cost)



            if (PRE_TRAIN_MODE=='iters' and total_iters-last_print_iters == PRE_PRINT_ITERS) or \
                (PRE_TRAIN_MODE=='time' and total_time-last_print_time >= PRE_PRINT_TIME):
                
                print "epoch:{}\ttotal iters:{}\ttrain cost:{}\ttotal time:{}\ttime per iter:{}".format(
                    epoch,
                    total_iters,
                    numpy.mean(costs),
                    total_time,
                    total_time / total_iters
                )
                tag = "iters{}_time{}".format(total_iters, total_time)
                lib.save_params('params_pretrain_{}.pkl'.format(tag))

                costs = []
                last_print_time += PRE_PRINT_TIME
                last_print_iters += PRE_PRINT_ITERS

            if (PRE_TRAIN_MODE=='iters' and total_iters == PRE_STOP_ITERS) or \
                (PRE_TRAIN_MODE=='time' and total_time >= PRE_STOP_TIME):

                print "Done!"

                pretrain_finished = True

print "Training!"
total_iters = 0
total_time = 0.
last_print_time = 0.
last_print_iters = 0
last_eigs = 0.
finetune = False
for epoch in itertools.count():

    big_h0 = numpy.zeros((BATCH_SIZE, N_BIG_GRUS, BIG_DIM), dtype='float32')
    h0 = numpy.zeros((BATCH_SIZE, N_GRUS, DIM), dtype='float32')
    costs = []
    data_feeder = dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, BIG_FRAME_SIZE, Q_LEVELS, Q_ZERO)

    for seqs, reset in data_feeder:

        if finetune:
            _train_fn = train_fn
        else:
            _train_fn = other_train_fn

        start_time = time.time()
        cost, big_h0, h0 = _train_fn(seqs, big_h0, h0, reset)
        total_time += time.time() - start_time
        total_iters += 1

        costs.append(cost)

        if (TRAIN_MODE=='iters' and total_iters-last_print_iters == PRINT_ITERS) or \
            (TRAIN_MODE=='time' and total_time-last_print_time >= PRINT_TIME):
            
            print "epoch:{}\ttotal iters:{}\ttrain cost:{}\ttotal time:{}\ttime per iter:{}".format(
                epoch,
                total_iters,
                numpy.mean(costs),
                total_time,
                total_time / total_iters
            )
            print "Warning not generating samples"
            # tag = "iters{}_time{}".format(total_iters, total_time)
            # generate_and_save_samples(tag)
            # lib.save_params('params_{}.pkl'.format(tag))

            if last_print_time <= TIME_BEFORE_FINETUNE <= last_print_time + PRINT_TIME:
                print "Switching to fine-tuning!"
                finetune = True

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