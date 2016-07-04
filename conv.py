"""
Convolutional Speech Generation Model
Ishaan Gulrajani
"""
import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.register_crash_notifier()
    experiment_tools.wait_for_gpu(high_priority=True)
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
import lib
import lasagne
import scipy.io.wavfile

import time
import functools
import itertools

# Hyperparams
BATCH_SIZE = 128
SEQ_LEN = 256
DIM = 128
Q_LEVELS = 256 # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
GRAD_CLIP = 1

LAYERS = 5
FILTER_SIZE = 17

# Dataset
DATA_PATH = '/media/seagate/blizzard/parts'
N_FILES = 141703
# DATA_PATH = '/PersimmonData/kiwi_parts'
# N_FILES = 516
BITRATE = 16000

# Other constants
TRAIN_MODE = 'iters' # 'iters' to use PRINT_ITERS and STOP_ITERS, 'time' to use PRINT_TIME and STOP_TIME
PRINT_ITERS = 10 # Print cost, generate samples, save model checkpoint every N iterations.
STOP_ITERS = 1000 # Stop after this many iterations
PRINT_TIME = 60*60 # Print cost, generate samples, save model checkpoint every N seconds.
STOP_TIME = 60*60*3 # Stop after this many seconds of actual training (not including time req'd to generate samples etc.)
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude

print "Model settings:"
all_vars = [(k,v) for (k,v) in locals().items() if (k.isupper() and k != 'T')]
all_vars = sorted(all_vars, key=lambda x: x[0])
for var_name, var_value in all_vars:
    print "\t{}: {}".format(var_name, var_value)

def MaskedConv1D(name, input_dim, output_dim, filter_size, inputs, mask_type=None, he_init=False):
    """
    inputs.shape: (batch size, input_dim, 1, width)
    mask_type: None, 'a', 'b'
    output.shape: (batch size, output_dim, 1, width)
    """

    if mask_type is not None:
        mask = numpy.ones(
            (output_dim, input_dim, 1, filter_size), 
            dtype=theano.config.floatX
        )
        center = filter_size//2
        mask[:,:,0,center+1:] = 0.
        if mask_type == 'a':
            mask[:,:,0,center] = 0.

    def uniform(stdev, size):
        """uniform distribution with the given stdev and size"""
        return numpy.random.uniform(
            low=-stdev * numpy.sqrt(3),
            high=stdev * numpy.sqrt(3),
            size=size
        ).astype(theano.config.floatX)

    if mask_type=='a':
        n_in = filter_size//2
    elif mask_type=='b':
        n_in = filter_size//2 + 1
    else:
        n_in = filter_size
    n_in *= input_dim

    if he_init:
        init_stdev = numpy.sqrt(2./n_in)
    else:
        init_stdev = numpy.sqrt(1./n_in)

    filters = lib.param(
        name+'.Filters',
        uniform(
            init_stdev,
            (output_dim, input_dim, 1, filter_size)
        )
    )

    if mask_type is not None:
        filters = filters * mask

    # TODO benchmark against the lasagne 'conv1d' implementations
    result = T.nnet.conv2d(inputs, filters, filter_flip=False, border_mode='half')

    if mask_type is not None:
        result = result[:, :, :, :inputs.shape[3]]

    biases = lib.param(
        name+'.Biases',
        numpy.zeros(output_dim, dtype=theano.config.floatX)
    )
    result += biases[None, :, None, None]

    return result

def Conv1D(name, input_dim, output_dim, filter_size, inputs, mask_type=None, he_init=False):
    """
    inputs.shape: (batch size, input_dim, 1, width)
    mask_type: None, 'a', 'b'
    output.shape: (batch size, output_dim, 1, width)
    """

    # if mask_type is not None:
    #     mask = numpy.ones(
    #         (output_dim, input_dim, 1, filter_size), 
    #         dtype=theano.config.floatX
    #     )
    #     center = filter_size//2
    #     mask[:,:,0,center+1:] = 0.
    #     if mask_type == 'a':
    #         mask[:,:,0,center] = 0.

    if mask_type=='a':
        filter_size = filter_size//2
    elif mask_type=='b':
        filter_size = filter_size//2 + 1

    def uniform(stdev, size):
        """uniform distribution with the given stdev and size"""
        return numpy.random.uniform(
            low=-stdev * numpy.sqrt(3),
            high=stdev * numpy.sqrt(3),
            size=size
        ).astype(theano.config.floatX)

    # if mask_type is not None:
    #     n_in = numpy.sum(mask)
    # else:
    n_in = input_dim * filter_size

    if he_init:
        init_stdev = numpy.sqrt(2./n_in)
    else:
        init_stdev = numpy.sqrt(1./n_in)

    filters = lib.param(
        name+'.Filters',
        uniform(
            init_stdev,
            (output_dim, input_dim, 1, filter_size)
        )
    )

    # if mask_type is not None:
    #     filters = filters * mask

    if mask_type=='a':
        pad = filter_size
    elif mask_type=='b':
        pad = filter_size-1
    else:
        # border mode 'half'
        pad = filter_size//2

    # TODO benchmark against the lasagne 'conv1d' implementations
    result = T.nnet.conv2d(inputs, filters, filter_flip=False, border_mode=(0,pad))

    if mask_type is not None:
        result = result[:, :, :, :inputs.shape[3]]

    biases = lib.param(
        name+'.Biases',
        numpy.zeros(output_dim, dtype=theano.config.floatX)
    )
    result += biases[None, :, None, None]

    return result

sequences = T.imatrix('sequences')

INPUT_DIM = Q_LEVELS
inputs = lib.ops.Embedding('Embedding', Q_LEVELS, Q_LEVELS, sequences)
inputs = inputs.dimshuffle(0, 2, 'x', 1)

# INPUT_DIM = 1
# inputs = lib.floatX(4)*sequences.astype('float32')/lib.floatX(Q_LEVELS) - lib.floatX(2)
# inputs = inputs[:, None, None, :]

output = MaskedConv1D('InputConv', INPUT_DIM, DIM, FILTER_SIZE, inputs, mask_type='a', he_init=True)
output = T.nnet.relu(output)

for i in xrange(1,LAYERS):
    output = MaskedConv1D('Conv'+str(i), DIM, DIM, FILTER_SIZE, output, mask_type='b', he_init=True)
    output = T.nnet.relu(output)

output = MaskedConv1D('OutputConv', DIM, Q_LEVELS, 1, output, mask_type='b')

output = output.dimshuffle(0,2,3,1) # Move the Q_LEVELS dim to the end
cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(output.reshape((-1, Q_LEVELS))),
    sequences.flatten()
).mean()

# By default we report cross-entropy cost in bits. 
# Switch to nats by commenting out this line:
cost = cost * lib.floatX(1.44269504089)

params = lib.search(cost, lambda x: hasattr(x, 'param'))
lib._train.print_params_info(cost, params)

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
# Do people use grad clipping in convnets?
# grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

updates = lasagne.updates.adam(grads, params)

train_fn = theano.function(
    [sequences],
    cost,
    updates=updates,
    on_unused_input='warn'
)

# generate_outputs, generate_new_h0 = sample_level_rnn(sequences, h0, reset)
# generate_fn = theano.function(
#     [sequences, h0, reset],
#     [lib.ops.softmax_and_sample(generate_outputs), generate_new_h0],
#     on_unused_input='warn'
# )

# def generate_and_save_samples(tag):

#     def write_audio_file(name, data):
#         data = data.astype('float32')
#         data -= data.min()
#         data /= data.max()
#         data -= 0.5
#         data *= 0.95
#         scipy.io.wavfile.write(name+'.wav', BITRATE, data)

#     # Generate 5 sample files, each 5 seconds long
#     N_SEQS = 10
#     LENGTH = 5*BITRATE

#     samples = numpy.zeros((N_SEQS, LENGTH), dtype='int32')
#     samples[:, 0] = Q_ZERO

#     h0 = numpy.zeros((N_SEQS, N_GRUS, DIM), dtype='float32')

#     for t in xrange(1, LENGTH):
#         samples[:, t:t+1], h0 = generate_fn(
#             samples[:, t-1:t],
#             h0,
#             numpy.int32(t == 1)
#         )

#     for i in xrange(N_SEQS):
#         write_audio_file("sample_{}_{}".format(tag, i), samples[i])

print "Training!"
total_iters = 0
total_time = 0.
last_print_time = 0.
last_print_iters = 0
for epoch in itertools.count():

    costs = []
    data_feeder = dataset.feed_epoch(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, 0, Q_LEVELS, Q_ZERO)

    for seqs, reset in data_feeder:

        start_time = time.time()
        cost = train_fn(seqs)
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
            tag = "iters{}_time{}".format(total_iters, total_time)

            # generate_and_save_samples(tag)
            # lib.save_params('params_{}.pkl'.format(tag))

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