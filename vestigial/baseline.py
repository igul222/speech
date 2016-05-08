"""
RNN Speech Generation Model
Ishaan Gulrajani
(defunct)
"""

import os, sys
sys.path.append(os.getcwd())

import numpy
import theano
import theano.tensor as T
import swft
import lasagne
import scipy.io.wavfile
import scikits.audiolab

import random
import time
import functools

# Hyperparams
BATCH_SIZE = 128
SEQ_LEN = 128
DIM = 2048
Q_LEVELS = 256
GRAD_CLIP = 5
# Other constants
DATA_PATH = '/media/seagate/blizzard/segmented'
Q_ZERO = numpy.int32(127)

def load_segmented_blizzard_metadata():
    with open(DATA_PATH+'/prompts.gui') as prompts_file:
        lines = [l[:-1] for l in prompts_file]

    filepaths = [DATA_PATH + '/wavn/' + fname + '.wav' for fname in lines[::3]]
    transcripts = lines[1::3]

    # Clean up the transcripts
    for i in xrange(len(transcripts)):
        t = transcripts[i]
        t = t.replace('@ ', '')
        t = t.replace('# ', '')
        t = t.replace('| ', '')
        t = t.lower()
        transcripts[i] = t

    # We use '*' as a null padding character
    charmap = {'*': 0}
    inv_charmap = ['*']
    for t in transcripts:
        for char in t:
            if char not in charmap:
                charmap[char] = len(charmap)
                inv_charmap.append(char)

    all_data = zip(filepaths, transcripts)
    random.seed(123)
    random.shuffle(all_data)
    train_data = all_data[2*BATCH_SIZE:]
    test_data  = all_data[:2*BATCH_SIZE]

    return charmap, inv_charmap, train_data, test_data

def load_unsegmented_blizzard_metadata():
    filepaths = ['/media/seagate/blizzard/parts/p{}.flac'.format(i) for i in xrange(141703)]
    transcripts = ['.' for i in xrange(141703)]
    all_data = zip(filepaths, transcripts)
    random.seed(123)
    random.shuffle(all_data)
    train_data = all_data[2*BATCH_SIZE:]
    test_data  = all_data[:2*BATCH_SIZE]
    charmap = {'*': 0, '.': 1}
    inv_charmap = ['*', '.']
    return charmap, inv_charmap, train_data, test_data

# charmap, inv_charmap, train_data, test_data = load_segmented_blizzard_metadata()
charmap, inv_charmap, train_data, test_data = load_unsegmented_blizzard_metadata()

def feed_data(data):
    def read_audio_file(path):
        if path.endswith('wav'):
            audio = scipy.io.wavfile.read(path)[1].astype('float64')
        elif path.endswith('flac'):
            audio = scikits.audiolab.flacread(path)[0]
        else:
            raise Exception('Unknown filetype')

        eps = numpy.float64(1e-5)
        audio -= audio.min()
        audio *= (Q_LEVELS - eps) / audio.max()
        audio += eps/2
        return audio.astype('int32')

    _data = list(data)
    random.shuffle(_data)

    buffer = numpy.full((BATCH_SIZE, 16000*60), Q_ZERO, dtype='int32')
    transcripts = [None] * BATCH_SIZE

    while True:
        # Load new sequences into the buffer if necessary
        resets = numpy.zeros(BATCH_SIZE, dtype='int32')
        for i in xrange(BATCH_SIZE):
            if numpy.all(buffer[i] == Q_ZERO):
                if len(_data) == 0:
                    return # We've exhausted the dataset.
                path, transcript = _data.pop()
                audio = read_audio_file(path)
                if len(audio) > buffer.shape[1]:
                    raise Exception('Audio file too long!')
                buffer[i, :len(audio)] = audio
                transcripts[i] = transcript
                resets[i] = 1

        # Make padded_transcripts from transcripts
        padded_transcripts = numpy.full(
            (BATCH_SIZE, max(len(x) for x in transcripts)),
            charmap['*'],
            dtype='int32'
        )
        for i, t in enumerate(transcripts):
            padded_transcripts[i, :len(t)] = [charmap[c] for c in t]

        # Yield the data batch
        yield (
            buffer[:, :SEQ_LEN],
            padded_transcripts,
            resets
        )

        # Roll the buffer
        buffer[:, :SEQ_LEN] = Q_ZERO
        buffer = numpy.roll(buffer, -SEQ_LEN, axis=1)

def predict(prev_frames, h0):
    """
    prev_frames.shape: (batch size, seq len)
    output.shape: (batch size, seq len, DIM)
    """

    prev_frames = swft.ops.Embedding(
        'FrameLevel.Embedding', 
        Q_LEVELS, 
        Q_LEVELS, 
        prev_frames
    ).reshape((
        prev_frames.shape[0], 
        prev_frames.shape[1],
        Q_LEVELS
    ))

    gru1 = swft.ops.LowMemGRU('FrameLevel.GRU1', Q_LEVELS, DIM, prev_frames, h0=h0[:, :DIM])
    gru2 = swft.ops.LowMemGRU('FrameLevel.GRU2', DIM, DIM, gru1, h0=h0[:, DIM:2*DIM])
    gru3 = swft.ops.LowMemGRU('FrameLevel.GRU3', DIM, DIM, gru2, h0=h0[:, 2*DIM:3*DIM])

    output = swft.ops.Linear(
        'FrameLevel.Output', 
        DIM,
        Q_LEVELS,
        gru3
    )

    last_hidden = T.concatenate([gru1[:, -1], gru2[:, -1], gru3[:, -1]], axis=1)

    return (output, last_hidden)

sequences   = T.imatrix('sequences')
transcripts = T.imatrix('transcripts')
h0          = T.matrix('h0')

frame_level_outputs, new_h0 = predict(sequences, h0)

cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(frame_level_outputs[:, :-1].reshape((-1, Q_LEVELS))),
    sequences[:, 1:].flatten()
).mean()

cost = cost * swft.floatX(1.44269504089)
cost.name = 'cost'

params = swft.search(cost, lambda x: hasattr(x, 'param'))
swft._train._print_paramsets_info([cost], [params])

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
grads = [T.clip(g, swft.floatX(-GRAD_CLIP), swft.floatX(GRAD_CLIP)) for g in grads]

updates = lasagne.updates.adam(grads, params)

train_fn = theano.function(
    [sequences, transcripts, h0],
    [cost, new_h0],
    updates=updates,
    on_unused_input='warn'
)

predict_fn = theano.function(
    [sequences, transcripts, h0],
    [swft.ops.softmax_and_sample(frame_level_outputs), new_h0],
    on_unused_input='warn'
)

def save_samples(tag):
    if tag % 10 == 0:
        swft.save_params('params_{}.pkl'.format(tag))

    def write_audio(name, data):
        scipy.io.wavfile.write(name+'.wav', 16000, audio_tools.soundsc(data))

    samples = numpy.zeros((5, 5*16000), dtype='int32')
    samples[:, 0] = Q_ZERO

    h0 = numpy.zeros((5, 3*DIM), dtype='float32')
    transcripts = numpy.full((5, 1), charmap['.'], dtype='int32')
    for i in xrange(1, samples.shape[1]):
        outputs, h0 = predict_fn(samples[:, i-1:i], transcripts, h0)
        samples[:, i:i+1] = outputs

    for i in xrange(5):
        write_audio("sample_{}_{}".format(tag,i), samples[i].astype('float32'))

# super hacky train loop, rewrite when i have time.
while True: # TODO mark which epoch we're on
    h0 = numpy.zeros((BATCH_SIZE, 3*DIM), dtype='float32')
    costs = []
    for i, (seqs, transcripts, reset) in enumerate(feed_data(train_data)):
        for j in xrange(BATCH_SIZE):
            if reset[j]:
                h0[j, :] = 0.
        cost, h0 = train_fn(seqs, transcripts, h0)
        costs.append(cost)
        print cost
        if len(costs) == 10000:
            print "{} mean: {}".format(i+1, numpy.mean(costs))
            costs = []
            save_samples(i+1)