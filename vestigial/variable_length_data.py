"""
RNN Speech Generation Model
Ishaan Gulrajani

WARNING: I'm pretty sure there's a bug in here somewhere:
I can't get the same training loss that I get with data.py's feed_epoch using
load_sequential_flac_files and feed_data.
"""

import numpy
import scipy.io.wavfile
import scikits.audiolab

import random
import time

def load_segmented_blizzard_metadata(data_path, test_set_size):
    """
    data_path: path to the blizzard dataset (should have a subdirectory 'segmented' with a file 'prompts.gui')
    test_set_size: how many files to use for the test set
    """
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
    train_data = all_data[test_set_size:]
    test_data  = all_data[:test_set_size]

    return charmap, inv_charmap, train_data, test_data

def load_sequential_flac_files(data_path, n_files, test_set_size):
    """
    Load sequentially-named FLAC files in a directory
    (p0.flac, p1.flac, p2.flac, ..., p[n_files-1].flac)

    data_path: directory containing the flac files
    n_files: how many FLAC files are in the directory
    test_set_size: how many files to use for the test set
    """
    filepaths = [data_path+'/p{}.flac'.format(i) for i in xrange(n_files)]
    transcripts = ['*' for i in xrange(n_files)]
    charmap = {'*': 0}
    inv_charmap = ['*']
    all_data = zip(filepaths, transcripts)
    random.seed(123)
    random.shuffle(all_data)
    train_data = all_data[test_set_size:]
    test_data  = all_data[:test_set_size]
    return charmap, inv_charmap, train_data, test_data

def feed_data(data, charmap, shuffle, BATCH_SIZE, BITRATE, Q_LEVELS, Q_ZERO, N_PREV_SAMPLES, SEQ_LEN):
    """
    see the top of twotier.py for a description of the constants
    """
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
    if shuffle:
        random.shuffle(_data)

    # Make sure the buffer size is longer than the longest sample in the dataset
    buffer = numpy.full((BATCH_SIZE, BITRATE*40), Q_ZERO, dtype='int32')
    head = 0
    transcripts = [None] * BATCH_SIZE

    while True:
        # Load new sequences into the buffer if necessary
        resets = numpy.zeros(BATCH_SIZE, dtype='int32')
        for i in xrange(BATCH_SIZE):
            if numpy.array_equiv(buffer[i, head:], Q_ZERO):
                if len(_data) == 0:
                    return # We've exhausted the dataset.
                path, transcript = _data.pop()
                audio = read_audio_file(path)
                # We add a few samples of Q_ZERO in the beginning to match
                # generation time (where we generate starting from zeros).
                if len(audio) + N_PREV_SAMPLES > buffer.shape[1] - head:
                    raise Exception('Audio file too long!')
                buffer[i, head+N_PREV_SAMPLES:head+len(audio)+N_PREV_SAMPLES] = audio
                transcripts[i] = transcript
                resets[i] = 1

        # Make a dense (padded) transcript matrix from transcripts
        padded_transcripts = numpy.full(
            (BATCH_SIZE, max(len(x) for x in transcripts)),
            charmap['*'],
            dtype='int32'
        )
        for i, t in enumerate(transcripts):
            padded_transcripts[i, :len(t)] = [charmap[c] for c in t]

        # Yield the data batch
        yield (
            buffer[:, head:head+SEQ_LEN],
            padded_transcripts,
            resets
        )

        # Advance the head and if needed, roll the buffer
        buffer[:, head:head+SEQ_LEN] = Q_ZERO
        head += SEQ_LEN
        if head > buffer.shape[1] // 100:
            buffer = numpy.roll(buffer, -head, axis=1)
            head = 0