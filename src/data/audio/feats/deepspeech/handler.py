"""
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about VOCA is available at http://voca.is.tue.mpg.de.
For comments or questions, please email us at voca@tue.mpg.de
"""

# suppress tf logs
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import copy
import re
from functools import lru_cache

import numpy as np
import resampy
import tensorflow as tf
from python_speech_features import mfcc
from tqdm.auto import tqdm

tf.compat.v1.disable_eager_execution()


def interpolate_features(features, input_rate, output_rate, output_len=None):
    num_features = features.shape[1]
    input_len = features.shape[0]
    seq_len = input_len / float(input_rate)
    if output_len is None:
        output_len = int(seq_len * output_rate)
    input_timestamps = np.arange(input_len) / float(input_rate)
    output_timestamps = np.arange(output_len) / float(output_rate)
    output_features = np.zeros((output_len, num_features))
    for feat in range(num_features):
        output_features[:, feat] = np.interp(output_timestamps, input_timestamps, features[:, feat])
    return output_features


@lru_cache(maxsize=1)
def get_alphabet_list(alphabet_fname):
    assert os.path.exists(alphabet_fname), "Failed to find alphabet at: {}".format(alphabet_fname)
    # load alphabet
    alphabet = []
    with open(alphabet_fname) as fp:
        start = False
        for line in fp:
            line = line.strip()
            # comment
            if not start and len(line) > 0 and line[0] == "#":
                continue
            # first element: <space>
            if len(line) == 0:
                if not start:
                    start = True
                    alphabet.append(" ")
                continue
            if start and len(line) > 0:
                alphabet.append(line[0])
    return alphabet


graph = None


def convert_to_deepspeech(audio, config):

    # check path
    assert os.path.exists(config["deepspeech_graph_fname"]), "Failed to find graph file for deepspeech at: {}".format(
        config["deepspeech_graph_fname"]
    )

    def audioToInputVector(audio, fs, numcep, numcontext):
        # Get mfcc coefficients
        features = mfcc(audio, samplerate=fs, numcep=numcep)

        # We only keep every second feature (BiRNN stride = 2)
        features = features[::2]

        # One stride per time step in the input
        num_strides = len(features)

        # Add empty initial and final contexts
        empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)
        features = np.concatenate((empty_context, features, empty_context))

        # Create a view into the array with overlapping strides of size
        # numcontext (past) + 1 (present) + numcontext (future)
        window_size = 2 * numcontext + 1
        train_inputs = np.lib.stride_tricks.as_strided(
            features,
            (num_strides, window_size, numcep),
            (features.strides[0], features.strides[0], features.strides[1]),
            writeable=False,
        )

        # Flatten the second and third dimensions
        train_inputs = np.reshape(train_inputs, [num_strides, -1])

        train_inputs = np.copy(train_inputs)
        train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)

        # Return results
        return train_inputs

    if type(audio) == dict:
        pass
    else:
        raise ValueError("Wrong type for audio")

    global graph
    if graph is None:
        # Load graph and place_holders
        with tf.io.gfile.GFile(config["deepspeech_graph_fname"], "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        graph = tf.compat.v1.get_default_graph()
        tf.import_graph_def(graph_def, name="deepspeech")

    # input_tensor = graph.get_tensor_by_name("deepspeech/input_node:0")
    # seq_length = graph.get_tensor_by_name("deepspeech/input_lengths:0")
    # layer_6 = graph.get_tensor_by_name("deepspeech/logits:0")
    input_tensor = "deepspeech/input_node:0"
    seq_length = "deepspeech/input_lengths:0"
    layer_6 = "deepspeech/logits:0"

    n_input = 26
    n_context = 9

    processed_audio = copy.deepcopy(audio)
    with tf.compat.v1.Session(graph=graph) as sess:
        for subj in audio.keys():
            todo = list(audio[subj].keys())
            progress = tqdm(todo, disable=len(todo) <= 1)
            for seq in progress:
                progress.set_description(f"process {seq}")

                audio_sample = audio[subj][seq]["audio"]
                sample_rate = audio[subj][seq]["sample_rate"]
                # if sample_rate != 16000:
                #     resampled_audio = resampy.resample(audio_sample.astype(float), sample_rate, 16000)
                # else:
                #     resampled_audio = audio_sample
                # resampled_audio[resampled_audio >  32767] =  32767
                # resampled_audio[resampled_audio < -32768] = -32768
                # input_vector = audioToInputVector(resampled_audio.astype('int16'), 16000, n_input, n_context)
                input_vector = audioToInputVector(audio_sample.astype("int16"), sample_rate, n_input, n_context)

                network_output = sess.run(
                    layer_6,
                    feed_dict={input_tensor: input_vector[np.newaxis, ...], seq_length: [input_vector.shape[0]]},
                )
                network_output = network_output[:, 0]

                # the output is 50 fps
                processed_audio[subj][seq]["audio"] = np.array(network_output)
    return processed_audio
