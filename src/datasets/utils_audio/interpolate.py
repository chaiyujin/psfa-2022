import numpy as np


# adapt from https://github.com/TimoBolkart/voca/blob/master/utils/audio_handler.py
def interpolate_features(features, input_rate, output_rate, output_len=None):
    assert features.ndim == 2
    input_len, num_features = features.shape
    seq_len = input_len / float(input_rate)
    if output_len is None:
        output_len = int(seq_len * output_rate)
    input_timestamps = np.arange(input_len) / float(input_rate)
    output_timestamps = np.arange(output_len) / float(output_rate)
    output_features = np.zeros((output_len, num_features))
    for feat in range(num_features):
        output_features[:, feat] = np.interp(output_timestamps, input_timestamps, features[:, feat])
    return output_features
