import numpy as np
import pickle
from tqdm import tqdm


def interpolate_streamline(streamline, n):
    cumulative_lengths = np.cumsum(np.sqrt(np.sum(np.diff(streamline, axis=0) ** 2, axis=1)))
    cumulative_lengths = np.insert(cumulative_lengths, 0, 0)
    t = np.linspace(0, cumulative_lengths[-1], n)
    x = np.interp(t, cumulative_lengths, streamline[:, 0])
    y = np.interp(t, cumulative_lengths, streamline[:, 1])
    z = np.interp(t, cumulative_lengths, streamline[:, 2])
    return np.stack([x, y, z], axis=1)


if __name__ == '__main__':
    # streamlines = np.load('streamlines.npz')
    with open("streamlines.pkl", "rb") as f:
        streamlines = pickle.load(f)

    interpolated_streamlines = []
    for i in tqdm(range(len(streamlines))):
        streamline = streamlines[i].astype(np.float32)
        interpolated_streamline = interpolate_streamline(streamline, 256)
        interpolated_streamlines.append(interpolated_streamline)

    interpolated_streamlines = np.array(interpolated_streamlines)
    np.savez('interpolated_streamlines.npz', interpolated_streamlines)
