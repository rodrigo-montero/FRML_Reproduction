import numpy as np


def events_to_frames(events, sensor_size=(34, 34, 2), n_time_bins=30):
    """
    Convert event stream into fixed time-bin spike frames.

    Input events have fields: x, y, t, p.
    Output shape: (n_time_bins, 34*34*2)
    """
    height, width, polarities = sensor_size

    frames = np.zeros((n_time_bins, height, width, polarities), dtype=np.float32)

    t = events["t"]
    if len(t) == 0:
        return frames.reshape(n_time_bins, -1)

    t_min, t_max = t.min(), t.max()
    if t_max == t_min:
        return frames.reshape(n_time_bins, -1)

    bin_indices = ((t - t_min) / (t_max - t_min + 1e-8) * (n_time_bins - 1)).astype(int)

    x = events["x"]
    y = events["y"]
    p = events["p"]

    valid = (
        (x >= 0) & (x < width) &
        (y >= 0) & (y < height) &
        (p >= 0) & (p < polarities)
    )

    np.add.at(frames, (bin_indices[valid], y[valid], x[valid], p[valid]), 1.0)

    # Convert counts to binary spikes
    frames = (frames > 0).astype(np.float32)

    return frames.reshape(n_time_bins, -1)