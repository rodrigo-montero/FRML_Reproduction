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

def shd_events_to_bins(events, n_time_bins=1000, input_size=700):
    """
    Convert SHD event stream to fixed time-bin spike representation.

    Expected output shape:
        (n_time_bins, input_size)

    SHD events usually contain:
        t = spike time
        x = input neuron/channel id
    """
    frames = np.zeros((n_time_bins, input_size), dtype=np.float32)

    if len(events) == 0:
        return frames

    names = events.dtype.names

    if "t" not in names:
        raise ValueError(f"Expected field 't' in SHD events, got {names}")

    if "x" in names:
        channels = events["x"]
    elif "addr" in names:
        channels = events["addr"]
    else:
        raise ValueError(f"Expected field 'x' or 'addr' in SHD events, got {names}")

    times = events["t"]

    t_min, t_max = times.min(), times.max()
    if t_max == t_min:
        return frames

    bin_indices = ((times - t_min) / (t_max - t_min + 1e-8) * (n_time_bins - 1)).astype(int)

    valid = (
        (bin_indices >= 0) & (bin_indices < n_time_bins) &
        (channels >= 0) & (channels < input_size)
    )

    np.add.at(frames, (bin_indices[valid], channels[valid]), 1.0)

    frames = (frames > 0).astype(np.float32)

    return frames


def dvs_gesture_events_to_frames(
    events,
    sensor_size=(128, 128, 2),
    output_size=(64, 64, 2),
    time_window=20000,
    max_time_bins=80,
):
    """
    Convert DVSGesture events to time-binned frames.

    Paper:
    - original frames: 128x128x2
    - scaled down to 64x64x2
    - Tonic time window = 20000

    Output:
        (time_bins, 64*64*2)
    """
    import numpy as np

    out_h, out_w, polarities = output_size

    if len(events) == 0:
        return np.zeros((max_time_bins, out_h * out_w * polarities), dtype=np.float32)

    x = events["x"].astype(np.int64)
    y = events["y"].astype(np.int64)
    t = events["t"].astype(np.int64)
    p = events["p"].astype(np.int64)

    # Downsample 128x128 -> 64x64
    x = x // 2
    y = y // 2

    t0 = t.min()
    bin_indices = ((t - t0) // time_window).astype(np.int64)

    # Cap sequence length for manageable experiments
    valid = (
        (bin_indices >= 0) & (bin_indices < max_time_bins) &
        (x >= 0) & (x < out_w) &
        (y >= 0) & (y < out_h) &
        (p >= 0) & (p < polarities)
    )

    frames = np.zeros((max_time_bins, out_h, out_w, polarities), dtype=np.float32)

    np.add.at(
        frames,
        (bin_indices[valid], y[valid], x[valid], p[valid]),
        1.0
    )

    frames = (frames > 0).astype(np.float32)

    return frames.reshape(max_time_bins, -1)