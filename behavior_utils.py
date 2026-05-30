import numpy as np
import pandas as pd

def preprocess_positions(
    behav_df: pd.DataFrame,
    likelihood_thr: float = 0.7,
    distance_thr: float = 50.0,
    cols_to_clear=('x', 'y'),
    max_iter: int = 10,
):
    """
    Preprocess position data in a DLC-style MultiIndex DataFrame.

    Steps:
    1) Set values to NaN where likelihood < likelihood_thr.
    2) Iteratively detect sudden large jumps based on the distance to the
       most recent valid previous frame, and set those frames to NaN,
       until no jumps remain or max_iter is reached.
    """
    df = behav_df.copy()
    bodyparts = df.columns.get_level_values(0).unique()

    for bp in bodyparts:
        # Skip bodyparts without x/y columns
        if (bp, 'x') not in df.columns or (bp, 'y') not in df.columns:
            continue

        x = df[(bp, 'x')]
        y = df[(bp, 'y')]

        # ---- 1) Low-likelihood mask
        mask_low_like = pd.Series(False, index=df.index)
        if (bp, 'likelihood') in df.columns:
            mask_low_like = df[(bp, 'likelihood')] < likelihood_thr

        # ---- Clear specified sub-columns (e.g., x/y) for low-likelihood frames
        for sub in cols_to_clear:
            if (bp, sub) in df.columns:
                df.loc[mask_low_like, (bp, sub)] = np.nan

        # ---- 2) Iterative jump detection and removal
        # Limit the number of iterations to avoid infinite loops
        iter_count = 0

        while True:
            iter_count += 1

            # Current x, y positions (may contain NaNs)
            x = df[(bp, 'x')]
            y = df[(bp, 'y')]

            # Position of the most recent valid previous frame
            x_prev = x.ffill().shift(1)
            y_prev = y.ffill().shift(1)

            # Compute distance only when both current and previous positions are valid
            valid = x.notna() & y.notna() & x_prev.notna() & y_prev.notna()

            dx = x - x_prev
            dy = y - y_prev
            dist = np.sqrt(dx**2 + dy**2).where(valid)

            # Detect frames with large jumps
            mask_jumps = dist > distance_thr
            n_jumps = mask_jumps.sum()

            # Stop if no jumps are detected
            if n_jumps == 0:
                break

            # Remove detected jump frames by setting specified sub-columns to NaN
            for sub in cols_to_clear:
                if (bp, sub) in df.columns:
                    df.loc[mask_jumps, (bp, sub)] = np.nan

            # Update x / y after removal (critical for iterative detection)
            x = df[(bp, 'x')]
            y = df[(bp, 'y')]

            # Stop if maximum number of iterations is reached
            if iter_count >= max_iter:
                break

    return df


def occupancy_map(x, y, t, xbins=15, ybins=15):
    x = np.asarray(x)
    y = np.asarray(y)
    t = np.asarray(t)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    x = x[valid]
    y = y[valid]
    t = t[valid]

    if len(t) < 2:
        return np.zeros((xbins, ybins))

    t_ = np.append(t, t[-1] + np.median(np.diff(t)))
    time_in_bin = np.diff(t_)
    values, _, _ = np.histogram2d(
        x, y, bins=[xbins, ybins], weights=time_in_bin
    )
    return values


def compute_distance_speed(behav_df_filtered: pd.DataFrame, window_size: int = 5, bodyparts: list = None):
    """
    Compute instantaneous speed and smoothed speed for each bodypart in the filtered DataFrame.

    Parameters:
    - behav_df_filtered_dic: Dictionary of filtered DataFrames for each video.
    - window_size: Size of the rolling window for smoothing speed.

    Returns:
    - Updated dictionary with new columns for distance and mean_speed for each bodypart.
    """
    if bodyparts is None:
        bodyparts = ['Left_ear',
                    'Right_ear',
                    'Left_fhip',
                    'Right_fhip',
                    'Left_bhip',
                    'Right_bhip',
                    'Spine1',
                    'Center',
                    'Spine2',
                    'Tail_base',
                    'Tail1',
                    'Tail2',
                    'Tail_tip']
    df_copy = behav_df_filtered.copy()
    for bp in bodyparts:
        x = df_copy[(bp, 'x')]
        y = df_copy[(bp, 'y')]
        dt = df_copy['timestamp'].diff()
        # Compute frame-to-frame distance
        dx = x.diff()
        dy = y.diff()
        distance = np.sqrt(dx**2 + dy**2)
        df_copy[(bp, 'distance')] = distance

        # Compute instantaneous speed (distance / time)
        speed = distance / dt
        df_copy[(bp, 'speed')] = speed
        # Smooth the speed trace with a centered rolling window.
        mean_speed = speed.rolling(window_size, center=True, min_periods=2).mean()
        df_copy[(bp, 'mean_speed')] = mean_speed
    return df_copy