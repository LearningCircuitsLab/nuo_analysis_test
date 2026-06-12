import numpy as np
import pandas as pd
import re
from statsmodels.tsa.stattools import acf


def observation_group(observation):
    observation = str(observation).lower()
    if "dcz" in observation:
        return "DCZ"
    if "saline" in observation:
        return "saline"
    return None


def get_injection_observation_dates(
    injection_info_df: pd.DataFrame,
    mice_selected=None,
    date_col=None,
    subject_col: str = "subject",
    observation_col: str = "observations",
):
    if date_col is None:
        date_col = injection_info_df.columns[0]

    mouse_columns = [column for column in injection_info_df.columns if column != date_col]
    if mice_selected is not None:
        mice_selected = {str(mouse) for mouse in mice_selected}
        mouse_columns = [
            column for column in mouse_columns if str(column) in mice_selected
        ]

    observation_dates = (
        injection_info_df[[date_col, *mouse_columns]]
        .melt(
            id_vars=date_col,
            value_vars=mouse_columns,
            var_name=subject_col,
            value_name=observation_col,
        )
        .dropna(subset=[date_col, observation_col])
        .copy()
    )
    observation_dates[subject_col] = observation_dates[subject_col].astype(str)
    observation_dates["date"] = pd.to_datetime(
        observation_dates[date_col],
        errors="coerce",
    )
    observation_dates["observation_group"] = observation_dates[observation_col].apply(
        observation_group
    )
    observation_dates = observation_dates.dropna(
        subset=["date", "observation_group"]
    )
    return observation_dates[
        [subject_col, "date", observation_col, "observation_group"]
    ].reset_index(drop=True)


def get_paired_injection_dates(
    injection_info_df: pd.DataFrame,
    mice_selected=None,
    date_col=None,
    subject_col: str = "subject",
    observation_col: str = "observations",
):
    observation_dates = get_injection_observation_dates(
        injection_info_df,
        mice_selected=mice_selected,
        date_col=date_col,
        subject_col=subject_col,
        observation_col=observation_col,
    )

    saline_dates = observation_dates[
        observation_dates["observation_group"] == "saline"
    ].copy()
    dcz_dates = observation_dates[
        observation_dates["observation_group"] == "DCZ"
    ].copy()

    paired_rows = []
    for _, dcz_row in dcz_dates.sort_values([subject_col, "date"]).iterrows():
        previous_saline = saline_dates[
            (saline_dates[subject_col] == dcz_row[subject_col])
            & (saline_dates["date"] < dcz_row["date"])
        ].sort_values("date")
        if previous_saline.empty:
            continue

        saline_row = previous_saline.iloc[-1]
        paired_rows.append(
            {
                subject_col: dcz_row[subject_col],
                "saline_date": saline_row["date"],
                "DCZ_date": dcz_row["date"],
                "days_between": (dcz_row["date"] - saline_row["date"]).days,
                "saline_observation": saline_row[observation_col],
                "DCZ_observation": dcz_row[observation_col],
            }
        )

    paired_dates = pd.DataFrame(paired_rows)
    if paired_dates.empty:
        return pd.DataFrame(
            columns=[
                subject_col,
                "saline_date",
                "DCZ_date",
                "days_between",
                "saline_observation",
                "DCZ_observation",
                "pair_index",
                "pair_id",
            ]
        )

    paired_dates = paired_dates.sort_values([subject_col, "DCZ_date"])
    paired_dates["pair_index"] = paired_dates.groupby(subject_col).cumcount() + 1
    paired_dates["pair_id"] = (
        paired_dates[subject_col].astype(str)
        + "_pair_"
        + paired_dates["pair_index"].astype(str).str.zfill(2)
    )
    return paired_dates.reset_index(drop=True)


def add_metric_values_to_pairs(
    observation_summary: pd.DataFrame,
    paired_dates: pd.DataFrame,
    metric_col: str,
    subject_col: str = "subject",
    date_col: str = "year_month_day",
):
    if observation_summary.empty or paired_dates.empty:
        return pd.DataFrame()

    summary = observation_summary.copy()
    summary["_date_key"] = pd.to_datetime(
        summary[date_col],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    paired_summary = paired_dates.copy()
    paired_summary[subject_col] = paired_summary[subject_col].astype(str)
    paired_summary["_saline_date_key"] = pd.to_datetime(
        paired_summary["saline_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    paired_summary["_DCZ_date_key"] = pd.to_datetime(
        paired_summary["DCZ_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    saline_values = (
        summary[summary["observation_group"] == "saline"][
            [subject_col, "_date_key", metric_col]
        ]
        .rename(columns={"_date_key": "_saline_date_key", metric_col: "saline"})
        .copy()
    )
    dcz_values = (
        summary[summary["observation_group"] == "DCZ"][
            [subject_col, "_date_key", metric_col]
        ]
        .rename(columns={"_date_key": "_DCZ_date_key", metric_col: "DCZ"})
        .copy()
    )

    paired_summary = paired_summary.merge(
        saline_values,
        on=[subject_col, "_saline_date_key"],
        how="left",
    ).merge(
        dcz_values,
        on=[subject_col, "_DCZ_date_key"],
        how="left",
    )
    paired_summary["saline"] = pd.to_numeric(
        paired_summary["saline"],
        errors="coerce",
    )
    paired_summary["DCZ"] = pd.to_numeric(
        paired_summary["DCZ"],
        errors="coerce",
    )
    paired_summary = paired_summary.dropna(subset=["saline", "DCZ"])
    return paired_summary.drop(
        columns=["_saline_date_key", "_DCZ_date_key"],
        errors="ignore",
    )


def find_behavior_key_for_pair(behavior_keys, subject, date):
    date_string = pd.to_datetime(date).strftime("%Y%m%d")
    subject = str(subject)
    matches = [
        key
        for key in behavior_keys
        if str(key).startswith(subject) and re.search(fr"_{date_string}_", str(key))
    ]
    return sorted(matches)[0] if matches else None


def split_paired_behavior_dicts(
    behav_df_dic: dict,
    paired_dates: pd.DataFrame,
    subject_col: str = "subject",
):
    behav_df_dic_saline = {}
    behav_df_dic_dcz = {}
    pair_rows = []
    missing_rows = []
    behavior_keys = list(behav_df_dic.keys())

    for _, paired_row in paired_dates.iterrows():
        subject = str(paired_row[subject_col])
        pair_id = paired_row["pair_id"]
        saline_key = find_behavior_key_for_pair(
            behavior_keys,
            subject,
            paired_row["saline_date"],
        )
        dcz_key = find_behavior_key_for_pair(
            behavior_keys,
            subject,
            paired_row["DCZ_date"],
        )

        if saline_key is None or dcz_key is None:
            missing_rows.append(
                {
                    subject_col: subject,
                    "pair_id": pair_id,
                    "saline_date": paired_row["saline_date"],
                    "DCZ_date": paired_row["DCZ_date"],
                    "saline_key": saline_key,
                    "DCZ_key": dcz_key,
                }
            )
            continue

        behav_df_dic_saline[pair_id] = behav_df_dic[saline_key]
        behav_df_dic_dcz[pair_id] = behav_df_dic[dcz_key]
        pair_rows.append(
            {
                subject_col: subject,
                "pair_id": pair_id,
                "saline_date": paired_row["saline_date"],
                "DCZ_date": paired_row["DCZ_date"],
                "days_between": paired_row["days_between"],
                "stimulus_modality": paired_row.get(
                    "stimulus_modality",
                    pd.NA,
                ),
                "saline_key": saline_key,
                "DCZ_key": dcz_key,
            }
        )

    return (
        behav_df_dic_saline,
        behav_df_dic_dcz,
        pd.DataFrame(pair_rows),
        pd.DataFrame(missing_rows),
    )


def get_behavior_column(behav_df: pd.DataFrame, column):
    if column in behav_df.columns:
        return behav_df[column]
    if isinstance(behav_df.columns, pd.MultiIndex):
        if isinstance(column, str) and (column, "") in behav_df.columns:
            return behav_df[(column, "")]
        if isinstance(column, tuple) and column[0] in behav_df.columns.get_level_values(0):
            matches = [
                existing_column
                for existing_column in behav_df.columns
                if existing_column[0] == column[0]
            ]
            if len(matches) == 1:
                return behav_df[matches[0]]
    raise KeyError(f"Column not found: {column}")


def get_roi_time_ratio(
    behav_df: pd.DataFrame,
    roi_left: float,
    roi_right: float,
    roi_bottom: float,
    roi_top: float,
    bodypart: str = "Center",
    timestamp=None,
    timestamp_col=("timestamp", ""),
):
    x = pd.to_numeric(
        get_behavior_column(behav_df, (bodypart, "x")),
        errors="coerce",
    ).to_numpy()
    y = pd.to_numeric(
        get_behavior_column(behav_df, (bodypart, "y")),
        errors="coerce",
    ).to_numpy()

    if timestamp is None:
        timestamp = get_behavior_column(behav_df, timestamp_col)
    t = pd.to_numeric(timestamp, errors="coerce").to_numpy()

    common_length = min(len(x), len(y), len(t))
    x = x[:common_length]
    y = y[:common_length]
    t = t[:common_length]

    valid_time = np.isfinite(t)
    x = x[valid_time]
    y = y[valid_time]
    t = t[valid_time]
    if len(t) < 2:
        return np.nan

    dt = np.diff(t)
    positive_dt = dt[np.isfinite(dt) & (dt > 0)]
    last_dt = np.median(positive_dt) if len(positive_dt) else 0.0
    time_per_frame = np.diff(np.append(t, t[-1] + last_dt))
    time_per_frame = np.where(np.isfinite(time_per_frame), time_per_frame, 0.0)
    time_per_frame = np.clip(time_per_frame, a_min=0, a_max=None)
    total_time = time_per_frame.sum()
    if total_time <= 0:
        return np.nan

    valid_position = np.isfinite(x) & np.isfinite(y)
    in_roi = (
        valid_position
        & (x >= roi_left)
        & (x <= roi_right)
        & (y >= roi_bottom)
        & (y <= roi_top)
    )
    roi_time = time_per_frame[in_roi].sum()
    return float(roi_time / total_time)


def paired_roi_time_ratio_comparison(
    behav_df_dic_saline: dict,
    behav_df_dic_dcz: dict,
    roi_left: float,
    roi_right: float,
    roi_bottom: float,
    roi_top: float,
    video_df_dic_saline: dict = None,
    video_df_dic_dcz: dict = None,
    pair_map: pd.DataFrame = None,
    bodypart: str = "Center",
    timestamp_col=("timestamp", ""),
):
    rows = []
    shared_pair_ids = sorted(
        set(behav_df_dic_saline).intersection(behav_df_dic_dcz)
    )

    for pair_id in shared_pair_ids:
        saline_timestamp = None
        dcz_timestamp = None
        if video_df_dic_saline is not None and pair_id in video_df_dic_saline:
            saline_timestamp = video_df_dic_saline[pair_id]["timestamp"]
        if video_df_dic_dcz is not None and pair_id in video_df_dic_dcz:
            dcz_timestamp = video_df_dic_dcz[pair_id]["timestamp"]

        saline_ratio = get_roi_time_ratio(
            behav_df_dic_saline[pair_id],
            roi_left=roi_left,
            roi_right=roi_right,
            roi_bottom=roi_bottom,
            roi_top=roi_top,
            bodypart=bodypart,
            timestamp=saline_timestamp,
            timestamp_col=timestamp_col,
        )
        dcz_ratio = get_roi_time_ratio(
            behav_df_dic_dcz[pair_id],
            roi_left=roi_left,
            roi_right=roi_right,
            roi_bottom=roi_bottom,
            roi_top=roi_top,
            bodypart=bodypart,
            timestamp=dcz_timestamp,
            timestamp_col=timestamp_col,
        )
        rows.append(
            {
                "pair_id": pair_id,
                "saline": saline_ratio,
                "DCZ": dcz_ratio,
                "DCZ_minus_saline": dcz_ratio - saline_ratio,
            }
        )

    roi_time_ratio_summary = pd.DataFrame(rows)
    if not roi_time_ratio_summary.empty:
        roi_time_ratio_summary = roi_time_ratio_summary.dropna(
            subset=["saline", "DCZ"]
        )
    if pair_map is not None and not pair_map.empty and not roi_time_ratio_summary.empty:
        roi_time_ratio_summary = pair_map.merge(
            roi_time_ratio_summary,
            on="pair_id",
            how="inner",
        )
    return roi_time_ratio_summary


def get_stationary_time_ratio(
    behav_df: pd.DataFrame,
    speed_threshold: float,
    bodypart: str = "Center",
    speed_col: str = "mean_speed",
    timestamp=None,
    timestamp_col=("timestamp", ""),
):
    speed = pd.to_numeric(
        get_behavior_column(behav_df, (bodypart, speed_col)),
        errors="coerce",
    ).to_numpy()

    if timestamp is None:
        timestamp = get_behavior_column(behav_df, timestamp_col)
    t = pd.to_numeric(timestamp, errors="coerce").to_numpy()

    common_length = min(len(speed), len(t))
    speed = speed[-common_length:]
    t = t[-common_length:]

    valid_time = np.isfinite(t)
    speed = speed[valid_time]
    t = t[valid_time]
    if len(t) < 2:
        return np.nan

    dt = np.diff(t)
    positive_dt = dt[np.isfinite(dt) & (dt > 0)]
    last_dt = np.median(positive_dt) if len(positive_dt) else 0.0
    time_per_frame = np.diff(np.append(t, t[-1] + last_dt))
    time_per_frame = np.where(np.isfinite(time_per_frame), time_per_frame, 0.0)
    time_per_frame = np.clip(time_per_frame, a_min=0, a_max=None)
    total_time = time_per_frame.sum()
    if total_time <= 0:
        return np.nan

    stationary = np.isfinite(speed) & (speed <= speed_threshold)
    stationary_time = time_per_frame[stationary].sum()
    return float(stationary_time / total_time)


def paired_stationary_time_ratio_comparison(
    behav_df_dic_saline: dict,
    behav_df_dic_dcz: dict,
    speed_threshold: float,
    video_df_dic_saline: dict = None,
    video_df_dic_dcz: dict = None,
    pair_map: pd.DataFrame = None,
    bodypart: str = "Center",
    speed_col: str = "mean_speed",
    timestamp_col=("timestamp", ""),
):
    rows = []
    shared_pair_ids = sorted(
        set(behav_df_dic_saline).intersection(behav_df_dic_dcz)
    )

    for pair_id in shared_pair_ids:
        saline_timestamp = None
        dcz_timestamp = None
        try:
            get_behavior_column(behav_df_dic_saline[pair_id], timestamp_col)
        except KeyError:
            if video_df_dic_saline is not None and pair_id in video_df_dic_saline:
                saline_timestamp = video_df_dic_saline[pair_id]["timestamp"]
        try:
            get_behavior_column(behav_df_dic_dcz[pair_id], timestamp_col)
        except KeyError:
            if video_df_dic_dcz is not None and pair_id in video_df_dic_dcz:
                dcz_timestamp = video_df_dic_dcz[pair_id]["timestamp"]

        saline_ratio = get_stationary_time_ratio(
            behav_df_dic_saline[pair_id],
            speed_threshold=speed_threshold,
            bodypart=bodypart,
            speed_col=speed_col,
            timestamp=saline_timestamp,
            timestamp_col=timestamp_col,
        )
        dcz_ratio = get_stationary_time_ratio(
            behav_df_dic_dcz[pair_id],
            speed_threshold=speed_threshold,
            bodypart=bodypart,
            speed_col=speed_col,
            timestamp=dcz_timestamp,
            timestamp_col=timestamp_col,
        )
        rows.append(
            {
                "pair_id": pair_id,
                "saline": saline_ratio,
                "DCZ": dcz_ratio,
                "DCZ_minus_saline": dcz_ratio - saline_ratio,
                "speed_threshold": speed_threshold,
            }
        )

    stationary_time_ratio_summary = pd.DataFrame(rows)
    if not stationary_time_ratio_summary.empty:
        stationary_time_ratio_summary = stationary_time_ratio_summary.dropna(
            subset=["saline", "DCZ"]
        )
    if (
        pair_map is not None
        and not pair_map.empty
        and not stationary_time_ratio_summary.empty
    ):
        stationary_time_ratio_summary = pair_map.merge(
            stationary_time_ratio_summary,
            on="pair_id",
            how="inner",
        )
    return stationary_time_ratio_summary


def get_speed_acf_auc(
    behav_df: pd.DataFrame,
    fps: float = 30,
    max_lag_sec: float = 60,
    bodypart: str = "Center",
    speed_col: str = "mean_speed",
):
    speed = pd.to_numeric(
        get_behavior_column(behav_df, (bodypart, speed_col)),
        errors="coerce",
    )
    speed = speed.replace([np.inf, -np.inf], np.nan)
    speed = speed.interpolate(limit_direction="both").dropna().to_numpy()

    nlags = min(int(fps * max_lag_sec), len(speed) - 1)
    if nlags < 1:
        return np.nan

    acf_values = acf(speed, nlags=nlags, fft=True, missing="drop")
    lag_sec = np.arange(len(acf_values)) / fps
    lag_mask = lag_sec <= max_lag_sec
    acf_values = acf_values[lag_mask]
    lag_sec = lag_sec[lag_mask]
    if len(lag_sec) < 2 or not np.isfinite(acf_values).any():
        return np.nan

    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(acf_values, lag_sec))


def paired_speed_acf_auc_comparison(
    behav_df_dic_saline: dict,
    behav_df_dic_dcz: dict,
    fps: float = 30,
    max_lag_sec: float = 60,
    pair_map: pd.DataFrame = None,
    bodypart: str = "Center",
    speed_col: str = "mean_speed",
):
    rows = []
    shared_pair_ids = sorted(
        set(behav_df_dic_saline).intersection(behav_df_dic_dcz)
    )

    for pair_id in shared_pair_ids:
        saline_auc = get_speed_acf_auc(
            behav_df_dic_saline[pair_id],
            fps=fps,
            max_lag_sec=max_lag_sec,
            bodypart=bodypart,
            speed_col=speed_col,
        )
        dcz_auc = get_speed_acf_auc(
            behav_df_dic_dcz[pair_id],
            fps=fps,
            max_lag_sec=max_lag_sec,
            bodypart=bodypart,
            speed_col=speed_col,
        )
        rows.append(
            {
                "pair_id": pair_id,
                "saline": saline_auc,
                "DCZ": dcz_auc,
                "DCZ_minus_saline": dcz_auc - saline_auc,
                "fps": fps,
                "max_lag_sec": max_lag_sec,
            }
        )

    speed_acf_auc_summary = pd.DataFrame(rows)
    if not speed_acf_auc_summary.empty:
        speed_acf_auc_summary = speed_acf_auc_summary.dropna(
            subset=["saline", "DCZ"]
        )
    if pair_map is not None and not pair_map.empty and not speed_acf_auc_summary.empty:
        speed_acf_auc_summary = pair_map.merge(
            speed_acf_auc_summary,
            on="pair_id",
            how="inner",
        )
    return speed_acf_auc_summary


def preprocess_positions(
    behav_df: pd.DataFrame,
    likelihood_thr: float = 0.8,
    distance_thr: float = 50.0,
    speed_thr: float = None,
    cols_to_clear=('x', 'y'),
    max_iter: int = 10,
    timestamp_col=("timestamp", ""),
):
    """
    Preprocess position data in a DLC-style MultiIndex DataFrame.

    Steps:
    1) Set values to NaN where likelihood < likelihood_thr.
    2) Iteratively detect sudden large jumps based on the distance to the
       most recent valid previous frame.
    3) If speed_thr is provided, detect frames whose speed to the most
       recent valid previous frame exceeds speed_thr.
    4) Set detected frames to NaN until no outliers remain or max_iter is reached.
    """
    df = behav_df.copy()
    bodyparts = df.columns.get_level_values(0).unique()
    timestamp = None
    if speed_thr is not None:
        try:
            timestamp = pd.to_numeric(
                get_behavior_column(df, timestamp_col),
                errors="coerce",
            )
        except KeyError:
            timestamp = None

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
            mask_speed = pd.Series(False, index=df.index)
            if speed_thr is not None and timestamp is not None:
                t_prev = timestamp.where(x.notna() & y.notna()).ffill().shift(1)
                dt = timestamp - t_prev
                speed = (dist / dt).where(valid & (dt > 0))
                mask_speed = speed > speed_thr

            mask_outliers = mask_jumps | mask_speed
            n_outliers = mask_outliers.sum()

            # Stop if no outliers are detected
            if n_outliers == 0:
                break

            # Remove detected jump frames by setting specified sub-columns to NaN
            for sub in cols_to_clear:
                if (bp, sub) in df.columns:
                    df.loc[mask_outliers, (bp, sub)] = np.nan

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
