import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import json
    from pathlib import Path
    import sys
    import warnings


    REPO_ROOT = Path("/home/kudongdong/code/brainCircuitsLab")
    _package_path = REPO_ROOT / "lecilab-behavior-analysis"
    if _package_path.exists() and str(_package_path) not in sys.path:
        sys.path.insert(0, str(_package_path))

    import matplotlib.cm as cm
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib as mpl
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy import stats
    from datetime import datetime
    import behavior_utils
    import plot_test
    import os

    from lecilab_behavior_analysis import utils as utils
    utils.IDIBAPS_TV_PROJECTS = "/storage/training_village/"

    from lecilab_behavior_analysis import df_transforms as dft
    from lecilab_behavior_analysis import plots
    from lecilab_behavior_analysis.figure_maker import (
        session_summary_figure,
        subject_progress_figure,
    )

    warnings.filterwarnings("ignore")
    return (
        Path,
        behavior_utils,
        mpl,
        np,
        os,
        pd,
        plot_test,
        plt,
        sns,
        stats,
        utils,
    )


@app.cell
def _(Path):
    figure_export_root = Path("/mnt/e/data/LeciLab/behavioral_data/tmp/escape")

    def _safe_figure_file_stem(text):
        safe_text = "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "_"
            for char in str(text).strip()
        )
        while "__" in safe_text:
            safe_text = safe_text.replace("__", "_")
        safe_text = safe_text.strip("_.")
        return safe_text or "figure"

    def save_figures_to_folder(
        figure_items,
        folder_name,
        file_format="svg",
        dpi=300,
    ):
        figure_folder = figure_export_root / str(folder_name)
        figure_folder.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        file_stem_counts = {}
        for title, fig in figure_items:
            file_stem = _safe_figure_file_stem(title)
            file_stem_counts[file_stem] = file_stem_counts.get(file_stem, 0) + 1
            if file_stem_counts[file_stem] > 1:
                file_stem = f"{file_stem}_{file_stem_counts[file_stem]:02d}"

            figure_path = figure_folder / f"{file_stem}.{file_format}"
            fig.savefig(
                figure_path,
                format=file_format,
                dpi=dpi,
                bbox_inches="tight",
            )
            saved_paths.append(figure_path)

        return saved_paths

    return (save_figures_to_folder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # plot_testing.ipynb in marimo
    """)
    return


@app.cell
def _(utils):
    def safe_rsync_cluster_data(project_name, file_path, credentials, local_path):
        try:
            return utils.rsync_cluster_data(
                project_name=project_name,
                file_path=file_path,
                credentials=credentials,
                local_path=local_path,
            )
        except AttributeError as exc:
            if "NoneType" not in str(exc) or "decode" not in str(exc):
                raise
            print(
                f"Could not sync {file_path}; rsync failed. "
                "See the console output above for details."
            )
            return False

    return


@app.cell
def _():
    project = 'auditory_escape_data'
    return (project,)


@app.cell
def _(utils):
    credential=utils.get_idibaps_cluster_credentials()
    return (credential,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # import data
    """)
    return


@app.cell
def _():
    # mouse_select = ['NUO060', 'NUO061', 'NUO064', 'NUO065']
    mouse_select = ['NUO001', 'NUO002', 'NUO005', 'NUO007', 'NUO008', 'NUO009', 'NUO010', 'NUO060', 'NUO061', 'NUO064', 'NUO065']
    # mouse_select = ['NUO060', 'NUO061', 'NUO064', 'NUO065', 'NUO002','NUO010']
    return (mouse_select,)


@app.cell
def _(mo):
    download_button = mo.ui.run_button(label="Download / update this mouse")
    download_button
    return (download_button,)


@app.cell
def _(credential, download_button, pd, utils):
    if download_button.value:
        utils.rsync_specific_file(
                            credentials=credential,
                            file_path="/storage/training_village/auditory_escape_data/village03/deleted_sessions.csv",
                            local_path='/home/kudongdong/data/LeciLab/behavioral_data/auditory_escape_data/village03/',
                        )
    deleted_sessions = pd.read_csv(
        "/home/kudongdong/data/LeciLab/behavioral_data/auditory_escape_data/village03/deleted_sessions.csv"
    )
    deleted_sessions_list = [d[0][:37] for d in deleted_sessions.values]
    deleted_sessions_list.append('NUO002_EscapeBehavior_20260625_134044')
    deleted_sessions_list.append('NUO002_EscapeBehavior_20260625_133935')
    deleted_sessions_list.append('NUO002_EscapeBehavior_20260625_141002')
    deleted_sessions_list.append('NUO002_EscapeBehavior_20260625_141244')
    deleted_sessions_list.append('NUO005_EscapeBehavior_20260625_141444')
    return (deleted_sessions_list,)


@app.cell
def _(
    Path,
    credential,
    deleted_sessions_list,
    download_button,
    mouse_select,
    pd,
    project,
    utils,
):
    def sync_if_missing(credentials, remote_file_path, local_dir, filename):
        local_dir = Path(local_dir)
        local_file = local_dir / filename

        if local_file.exists():
            print(f"[skip] already exists: {local_file}")
            return local_file

        utils.rsync_specific_file(
            credentials=credentials,
            file_path=remote_file_path,
            local_path=local_dir,
        )
        return local_file


    behav_df_dic = {}

    parent_path = "/home/kudongdong/data/behavior_DLC/escape_auditory-LeciLab-2025-10-20/v2/"
    session_data_path = "/storage/training_village/auditory_escape_data/sessions"
    video_data_path = "/storage/training_village/auditory_escape_data/videos"

    if download_button.value:
        behavior_files = utils.get_folders_from_server(
            credentials=credential,
            path=parent_path,
        )

    for sub in mouse_select:
        local_path = Path(utils.get_outpath()) / project / "sessions" / sub
        local_path.mkdir(parents=True, exist_ok=True)

        local_path_dlc_sub = Path(
            local_path,
            "behavioral_data",
            project,
            "DLC_output",
            sub,
        )
        local_path_session_sub = Path(
            local_path,
            "behavioral_data",
            project,
            "sessions",
            sub,
        )
        local_path_video_sub = Path(
            local_path,
            "behavioral_data",
            project,
            "videos",
            sub,
        )

        local_path_dlc_sub.mkdir(parents=True, exist_ok=True)
        local_path_session_sub.mkdir(parents=True, exist_ok=True)
        local_path_video_sub.mkdir(parents=True, exist_ok=True)

        if download_button.value:
            for f in behavior_files:
                if f.endswith(".csv") and (sub in f) and ("DLC" in f):
                    sync_if_missing(
                        credentials=credential,
                        remote_file_path=parent_path + f,
                        local_dir=local_path_dlc_sub,
                        filename=f,
                    )

        for csv_path in local_path_dlc_sub.glob("*DLC*.csv"):
            if (
                csv_path.name[:37] not in deleted_sessions_list
                and sub in csv_path.name
            ):
                random_df = pd.read_csv(csv_path, header=[1, 2])
                behav_df_dic[csv_path.name[:37]] = random_df


    analyzed_video_names = list(behav_df_dic.keys())

    session_df_dic = {}
    video_df_dic = {}

    for name in analyzed_video_names:
        subject = name[:6]

        local_path = Path(utils.get_outpath()) / project / "sessions" / subject

        local_path_session_sub = Path(
            local_path,
            "behavioral_data",
            project,
            "sessions",
            subject,
        )
        local_path_video_sub = Path(
            local_path,
            "behavioral_data",
            project,
            "videos",
            subject,
        )

        local_path_session_sub.mkdir(parents=True, exist_ok=True)
        local_path_video_sub.mkdir(parents=True, exist_ok=True)

        # ========= session csv =========
        session_file = local_path_session_sub / f"{name}.csv"

        if not session_file.exists():
            session_data_path_subject = session_data_path + "/" + subject
            session_files = utils.get_folders_from_server(
                credentials=credential,
                path=session_data_path_subject,
            )

            matched_session_files = [
                f for f in session_files
                if f.endswith(".csv") and ("RAW" not in f) and (name in f)
            ]

            if len(matched_session_files) == 0:
                raise FileNotFoundError(
                    f"No session csv found for {name} on server: "
                    f"{session_data_path_subject}"
                )

            for f in matched_session_files:
                sync_if_missing(
                    credentials=credential,
                    remote_file_path=session_data_path_subject + "/" + f,
                    local_dir=local_path_session_sub,
                    filename=f,
                )

        if not session_file.exists():
            raise FileNotFoundError(
                f"Session file still missing after sync: {session_file}"
            )

        session_df = pd.read_csv(
            session_file,
            sep=";",
        )
        session_df_dic[name] = session_df

        # ========= video csv =========
        video_file = local_path_video_sub / f"{name}.csv"

        if not video_file.exists():
            video_data_path_subject = video_data_path + "/" + subject
            video_files = utils.get_folders_from_server(
                credentials=credential,
                path=video_data_path_subject,
            )

            matched_video_files = [
                f for f in video_files
                if f.endswith(".csv") and (name in f)
            ]

            if len(matched_video_files) == 0:
                raise FileNotFoundError(
                    f"No video csv found for {name} on server: "
                    f"{video_data_path_subject}"
                )

            for f in matched_video_files:
                sync_if_missing(
                    credentials=credential,
                    remote_file_path=video_data_path_subject + "/" + f,
                    local_dir=local_path_video_sub,
                    filename=f,
                )

        if not video_file.exists():
            raise FileNotFoundError(
                f"Video file still missing after sync: {video_file}"
            )

        video_df = pd.read_csv(
            video_file,
            sep=";",
            index_col="frame",
        )
        video_df = video_df[2:]
        video_df.index = range(len(video_df))
        video_df_dic[name] = video_df
    return analyzed_video_names, behav_df_dic, session_df_dic, video_df_dic


@app.cell
def _(analyzed_video_names, behav_df_dic):
    bodyparts = behav_df_dic[analyzed_video_names[0]].columns.get_level_values(0).unique().tolist()[1:]
    return (bodyparts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # preprocess the behavior data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## set actual measure of box
    """)
    return


@app.cell
def _():
    x1 = [21, 608]
    x2 = [28, 588]
    y1 = [60, 434]
    y2 = [72, 451]
    frame_xy = [640, 480]
    x_length_pixel = ((x1[1] - x1[0]) + (x2[1] - x2[0]))/2
    y_length_pixel = ((y1[1] - y1[0]) + (y2[1] - y2[0]))/2
    x_length = 86
    y_length = 56
    x_len_tran = x_length_pixel / x_length
    y_len_tran = y_length_pixel / y_length
    trigger_zone_x_threshold = 365 / x_len_tran
    home_zone_x_threshold = 150 / x_len_tran
    return (
        frame_xy,
        home_zone_x_threshold,
        trigger_zone_x_threshold,
        x_len_tran,
        y_len_tran,
    )


@app.cell
def _(np):
    def add_timestamp_from_end(behav_df, video_df):
        behav_df[("timestamp", "")] = np.nan

        n = min(len(behav_df), len(video_df))

        behav_df.iloc[
            -n:,
            behav_df.columns.get_loc(("timestamp", "")),
        ] = video_df["timestamp"].to_numpy()[-n:]

        return behav_df

    return (add_timestamp_from_end,)


@app.cell
def _(
    add_timestamp_from_end,
    analyzed_video_names,
    behav_df_dic,
    behavior_utils,
    bodyparts,
    video_df_dic,
    x_len_tran,
    y_len_tran,
):
    behav_df_filtered_dic = {}
    for name1 in analyzed_video_names:
        df = behav_df_dic[name1]
        df['timestamp'] = video_df_dic[name1]['timestamp']
        df = behavior_utils.preprocess_positions(df, likelihood_thr=0.83, distance_thr=200, speed_thr=480, max_iter=100)
        behav_df_filtered_dic[name1] = df

    # interpolate the positions
    for name1 in analyzed_video_names:
        behav_df_filtered = behav_df_filtered_dic[name1]
        for bp in bodyparts:
            for coord in ["x", "y"]:
                behav_df_filtered[(bp, coord)] = (
                    behav_df_filtered[(bp, coord)]
                    .interpolate(method="linear", limit_direction="both")
                )
                # tranvert to actual coordinates
                behav_df_filtered[(bp, coord)] = behav_df_filtered[(bp, coord)]/x_len_tran if coord == "x" else behav_df_filtered[(bp, coord)]/y_len_tran
        behav_df_filtered_dic[name1] = behav_df_filtered

        behav_df_filtered_dic[name1] = add_timestamp_from_end(
            behav_df_filtered_dic[name1],
            video_df_dic[name1],
        )
        behav_df_filtered_dic[name1].dropna(subset=[("timestamp", "")], inplace=True)

        behav_df_filtered_dic[name1] = behavior_utils.compute_distance_speed(
            behav_df_filtered_dic[name1],
            window_size=5,
        )
    return (behav_df_filtered_dic,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## manual mouse select
    """)
    return


@app.cell
def _():
    # hM3Dq_mice = ['NUO062', 'NUO063', 'NUO064', 'NUO065', 'NUO007', 'NUO008', 'NUO009', 'NUO010', 'NUO011', 'NUO012']
    # hM4Di_mice = ['NUO057', 'NUO058', 'NUO059', 'NUO060', 'NUO061', 'NUO001', 'NUO002', 'NUO003', 'NUO005', 'NUO006']
    hM3Dq_mice = ['NUO064', 'NUO065', 'NUO007', 'NUO008', 'NUO009', 'NUO010', 'NUO011', 'NUO012']
    hM4Di_mice = ['NUO060', 'NUO061', 'NUO001', 'NUO002', 'NUO005', 'NUO006']
    return hM3Dq_mice, hM4Di_mice


@app.cell
def _(behavior_utils, mouse_select, pd):
    injection_info_file_path = "/mnt/e/data/LeciLab/behavioral_data/data_test/escape_test_record_nuo.xlsx"
    injection_info_df = pd.read_excel(injection_info_file_path)
    paired_dlc_dates = behavior_utils.get_paired_injection_dates(
        injection_info_df,
        mice_selected=mouse_select,
        saline_position = "after"
    )
    paired_dlc_dates
    return (paired_dlc_dates,)


@app.cell
def _(behav_df_filtered_dic, behavior_utils, paired_dlc_dates, video_df_dic):
    (
        behav_df_dic_saline,
        behav_df_dic_dcz,
        behav_pair_map,
        missing_behav_pairs,
    ) = behavior_utils.split_paired_behavior_dicts(behav_df_filtered_dic, paired_dlc_dates)
    (
        video_df_dic_saline,
        video_df_dic_dcz,
        video_pair_map,
        missing_video_pairs,
    ) = behavior_utils.split_paired_behavior_dicts(video_df_dic, paired_dlc_dates)
    return behav_df_dic_dcz, behav_df_dic_saline, behav_pair_map


@app.cell
def _(behav_pair_map, pd):
    def build_session_axis_lookup(behav_pair_map):
        session_axis_lookup = {}
        if behav_pair_map.empty:
            return session_axis_lookup

        pair_map = behav_pair_map.copy()
        pair_map["_DCZ_date"] = pd.to_datetime(
            pair_map["DCZ_date"],
            errors="coerce",
        )
        pair_map = pair_map.sort_values(["subject", "_DCZ_date", "pair_id"])

        for _subject, subject_pair_map in pair_map.groupby("subject"):
            for pair_index, (_, pair_row) in enumerate(
                subject_pair_map.iterrows(),
                start=1,
            ):
                for condition, key_col, offset in [
                    ("DCZ", "DCZ_key", 1),
                    ("saline", "saline_key", 2),
                ]:
                    session_key = pair_row.get(key_col)
                    if pd.isna(session_key):
                        continue
                    session_axis_lookup[session_key] = {
                        "label": f"{condition}{pair_index}",
                        "order": (pair_index - 1) * 2 + offset,
                        "condition": condition,
                        "pair_index": pair_index,
                        "pair_id": pair_row["pair_id"],
                    }

        return session_axis_lookup

    session_axis_lookup = build_session_axis_lookup(behav_pair_map)
    return (session_axis_lookup,)


@app.cell
def _(behav_pair_map, mo):

    mouse_options = sorted(behav_pair_map["subject"].unique())

    mouse_dropdown = mo.ui.dropdown(
        options=mouse_options,
        value=mouse_options[0],
        label="Mouse",
    )

    mouse_dropdown
    return (mouse_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # cut sound play and not play df
    """)
    return


@app.cell
def _(analyzed_video_names, behav_df_filtered_dic, pd, session_df_dic):
    import ast as _ast

    def get_event_times(value):
        if value is None:
            return []
        if isinstance(value, str):
            if not value.strip():
                return []
            value = _ast.literal_eval(value)
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def build_behavior_window_dataframes(mice, time_bin=15, pretimebin=0, timeupd = True):
        s_behav_df_dic = {}
        ns_behav_df_dic = {}

        for name in sorted(analyzed_video_names):
            if name[:6] not in mice or name not in session_df_dic:
                continue

            behav_df = behav_df_filtered_dic[name]
            session_df = session_df_dic[name]

            if ("timestamp", "") in behav_df.columns:
                timestamp = behav_df[("timestamp", "")]
            else:
                timestamp = behav_df["timestamp"]
            timestamp = pd.to_numeric(timestamp, errors="coerce")

            sound_by_trial = {}
            no_sound_by_trial = {}

            for _, trial_row in session_df.iterrows():
                trial = trial_row["trial"]

                if "sound_not_played" in session_df.columns:
                    no_sound_times = get_event_times(
                        trial_row["sound_not_played"]
                    )
                    for t0 in no_sound_times:
                        mask = (timestamp >= (t0 - pretimebin)) & (timestamp <= t0 + time_bin)
                        window_df = behav_df.loc[mask].copy()
                        if timeupd:
                            window_df[("timestamp", "")] = window_df[("timestamp", "")] - t0
                        if not window_df.empty:
                            window_df.attrs["trigger_time"] = float(t0)
                            no_sound_by_trial.setdefault(trial, []).append(
                                window_df
                            )

                if "sound_played" in session_df.columns:
                    sound_times = get_event_times(trial_row["sound_played"])
                    for t0 in sound_times:
                        mask = (timestamp >= (t0 - pretimebin)) & (timestamp <= t0 + time_bin)
                        window_df = behav_df.loc[mask].copy()
                        if timeupd:
                            window_df[("timestamp", "")] = window_df[("timestamp", "")] - t0
                        if not window_df.empty:
                            window_df.attrs["trigger_time"] = float(t0)
                            sound_by_trial.setdefault(trial, []).append(window_df)

            s_behav_df_dic[name] = sound_by_trial
            ns_behav_df_dic[name] = no_sound_by_trial

        return s_behav_df_dic, ns_behav_df_dic



    return (build_behavior_window_dataframes,)


@app.cell
def _(build_behavior_window_dataframes, hM3Dq_mice, hM4Di_mice):
    s_behav_df_dic_hm3, ns_behav_df_dic_hm3 = (
        build_behavior_window_dataframes(hM3Dq_mice)
    )
    s_behav_df_dic_hm4, ns_behav_df_dic_hm4 = (
        build_behavior_window_dataframes(hM4Di_mice)
    )
    return (
        ns_behav_df_dic_hm3,
        ns_behav_df_dic_hm4,
        s_behav_df_dic_hm3,
        s_behav_df_dic_hm4,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## cut sound play and not play df but with pretime window
    """)
    return


@app.cell
def _(build_behavior_window_dataframes, hM3Dq_mice, hM4Di_mice):
    s_behav_df_dic_hm3_prebin, ns_behav_df_dic_hm3_prebin = (
        build_behavior_window_dataframes(hM3Dq_mice, pretimebin=2)
    )
    s_behav_df_dic_hm4_prebin, ns_behav_df_dic_hm4_prebin = (
        build_behavior_window_dataframes(hM4Di_mice, pretimebin=2)
    )
    return (
        ns_behav_df_dic_hm3_prebin,
        ns_behav_df_dic_hm4_prebin,
        s_behav_df_dic_hm3_prebin,
        s_behav_df_dic_hm4_prebin,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # plot trajectory
    """)
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    frame_xy,
    home_zone_x_threshold,
    mo,
    mouse_dropdown,
    mpl,
    pd,
    plot_test,
    plt,
    session_df_dic,
    trigger_zone_x_threshold,
    x_len_tran,
    y_len_tran,
):
    def get_timestamp(behav_df):
        if ("timestamp", "") in behav_df.columns:
            return behav_df[("timestamp", "")]
        return behav_df["timestamp"]


    def plot_one_session_trajectory(behav_df_filtered, session_df, ax, norm):
        timestamp = get_timestamp(behav_df_filtered)

        # ========= 2) Plot trajectories =========
        # ---- no-sound windows: all trials with sound_not_played values
        if "sound_not_played" in session_df.columns:
            for _, trial_row in session_df.iterrows():
                if pd.isna(trial_row["sound_not_played"]):
                    no_sound_list = []
                else:
                    no_sound_list = eval(trial_row["sound_not_played"])

                for t0 in no_sound_list:
                    df_ns = behav_df_filtered[
                        (timestamp >= t0)
                        & (timestamp <= t0 + 15)
                    ]["Center"]

                    df_ns_inter = df_ns.copy()
                    df_ns_inter[["x", "y", "mean_speed"]] = (
                        df_ns_inter[["x", "y", "mean_speed"]]
                        .interpolate(limit_direction="both")
                        .dropna()
                    )

                    plot_test.plot_traj_speed(
                        df_ns_inter,
                        cmap="viridis",
                        ax=ax,
                        norm=norm,
                    )
                    if not df_ns_inter.empty:
                        ax.scatter(
                            df_ns_inter["x"].iloc[0],
                            df_ns_inter["y"].iloc[0],
                            color="k",
                            s=200,
                            marker="o",
                            edgecolors="k",
                            zorder=3,
                        )

        # ---- sound windows: all trials with sound_played values
        for _, trial_row in session_df.iterrows():
            if pd.isna(trial_row["sound_played"]):
                continue

            sound_play_list = eval(trial_row["sound_played"])

            for sound_play_start in sound_play_list:
                df_s = behav_df_filtered[
                    (timestamp >= sound_play_start)
                    & (timestamp <= sound_play_start + 15)
                ]["Center"]

                df_s_inter = df_s.copy()
                df_s_inter[["x", "y", "mean_speed"]] = (
                    df_s_inter[["x", "y", "mean_speed"]]
                    .interpolate(limit_direction="both")
                    .dropna()
                )

                plot_test.plot_traj_speed(
                    df_s_inter,
                    cmap="inferno",
                    ax=ax,
                    norm=norm,
                )
                if not df_s_inter.empty:
                    ax.scatter(
                        df_s_inter["x"].iloc[0],
                        df_s_inter["y"].iloc[0],
                        color="w",
                        s=200,
                        marker="o",
                        edgecolors="k",
                        zorder=3,
                    )

        # ========= 3) Axes and reference lines =========
        ax.set_xlim(0, frame_xy[0]/x_len_tran)
        ax.set_ylim(0, frame_xy[1]/y_len_tran)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.axvline(x=trigger_zone_x_threshold, c="magenta", linestyle="--", linewidth=2)
        ax.axvline(x=home_zone_x_threshold, c="green", linestyle="--", linewidth=2)


    def plot_all_pairs_for_mouse(mouse):
        mouse_pair_map = behav_pair_map[
            behav_pair_map["subject"] == mouse
        ].copy()

        pair_figures = []

        for _, pair_row in mouse_pair_map.iterrows():
            pair_id = pair_row["pair_id"]

            saline_df = behav_df_dic_saline[pair_id]
            dcz_df = behav_df_dic_dcz[pair_id]

            saline_key = pair_row["saline_key"]
            dcz_key = pair_row["DCZ_key"]

            # ========= 1) Unify the colorbar scale (global vmin/vmax) =========
            all_speed = pd.concat(
                [
                    saline_df[('Center', 'mean_speed')].dropna(),
                    dcz_df[('Center', 'mean_speed')].dropna(),
                ]
            )

            vmin = all_speed.quantile(0.01)
            vmax = all_speed.quantile(0.99)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True, sharey=True)

            plot_one_session_trajectory(
                saline_df,
                session_df_dic[saline_key],
                axes[0],
                norm,
            )
            plot_one_session_trajectory(
                dcz_df,
                session_df_dic[dcz_key],
                axes[1],
                norm,
            )

            axes[0].set_title(f"saline\n{saline_key}")
            axes[1].set_title(f"DCZ\n{dcz_key}")

            # ========= 4) Unified colorbars (two colormaps sharing the same norm) =========
            sm_inferno = mpl.cm.ScalarMappable(norm=norm, cmap="inferno")
            sm_viridis = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")

            cbar1 = plt.colorbar(
                sm_inferno,
                orientation="vertical",
                ax=axes,
                shrink=0.3,
                pad=0.02,
            )
            cbar1.set_label("speed (sound) pixels/s", rotation=90)

            cbar2 = plt.colorbar(
                sm_viridis,
                orientation="vertical",
                ax=axes,
                shrink=0.3,
                pad=0.10,
            )
            cbar2.set_label("speed (no-sound) pixels/s", rotation=90)

            fig.suptitle(pair_id)
            fig.tight_layout()

            pair_figures.append(fig)

        return pair_figures


    trajectory_pair_figures = plot_all_pairs_for_mouse(mouse_dropdown.value)

    mo.vstack(trajectory_pair_figures)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # compare mean speed by different time window
    """)
    return


@app.cell
def _(
    build_subject_day_difference,
    difference_summary_method,
    mo,
    np,
    ns_behav_df_dic_hm3,
    ns_behav_df_dic_hm4,
    pd,
    plot_ratio_boxplot,
    plot_subject_day_difference,
    ratio_dictionaries_to_long,
    s_behav_df_dic_hm3,
    s_behav_df_dic_hm4,
    save_figures_to_folder,
):
    def build_mean_speed_dictionary(behav_df_dic, time_bin):
        mean_speed_dic = {}
        for name, trial_dic in behav_df_dic.items():
            mean_speed_dic[name] = {}
            for trial, window_df_list in trial_dic.items():
                trace_means = []
                for window_df in window_df_list:
                    if ("Center", "mean_speed") not in window_df.columns:
                        raise KeyError(
                            f"{name}, trial {trial} does not contain "
                            "Center mean_speed"
                        )
                    if ("timestamp", "") in window_df.columns:
                        timestamp = window_df[("timestamp", "")]
                    else:
                        timestamp = window_df["timestamp"]
                    timestamp = pd.to_numeric(timestamp, errors="coerce")
                    trigger_time = window_df.attrs.get("trigger_time")
                    if trigger_time is None:
                        valid_timestamp = timestamp.dropna()
                        if valid_timestamp.empty:
                            continue
                        trigger_time = valid_timestamp.iloc[0]
                    time_mask = (
                        (timestamp >= 0)
                        & (timestamp <= time_bin)
                    )
                    speed_trace = pd.to_numeric(
                        window_df.loc[
                            time_mask,
                            ("Center", "mean_speed"),
                        ],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                    if np.isfinite(speed_trace).any():
                        trace_means.append(float(np.nanmean(speed_trace)))
                if trace_means:
                    mean_speed_dic[name][trial] = trace_means
        return mean_speed_dic

    mean_speed_time_bins = range(1, 16, 2)
    mean_speed_difference_figs_hm3 = {}
    mean_speed_difference_figs_hm4 = {}

    for mean_speed_time_bin in mean_speed_time_bins:
        s_mean_speed_dic_hm3 = build_mean_speed_dictionary(
            s_behav_df_dic_hm3,
            time_bin=mean_speed_time_bin,
        )
        ns_mean_speed_dic_hm3 = build_mean_speed_dictionary(
            ns_behav_df_dic_hm3,
            time_bin=mean_speed_time_bin,
        )
        s_mean_speed_dic_hm4 = build_mean_speed_dictionary(
            s_behav_df_dic_hm4,
            time_bin=mean_speed_time_bin,
        )
        ns_mean_speed_dic_hm4 = build_mean_speed_dictionary(
            ns_behav_df_dic_hm4,
            time_bin=mean_speed_time_bin,
        )

        df_mean_speed_time_bin_hm3 = ratio_dictionaries_to_long(
            ns_mean_speed_dic_hm3,
            s_mean_speed_dic_hm3,
        )
        df_mean_speed_time_bin_hm4 = ratio_dictionaries_to_long(
            ns_mean_speed_dic_hm4,
            s_mean_speed_dic_hm4,
        )

        if mean_speed_time_bin == 15:
            df_mean_speed_hm3 = df_mean_speed_time_bin_hm3
            df_mean_speed_hm4 = df_mean_speed_time_bin_hm4

        df_mean_speed_difference_hm3 = build_subject_day_difference(
            df_mean_speed_time_bin_hm3,
            summary_method=difference_summary_method,
        )
        df_mean_speed_difference_hm4 = build_subject_day_difference(
            df_mean_speed_time_bin_hm4,
            summary_method=difference_summary_method,
        )
        mean_speed_difference_figs_hm3[mean_speed_time_bin] = (
            plot_subject_day_difference(
                df_mean_speed_difference_hm3,
                f"hM3Dq: first {mean_speed_time_bin}s",
                "Mean speed: sound - no sound (pixels/s)",
                summary_method=difference_summary_method,
                y_limits=None,
                figsize=(7, 4),
            )
        )
        mean_speed_difference_figs_hm4[mean_speed_time_bin] = (
            plot_subject_day_difference(
                df_mean_speed_difference_hm4,
                f"hM4Di: first {mean_speed_time_bin}s",
                "Mean speed: sound - no sound (pixels/s)",
                summary_method=difference_summary_method,
                y_limits=None,
                figsize=(7, 4),
            )
        )

    mean_speed_fig_hm3 = plot_ratio_boxplot(
        df_mean_speed_hm3,
        "hM3Dq: first 15s",
        "Mean speed (pixels/s)",
        y_limits=None,
    )
    mean_speed_fig_hm4 = plot_ratio_boxplot(
        df_mean_speed_hm4,
        "hM4Di: first 15s",
        "Mean speed (pixels/s)",
        y_limits=None,
    )

    mean_speed_figures_to_save = [
        ("hM3Dq: first 15s", mean_speed_fig_hm3),
        ("hM4Di: first 15s", mean_speed_fig_hm4),
    ]
    for mean_speed_time_bin in mean_speed_time_bins:
        mean_speed_figures_to_save.extend(
            [
                (
                    f"hM3Dq: first {mean_speed_time_bin}s",
                    mean_speed_difference_figs_hm3[mean_speed_time_bin],
                ),
                (
                    f"hM4Di: first {mean_speed_time_bin}s",
                    mean_speed_difference_figs_hm4[mean_speed_time_bin],
                ),
            ]
        )
    mean_speed_saved_figure_paths = save_figures_to_folder(
        mean_speed_figures_to_save,
        "mean_speed_sound-nosound",
    )

    difference_figure_rows = [
        mo.hstack(
            [
                mean_speed_difference_figs_hm3[time_bin],
                mean_speed_difference_figs_hm4[time_bin],
            ]
        )
        for time_bin in mean_speed_time_bins
    ]
    mo.vstack(
        [
            mo.hstack([mean_speed_fig_hm3, mean_speed_fig_hm4]),
            *difference_figure_rows,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # speed change during trials
    """)
    return


@app.cell
def _(
    analyzed_video_names,
    behav_df_filtered_dic,
    hM3Dq_mice,
    hM4Di_mice,
    np,
    pd,
    session_df_dic,
):
    import ast as _ast

    speed_change_time_bin = 15
    pre_time_bin = 1


    def build_speed_change_dictionaries(mice, pre_time_bin=None):
        s_speed_change_dic = {}
        ns_speed_change_dic = {}

        for name in sorted(analyzed_video_names):
            if name[:6] not in mice:
                continue
            if name not in session_df_dic:
                continue

            behav_df = behav_df_filtered_dic[name]
            session_df = session_df_dic[name]

            if ("Center", "mean_speed") not in behav_df.columns:
                raise KeyError(f"{name} does not contain Center mean_speed")

            if ("timestamp", "") in behav_df.columns:
                timestamp = behav_df[("timestamp", "")]
            else:
                timestamp = behav_df["timestamp"]

            sound_speed_by_trial = {}
            no_sound_speed_by_trial = {}

            for _, trial_row in session_df.iterrows():
                if pd.isna(trial_row["trial"]):
                    continue

                trial = int(trial_row["trial"])

                # All no-sound trigger times in this trial
                if (
                    "sound_not_played" in session_df.columns
                    and pd.notna(trial_row["sound_not_played"])
                ):
                    no_sound_times = _ast.literal_eval(
                        trial_row["sound_not_played"]
                    )

                    for t0 in no_sound_times:
                        window_start = (
                            t0
                            if pre_time_bin is None
                            else t0 - pre_time_bin
                        )
                        speed = behav_df.loc[
                            (timestamp >= window_start)
                            & (timestamp <= t0 + speed_change_time_bin),
                            ("Center", "mean_speed"),
                        ]

                        speed = pd.to_numeric(
                            speed,
                            errors="coerce",
                        ).to_numpy(dtype=float)

                        if np.isfinite(speed).sum() >= 2:
                            no_sound_speed_by_trial.setdefault(
                                trial,
                                [],
                            ).append(
                                speed.tolist()
                            )

                # All sound trigger times in this trial
                if (
                    "sound_played" in session_df.columns
                    and pd.notna(trial_row["sound_played"])
                ):
                    sound_times = _ast.literal_eval(
                        trial_row["sound_played"]
                    )

                    for t0 in sound_times:
                        window_start = (
                            t0
                            if pre_time_bin is None
                            else t0 - pre_time_bin
                        )
                        speed = behav_df.loc[
                            (timestamp >= window_start)
                            & (timestamp <= t0 + speed_change_time_bin),
                            ("Center", "mean_speed"),
                        ]

                        speed = pd.to_numeric(
                            speed,
                            errors="coerce",
                        ).to_numpy(dtype=float)

                        if np.isfinite(speed).sum() >= 2:
                            sound_speed_by_trial.setdefault(
                                trial,
                                [],
                            ).append(
                                speed.tolist()
                            )

            s_speed_change_dic[name] = sound_speed_by_trial
            ns_speed_change_dic[name] = no_sound_speed_by_trial

        return s_speed_change_dic, ns_speed_change_dic


    s_speed_change_dic_hm3_prebin, ns_speed_change_dic_hm3_prebin = (
        build_speed_change_dictionaries(
            hM3Dq_mice,
            pre_time_bin=pre_time_bin,
        )
    )

    s_speed_change_dic_hm4_prebin, ns_speed_change_dic_hm4_prebin = (
        build_speed_change_dictionaries(
            hM4Di_mice,
            pre_time_bin=pre_time_bin,
        )
    )
    return (
        ns_speed_change_dic_hm3_prebin,
        ns_speed_change_dic_hm4_prebin,
        pre_time_bin,
        s_speed_change_dic_hm3_prebin,
        s_speed_change_dic_hm4_prebin,
    )


@app.cell
def _(
    mo,
    np,
    ns_speed_change_dic_hm3_prebin,
    ns_speed_change_dic_hm4_prebin,
    pd,
    plt,
    pre_time_bin,
    s_speed_change_dic_hm3_prebin,
    s_speed_change_dic_hm4_prebin,
    save_figures_to_folder,
    session_axis_lookup,
):
    # speed change trial by trial
    trial_speed_time_bin = 1
    trial_speed_fps = 30
    trial_speed_grid = np.arange(-pre_time_bin, trial_speed_time_bin, 0.01)


    def organize_mouse_day_traces(sound_dic, no_sound_dic):
        organized = {}
        axis_labels = {}
        all_names = sorted(set(sound_dic) | set(no_sound_dic))

        for mouse in sorted({name[:6] for name in all_names}):
            mouse_names = sorted(
                name for name in all_names if name[:6] == mouse
            )
            dates = sorted({name.rsplit("_", 2)[-2] for name in mouse_names})
            fallback_start = max(
                [
                    axis_info["order"]
                    for session_name, axis_info in session_axis_lookup.items()
                    if str(session_name).startswith(mouse)
                ],
                default=0,
            )
            date_to_fallback = {
                date: fallback_start + fallback_index
                for fallback_index, date in enumerate(dates, start=1)
            }

            organized[mouse] = {}
            for name in mouse_names:
                date = name.rsplit("_", 2)[-2]
                axis_info = session_axis_lookup.get(name)
                if axis_info is None:
                    day = date_to_fallback[date]
                    axis_label = f"session{day - fallback_start}"
                else:
                    day = axis_info["order"]
                    axis_label = axis_info["label"]

                axis_labels[day] = axis_label
                organized[mouse].setdefault(day, [])

                sound_by_trial = sound_dic.get(name, {})
                no_sound_by_trial = no_sound_dic.get(name, {})
                trials = sorted(set(sound_by_trial) | set(no_sound_by_trial))
                for trial in trials:
                    organized[mouse][day].append(
                        {
                            "session": name,
                            "trial": trial,
                            "sound": sound_by_trial.get(trial, []),
                            "no_sound": no_sound_by_trial.get(trial, []),
                        }
                    )

        return organized, axis_labels


    def resample_trial_speed(speed_trace):
        speed = np.asarray(speed_trace, dtype=float)
        raw_time = np.arange(len(speed)) / trial_speed_fps - pre_time_bin
        valid = np.isfinite(speed)
        if valid.sum() < 2:
            return None

        raw_time = raw_time[valid]
        speed = speed[valid]
        interpolated = np.full(trial_speed_grid.shape, np.nan)
        inside = (
            (trial_speed_grid >= raw_time[0])
            & (trial_speed_grid <= raw_time[-1])
        )
        interpolated[inside] = np.interp(
            trial_speed_grid[inside],
            raw_time,
            speed,
        )
        return interpolated


    def resample_day_trials(day_trials):
        resampled_trials = []
        for trial_data in day_trials:
            resampled_trial = {
                "session": trial_data["session"],
                "trial": trial_data["trial"],
            }
            for condition in ["sound", "no_sound"]:
                traces = [
                    resample_trial_speed(trace)
                    for trace in trial_data[condition]
                ]
                resampled_trial[condition] = [
                    trace for trace in traces if trace is not None
                ]
            resampled_trials.append(resampled_trial)
        return resampled_trials


    def mean_trial_trace(traces):
        if not traces:
            return None
        return (
            pd.DataFrame(np.vstack(traces))
            .mean(axis=0, skipna=True)
            .to_numpy()
        )


    def plot_mouse_trial_grid(
        sound_dic,
        no_sound_dic,
        group_name,
        value_label="Mean speed",
        metric_name="speed",
        y_limits=None,
        horizontal_reference=None,
        save_folder_name=None,
    ):
        organized, axis_labels = organize_mouse_day_traces(
            sound_dic,
            no_sound_dic,
        )
        mice = sorted(organized)
        all_days = sorted({
            day
            for mouse_data in organized.values()
            for day in mouse_data
        })

        if not mice or not all_days:
            empty_fig, empty_ax = plt.subplots(figsize=(6, 3))
            empty_ax.text(
                0.5,
                0.5,
                f"No trial {metric_name} data for {group_name}",
                ha="center",
                va="center",
                transform=empty_ax.transAxes,
            )
            empty_ax.set_axis_off()
            return empty_fig

        resampled_data = {}
        mini_counts = {}
        mouse_ylimits = {}
        for mouse in mice:
            resampled_data[mouse] = {}
            mouse_values = []
            for day in all_days:
                day_trials = resample_day_trials(
                    organized[mouse].get(day, [])
                )
                resampled_data[mouse][day] = day_trials
                for trial_data in day_trials:
                    mouse_values.extend(trial_data["sound"])
                    mouse_values.extend(trial_data["no_sound"])

            mini_counts[mouse] = max(
                1,
                max(
                    len(resampled_data[mouse][day])
                    for day in all_days
                ),
            )
            finite_values = np.concatenate([
                trace[np.isfinite(trace)]
                for trace in mouse_values
                if np.isfinite(trace).any()
            ]) if mouse_values else np.array([])
            if y_limits is None:
                ymax = (
                    np.nanquantile(finite_values, 0.99) * 1.05
                    if finite_values.size
                    else 1.0
                )
                mouse_ylimits[mouse] = (0, ymax)
            else:
                mouse_ylimits[mouse] = y_limits

        row_heights = [
            mini_counts[mouse] * 0.28 + 2.0 for mouse in mice
        ]
        fig = plt.figure(
            figsize=(max(8, 3.6 * len(all_days)), sum(row_heights)),
            dpi=150,
        )
        outer_grid = fig.add_gridspec(
            len(mice),
            len(all_days),
            height_ratios=row_heights,
            wspace=0.22,
            hspace=0.30,
        )

        def draw_mouse_day_panel(
            panel_fig,
            panel_grid,
            mouse,
            day,
            day_trials,
            sound_traces,
            no_sound_traces,
            n_mini,
            ymin,
            ymax,
            show_ylabel,
        ):
            for trial_index in range(n_mini):
                trial_ax = panel_fig.add_subplot(panel_grid[trial_index, 0])
                if trial_index == 0:
                    trial_ax.set_title(axis_labels[day], fontsize=10)
                if trial_index >= len(day_trials):
                    trial_ax.set_axis_off()
                    continue

                trial_data = day_trials[trial_index]
                for trace in trial_data["no_sound"]:
                    trial_ax.plot(
                        trial_speed_grid,
                        trace,
                        color="steelblue",
                        linewidth=0.8,
                    )
                for trace in trial_data["sound"]:
                    trial_ax.plot(
                        trial_speed_grid,
                        trace,
                        color="firebrick",
                        linewidth=0.8,
                    )

                trial_ax.axvline(
                    0,
                    color="black",
                    linestyle="--",
                    linewidth=0.5,
                )
                if horizontal_reference is not None:
                    trial_ax.axhline(
                        horizontal_reference,
                        color="black",
                        linestyle="--",
                        linewidth=0.5,
                    )
                trial_ax.set_xlim(-pre_time_bin, trial_speed_time_bin)
                trial_ax.set_ylim(ymin, ymax)
                trial_ax.set_xticks([])
                trial_ax.set_yticks([])
                for spine in trial_ax.spines.values():
                    spine.set_linewidth(0.4)

                if show_ylabel:
                    trial_ax.set_ylabel(
                        f"T{trial_data['trial']}",
                        rotation=0,
                        ha="right",
                        va="center",
                        fontsize=6,
                    )

            summary_ax = panel_fig.add_subplot(panel_grid[-1, 0])
            for trace in no_sound_traces:
                summary_ax.plot(
                    trial_speed_grid,
                    trace,
                    color="lightsteelblue",
                    linewidth=0.8,
                    alpha=0.25,
                )
            for trace in sound_traces:
                summary_ax.plot(
                    trial_speed_grid,
                    trace,
                    color="lightcoral",
                    linewidth=0.8,
                    alpha=0.25,
                )

            no_sound_mean = mean_trial_trace(no_sound_traces)
            sound_mean = mean_trial_trace(sound_traces)
            if no_sound_mean is not None:
                summary_ax.plot(
                    trial_speed_grid,
                    no_sound_mean,
                    color="steelblue",
                    linewidth=2.5,
                    label=f"no sound (n={len(no_sound_traces)})",
                )
            if sound_mean is not None:
                summary_ax.plot(
                    trial_speed_grid,
                    sound_mean,
                    color="firebrick",
                    linewidth=2.5,
                    label=f"sound (n={len(sound_traces)})",
                )

            summary_ax.axvline(
                0,
                color="black",
                linestyle="--",
                linewidth=1,
            )
            if horizontal_reference is not None:
                summary_ax.axhline(
                    horizontal_reference,
                    color="black",
                    linestyle="--",
                    linewidth=1,
                )
            summary_ax.set_xlim(-pre_time_bin, trial_speed_time_bin)
            summary_ax.set_ylim(ymin, ymax)
            summary_ax.set_xlabel("Time from trigger (s)", fontsize=8)
            if show_ylabel:
                summary_ax.set_ylabel(
                    f"{mouse}\n{value_label}",
                    fontsize=8,
                )
            else:
                summary_ax.set_yticklabels([])
            summary_ax.tick_params(labelsize=7)
            if sound_traces or no_sound_traces:
                summary_ax.legend(fontsize=6, frameon=False)

        def get_day_traces(mouse, day):
            day_trials = resampled_data[mouse][day]
            sound_traces = [
                trace
                for trial_data in day_trials
                for trace in trial_data["sound"]
            ]
            no_sound_traces = [
                trace
                for trial_data in day_trials
                for trace in trial_data["no_sound"]
            ]
            return day_trials, sound_traces, no_sound_traces

        for mouse_row, mouse in enumerate(mice):
            n_mini = mini_counts[mouse]
            ymin, ymax = mouse_ylimits[mouse]

            for day_col, day in enumerate(all_days):
                inner_grid = outer_grid[mouse_row, day_col].subgridspec(
                    n_mini + 1,
                    1,
                    height_ratios=[0.28] * n_mini + [2.0],
                    hspace=0.04,
                )
                day_trials, sound_traces, no_sound_traces = get_day_traces(
                    mouse,
                    day,
                )
                draw_mouse_day_panel(
                    fig,
                    inner_grid,
                    mouse,
                    day,
                    day_trials,
                    sound_traces,
                    no_sound_traces,
                    n_mini,
                    ymin,
                    ymax,
                    day_col == 0,
                )

        if save_folder_name is not None:
            for mouse in mice:
                n_mini = mini_counts[mouse]
                ymin, ymax = mouse_ylimits[mouse]
                for day in all_days:
                    day_trials, sound_traces, no_sound_traces = get_day_traces(
                        mouse,
                        day,
                    )
                    if not day_trials and not sound_traces and not no_sound_traces:
                        continue

                    single_fig = plt.figure(
                        figsize=(3.6, n_mini * 0.28 + 2.2),
                        dpi=150,
                    )
                    single_grid = single_fig.add_gridspec(
                        n_mini + 1,
                        1,
                        height_ratios=[0.28] * n_mini + [2.0],
                        hspace=0.04,
                    )
                    draw_mouse_day_panel(
                        single_fig,
                        single_grid,
                        mouse,
                        day,
                        day_trials,
                        sound_traces,
                        no_sound_traces,
                        n_mini,
                        ymin,
                        ymax,
                        True,
                    )
                    single_title = f"{group_name} {mouse} {axis_labels[day]}"
                    single_fig.suptitle(single_title, fontsize=12, y=1.02)
                    single_fig.tight_layout()
                    save_figures_to_folder(
                        [(single_title, single_fig)],
                        save_folder_name,
                    )
                    plt.close(single_fig)

        fig.suptitle(
            f"{group_name}: trial-by-trial {metric_name}",
            fontsize=14,
            y=1.0,
        )
        return fig


    speed_trial_fig_hm3 = plot_mouse_trial_grid(
        s_speed_change_dic_hm3_prebin,
        ns_speed_change_dic_hm3_prebin,
        "hM3Dq",
        value_label="Mean speed",
        metric_name="speed",
        save_folder_name="speed change by different days",
    )
    speed_trial_fig_hm4 = plot_mouse_trial_grid(
        s_speed_change_dic_hm4_prebin,
        ns_speed_change_dic_hm4_prebin,
        "hM4Di",
        value_label="Mean speed",
        metric_name="speed",
        save_folder_name="speed change by different days",
    )

    mo.vstack([speed_trial_fig_hm3, speed_trial_fig_hm4])
    return (plot_mouse_trial_grid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## first 5 trials mean speed change combine all the days
    """)
    return


@app.cell
def _(
    analyzed_video_names,
    behav_df_filtered_dic,
    hM3Dq_mice,
    hM4Di_mice,
    mo,
    np,
    pd,
    plt,
    session_df_dic,
):
    import ast as _ast

    first5_speed_pre = 1
    first5_speed_time_bin = 2
    first5_speed_dt = 0.001
    first5_speed_time_grid = np.arange(
        -first5_speed_pre,
        first5_speed_time_bin,
        first5_speed_dt,
    )

    def _first5_speed_event_times(value):
        if value is None:
            return []
        if isinstance(value, str):
            if not value.strip():
                return []
            value = _ast.literal_eval(value)
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            return list(value)
        return [value]

    def _first5_speed_timestamp(behav_df):
        if ("timestamp", "") in behav_df.columns:
            return behav_df[("timestamp", "")]
        timestamp = behav_df["timestamp"]
        if isinstance(timestamp, pd.DataFrame):
            return timestamp.iloc[:, 0]
        return timestamp

    def _first5_resample_speed_trace(behav_df, trigger_time):
        timestamp = pd.to_numeric(
            _first5_speed_timestamp(behav_df),
            errors="coerce",
        )
        window_df = behav_df[
            (timestamp >= trigger_time - first5_speed_pre)
            & (timestamp <= trigger_time + first5_speed_time_bin)
        ]
        tmp_df = (
            pd.DataFrame(
                {
                    "timestamp": _first5_speed_timestamp(window_df).to_numpy(),
                    "speed": window_df[("Center", "mean_speed")].to_numpy(),
                }
            )
            .dropna()
            .sort_values("timestamp")
            .drop_duplicates("timestamp")
        )
        if len(tmp_df) < 2:
            return None

        relative_time = tmp_df["timestamp"].to_numpy() - trigger_time
        speed = tmp_df["speed"].to_numpy()
        return np.interp(
            first5_speed_time_grid,
            relative_time,
            speed,
        )

    def _first5_speed_mean_trace(traces):
        traces = [
            np.asarray(trace, dtype=float)
            for trace in traces
            if trace is not None and np.isfinite(trace).any()
        ]
        if not traces:
            return None
        return (
            pd.DataFrame(np.vstack(traces))
            .mean(axis=0, skipna=True)
            .to_numpy()
        )

    def _first5_session_speed_traces(name):
        behav_df_filtered = behav_df_filtered_dic[name]
        session_df = session_df_dic[name]

        no_sound_traces = []
        sound_traces = []
        for _, trial_row in session_df.iterrows():
            if "sound_not_played" in session_df.columns:
                for t_ns in _first5_speed_event_times(
                    trial_row["sound_not_played"]
                ):
                    no_sound_trace = _first5_resample_speed_trace(
                        behav_df_filtered,
                        t_ns,
                    )
                    if no_sound_trace is not None:
                        no_sound_traces.append(no_sound_trace)

            if "sound_played" in session_df.columns:
                for sound_play in _first5_speed_event_times(
                    trial_row["sound_played"]
                ):
                    sound_trace = _first5_resample_speed_trace(
                        behav_df_filtered,
                        sound_play,
                    )
                    if sound_trace is not None:
                        sound_traces.append(sound_trace)

        return no_sound_traces, sound_traces

    def _first5_speed_mouse_day_data(mice):
        mouse_day_data = {}
        for mouse in sorted(mice):
            mouse_names = sorted(
                name
                for name in analyzed_video_names
                if name[:6] == mouse and name in session_df_dic
            )
            if not mouse_names:
                continue

            dates = sorted({name.rsplit("_", 2)[-2] for name in mouse_names})
            date_to_day = {
                date: day for day, date in enumerate(dates, start=1)
            }
            mouse_day_data[mouse] = []
            for name in mouse_names:
                date = name.rsplit("_", 2)[-2]
                no_sound_traces, sound_traces = _first5_session_speed_traces(
                    name
                )
                mouse_day_data[mouse].append(
                    {
                        "name": name,
                        "day": date_to_day[date],
                        "no_sound_mean": _first5_speed_mean_trace(
                            no_sound_traces
                        ),
                        "sound_traces": sound_traces,
                    }
                )

            mouse_day_data[mouse] = sorted(
                mouse_day_data[mouse],
                key=lambda day_data: day_data["day"],
            )

        return mouse_day_data

    def _first5_speed_plot_group(mouse_day_data, group_name):
        mice = sorted(mouse_day_data)
        if not mice:
            fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
            ax.text(
                0.5,
                0.5,
                f"No first-5-trial speed data for {group_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return fig

        fig, axes = plt.subplots(
            len(mice),
            5,
            figsize=(20, max(3.5, 3.2 * len(mice))),
            dpi=150,
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for mouse_row, mouse in enumerate(mice):
            day_data_list = mouse_day_data[mouse]
            day_numbers = [day_data["day"] for day_data in day_data_list]
            blue_colors = dict(
                zip(
                    day_numbers,
                    plt.get_cmap("Blues")(
                        np.linspace(0.25, 0.9, len(day_numbers))
                    ),
                )
            )
            odd_days = [day for day in day_numbers if day % 2 == 1]
            even_days = [day for day in day_numbers if day % 2 == 0]
            red_colors = dict(
                zip(
                    odd_days,
                    plt.get_cmap("Reds")(
                        np.linspace(0.35, 0.9, max(len(odd_days), 1))
                    )[: len(odd_days)],
                )
            )
            yellow_colors = dict(
                zip(
                    even_days,
                    plt.get_cmap("YlOrBr")(
                        np.linspace(0.30, 0.9, max(len(even_days), 1))
                    )[: len(even_days)],
                )
            )

            for trial_count in range(1, 6):
                ax = axes[mouse_row, trial_count - 1]
                for day_data in day_data_list:
                    day = day_data["day"]
                    no_sound_mean = day_data["no_sound_mean"]
                    if no_sound_mean is not None:
                        ax.plot(
                            first5_speed_time_grid,
                            no_sound_mean,
                            color=blue_colors[day],
                            lw=2.5,
                            alpha=0.75,
                            label=(
                                f"no sound day{day}"
                                if mouse_row == 0 and trial_count == 1
                                else None
                            ),
                        )

                    sound_mean = _first5_speed_mean_trace(
                        day_data["sound_traces"][:trial_count]
                    )
                    if sound_mean is not None:
                        sound_color = (
                            red_colors[day]
                            if day % 2 == 1
                            else yellow_colors[day]
                        )
                        ax.plot(
                            first5_speed_time_grid,
                            sound_mean,
                            color=sound_color,
                            lw=2.5,
                            alpha=0.9,
                            label=(
                                f"sound day{day}"
                                if mouse_row == 0 and trial_count == 1
                                else None
                            ),
                        )

                ax.axvline(
                    x=0,
                    c="k",
                    linestyle="--",
                    linewidth=1.5,
                )
                ax.set_xlim(-first5_speed_pre, first5_speed_time_bin)
                if mouse_row == 0:
                    ax.set_title(
                        f"sound first {trial_count} trial"
                        f"{'s' if trial_count > 1 else ''}",
                        fontsize=10,
                    )
                if trial_count == 1:
                    ax.set_ylabel(f"{mouse}\nMean speed (pixels/s)")
                if mouse_row == len(mice) - 1:
                    ax.set_xlabel("Time (s)")
                ax.grid(axis="y", alpha=0.2)

        axes[0, 0].legend(
            fontsize=6,
            frameon=False,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        fig.suptitle(
            f"{group_name}: average speed (first 5 sound trials)",
            y=1.0,
        )
        fig.tight_layout()
        return fig

    first5_speed_mouse_day_data_hm3 = _first5_speed_mouse_day_data(hM3Dq_mice)
    first5_speed_mouse_day_data_hm4 = _first5_speed_mouse_day_data(hM4Di_mice)
    first5_speed_fig_hm3 = _first5_speed_plot_group(
        first5_speed_mouse_day_data_hm3,
        "hM3Dq",
    )
    first5_speed_fig_hm4 = _first5_speed_plot_group(
        first5_speed_mouse_day_data_hm4,
        "hM4Di",
    )

    mo.vstack([first5_speed_fig_hm3, first5_speed_fig_hm4])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## compare stationary time ratio by different time window
    """)
    return


@app.cell
def _(
    behavior_utils,
    ns_behav_df_dic_hm3,
    ns_behav_df_dic_hm4,
    pd,
    s_behav_df_dic_hm3,
    s_behav_df_dic_hm4,
):
    stationary_speed_threshold = 5

    def build_stationary_time_ratio_dictionary(behav_df_dic, time_bin):
        def stationary_ratio(window_df):
            if ("timestamp", "") in window_df.columns:
                timestamp = window_df[("timestamp", "")]
            else:
                timestamp = window_df["timestamp"]
            timestamp = pd.to_numeric(timestamp, errors="coerce")
            trigger_time = window_df.attrs.get("trigger_time")
            if trigger_time is None:
                valid_timestamp = timestamp.dropna()
                if valid_timestamp.empty:
                    return float("nan")
                trigger_time = valid_timestamp.iloc[0]
            time_mask = (
                (timestamp >= 0)
                & (timestamp <= time_bin)
            )
            return behavior_utils.get_stationary_time_ratio(
                window_df.loc[time_mask],
                speed_threshold=stationary_speed_threshold,
            )

        return {
            name: {
                trial: [
                    stationary_ratio(window_df)
                    for window_df in window_df_list
                ]
                for trial, window_df_list in trial_dic.items()
            }
            for name, trial_dic in behav_df_dic.items()
        }

    stationary_time_bins = range(1, 16, 2)
    s_stationary_time_ratio_by_time_bin_hm3 = {}
    ns_stationary_time_ratio_by_time_bin_hm3 = {}
    s_stationary_time_ratio_by_time_bin_hm4 = {}
    ns_stationary_time_ratio_by_time_bin_hm4 = {}

    for _stationary_time_bin in stationary_time_bins:
        s_stationary_time_ratio_by_time_bin_hm3[_stationary_time_bin] = (
            build_stationary_time_ratio_dictionary(
                s_behav_df_dic_hm3,
                time_bin=_stationary_time_bin,
            )
        )
        ns_stationary_time_ratio_by_time_bin_hm3[_stationary_time_bin] = (
            build_stationary_time_ratio_dictionary(
                ns_behav_df_dic_hm3,
                time_bin=_stationary_time_bin,
            )
        )
        s_stationary_time_ratio_by_time_bin_hm4[_stationary_time_bin] = (
            build_stationary_time_ratio_dictionary(
                s_behav_df_dic_hm4,
                time_bin=_stationary_time_bin,
            )
        )
        ns_stationary_time_ratio_by_time_bin_hm4[_stationary_time_bin] = (
            build_stationary_time_ratio_dictionary(
                ns_behav_df_dic_hm4,
                time_bin=_stationary_time_bin,
            )
        )

    s_stationary_time_ratio_dic_hm3 = (
        s_stationary_time_ratio_by_time_bin_hm3[15]
    )
    ns_stationary_time_ratio_dic_hm3 = (
        ns_stationary_time_ratio_by_time_bin_hm3[15]
    )
    s_stationary_time_ratio_dic_hm4 = (
        s_stationary_time_ratio_by_time_bin_hm4[15]
    )
    ns_stationary_time_ratio_dic_hm4 = (
        ns_stationary_time_ratio_by_time_bin_hm4[15]
    )
    return (
        ns_stationary_time_ratio_by_time_bin_hm3,
        ns_stationary_time_ratio_by_time_bin_hm4,
        ns_stationary_time_ratio_dic_hm3,
        ns_stationary_time_ratio_dic_hm4,
        s_stationary_time_ratio_by_time_bin_hm3,
        s_stationary_time_ratio_by_time_bin_hm4,
        s_stationary_time_ratio_dic_hm3,
        s_stationary_time_ratio_dic_hm4,
        stationary_time_bins,
    )


@app.cell
def _(
    mo,
    ns_stationary_time_ratio_by_time_bin_hm3,
    ns_stationary_time_ratio_by_time_bin_hm4,
    ns_stationary_time_ratio_dic_hm3,
    ns_stationary_time_ratio_dic_hm4,
    pd,
    plt,
    s_stationary_time_ratio_by_time_bin_hm3,
    s_stationary_time_ratio_by_time_bin_hm4,
    s_stationary_time_ratio_dic_hm3,
    s_stationary_time_ratio_dic_hm4,
    save_figures_to_folder,
    session_axis_lookup,
    sns,
    stationary_time_bins,
):
    from matplotlib.lines import Line2D as _Line2D
    from matplotlib.patches import Patch as _Patch

    def ratio_dictionaries_to_long(no_sound_dic, sound_dic):
        session_names = sorted(set(no_sound_dic) | set(sound_dic))
        session_day = {}
        session_day_order = {}

        for subject in sorted({name[:6] for name in session_names}):
            subject_sessions = sorted(
                name for name in session_names if name[:6] == subject
            )
            dates = sorted(
                {name.rsplit("_", 2)[-2] for name in subject_sessions}
            )
            fallback_start = max(
                [
                    axis_info["order"]
                    for session_name, axis_info in session_axis_lookup.items()
                    if str(session_name).startswith(subject)
                ],
                default=0,
            )
            date_to_fallback = {
                date: (fallback_start + fallback_index, f"session{fallback_index}")
                for fallback_index, date in enumerate(dates, start=1)
            }
            for name in subject_sessions:
                date = name.rsplit("_", 2)[-2]
                axis_info = session_axis_lookup.get(name)
                if axis_info is None:
                    day_order, day_label = date_to_fallback[date]
                else:
                    day_order = axis_info["order"]
                    day_label = axis_info["label"]
                session_day[name] = day_label
                session_day_order[name] = day_order

        rows = []
        for condition, ratio_dic in [
            ("no_sound", no_sound_dic),
            ("sound", sound_dic),
        ]:
            for name, trial_dic in ratio_dic.items():
                for trial, ratios in trial_dic.items():
                    for event_index, ratio in enumerate(ratios):
                        rows.append(
                            {
                                "session": name,
                                "subject": name[:6],
                                "day": session_day[name],
                                "day_order": session_day_order[name],
                                "condition": condition,
                                "trial": trial,
                                "event_index": event_index,
                                "ratio": ratio,
                            }
                        )

        ratio_long_df = pd.DataFrame(rows)
        if not ratio_long_df.empty:
            ratio_long_df = ratio_long_df.dropna(subset=["ratio"])
        return ratio_long_df

    def plot_ratio_boxplot(
        ratio_long_df,
        group_name,
        ylabel,
        y_limits=(-0.05, 1.05),
    ):
        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
        if ratio_long_df.empty:
            ax.text(
                0.5,
                0.5,
                f"No ratio data for {group_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return fig

        day_order = (
            ratio_long_df[["day", "day_order"]]
            .drop_duplicates()
            .sort_values("day_order")["day"]
            .to_list()
        )
        condition_order = ["no_sound", "sound"]
        palette = {
            "no_sound": "steelblue",
            "sound": "firebrick",
        }
        mice = sorted(ratio_long_df["subject"].unique())
        markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
        mouse_markers = {
            mouse: markers[index % len(markers)]
            for index, mouse in enumerate(mice)
        }

        sns.boxplot(
            data=ratio_long_df,
            x="day",
            y="ratio",
            hue="condition",
            order=day_order,
            hue_order=condition_order,
            palette=palette,
            fill=False,
            width=0.65,
            gap=0.12,
            showfliers=False,
            ax=ax,
        )

        for mouse in mice:
            sns.stripplot(
                data=ratio_long_df[
                    ratio_long_df["subject"] == mouse
                ],
                x="day",
                y="ratio",
                hue="condition",
                order=day_order,
                hue_order=condition_order,
                palette=palette,
                dodge=True,
                jitter=0.12,
                marker=mouse_markers[mouse],
                size=6,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.4,
                legend=False,
                ax=ax,
            )

        condition_handles = [
            _Patch(
                facecolor="none",
                edgecolor=palette[condition],
                linewidth=2,
                label=condition.replace("_", " "),
            )
            for condition in condition_order
        ]
        condition_legend = ax.legend(
            handles=condition_handles,
            title="Condition",
            frameon=False,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        ax.add_artist(condition_legend)

        mouse_handles = [
            _Line2D(
                [0],
                [0],
                marker=mouse_markers[mouse],
                linestyle="none",
                markerfacecolor="gray",
                markeredgecolor="black",
                markersize=7,
                label=mouse,
            )
            for mouse in mice
        ]
        ax.legend(
            handles=mouse_handles,
            title="Mouse",
            frameon=False,
            bbox_to_anchor=(1.02, 0.58),
            loc="upper left",
        )

        ax.set_xlabel("Injection")
        ax.set_ylabel(ylabel)
        ax.set_title(group_name)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        return fig

    def build_subject_day_difference(
        ratio_long_df,
        summary_method="mean",
    ):
        if summary_method not in {"mean", "median"}:
            raise ValueError(
                "summary_method must be either 'mean' or 'median'"
            )
        if ratio_long_df.empty:
            return pd.DataFrame(
                columns=[
                    "subject",
                    "day",
                    "day_order",
                    "no_sound",
                    "sound",
                    "difference",
                    "day_number",
                ]
            )

        condition_summary = (
            ratio_long_df.groupby(
                ["subject", "day", "day_order", "condition"],
                as_index=False,
            )["ratio"]
            .agg(summary_method)
            .pivot(
                index=["subject", "day", "day_order"],
                columns="condition",
                values="ratio",
            )
            .reset_index()
        )
        if not {"sound", "no_sound"}.issubset(condition_summary.columns):
            return pd.DataFrame(
                columns=[
                    "subject",
                    "day",
                    "day_order",
                    "no_sound",
                    "sound",
                    "difference",
                    "day_number",
                ]
            )

        condition_summary = condition_summary.dropna(
            subset=["sound", "no_sound"]
        )
        condition_summary["difference"] = (
            condition_summary["sound"] - condition_summary["no_sound"]
        )
        condition_summary["day_number"] = (
            condition_summary["day_order"].astype(int)
        )
        return condition_summary

    def plot_subject_day_difference(
        difference_df,
        group_name,
        ylabel,
        summary_method="mean",
        y_limits=(-1.05, 1.05),
        figsize=(8, 5),
    ):
        if summary_method not in {"mean", "median"}:
            raise ValueError(
                "summary_method must be either 'mean' or 'median'"
            )
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        if difference_df.empty:
            ax.text(
                0.5,
                0.5,
                f"No paired sound/no-sound data for {group_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return fig

        mice = sorted(difference_df["subject"].unique())
        markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
        colors = sns.color_palette("tab10", n_colors=len(mice))

        for index, mouse in enumerate(mice):
            mouse_df = difference_df[
                difference_df["subject"] == mouse
            ].sort_values("day_number")
            ax.plot(
                mouse_df["day_number"],
                mouse_df["difference"],
                color=colors[index],
                marker=markers[index % len(markers)],
                markeredgecolor="black",
                linewidth=1.8,
                markersize=7,
                alpha=0.8,
                label=mouse,
            )

        daily_summary = (
            difference_df.groupby("day_number", as_index=False)["difference"]
            .agg(summary_method)
            .sort_values("day_number")
        )
        ax.plot(
            daily_summary["day_number"],
            daily_summary["difference"],
            color="black",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            linewidth=3.5,
            markersize=8,
            label=f"{summary_method.capitalize()} across mice",
            zorder=5,
        )

        day_ticks = (
            difference_df[["day_number", "day"]]
            .drop_duplicates()
            .sort_values("day_number")
        )
        ax.set_xticks(day_ticks["day_number"])
        ax.set_xticklabels(day_ticks["day"])
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.set_xlabel("Injection")
        ax.set_ylabel(ylabel)
        ax.set_title(group_name)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(
            title="Mouse",
            frameon=False,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        fig.tight_layout()
        return fig

    df_stationary_time_ratio_hm3 = ratio_dictionaries_to_long(
        ns_stationary_time_ratio_dic_hm3,
        s_stationary_time_ratio_dic_hm3,
    )
    df_stationary_time_ratio_hm4 = ratio_dictionaries_to_long(
        ns_stationary_time_ratio_dic_hm4,
        s_stationary_time_ratio_dic_hm4,
    )

    stationary_time_ratio_fig_hm3 = plot_ratio_boxplot(
        df_stationary_time_ratio_hm3,
        "hM3Dq: first 15s",
        "Stationary time ratio",
    )
    stationary_time_ratio_fig_hm4 = plot_ratio_boxplot(
        df_stationary_time_ratio_hm4,
        "hM4Di: first 15s",
        "Stationary time ratio",
    )

    difference_summary_method = "mean"  # Change to "median" if needed.

    stationary_time_ratio_difference_figs_hm3 = {}
    stationary_time_ratio_difference_figs_hm4 = {}
    for _stationary_time_bin in stationary_time_bins:
        df_stationary_time_bin_hm3 = ratio_dictionaries_to_long(
            ns_stationary_time_ratio_by_time_bin_hm3[_stationary_time_bin],
            s_stationary_time_ratio_by_time_bin_hm3[_stationary_time_bin],
        )
        df_stationary_time_bin_hm4 = ratio_dictionaries_to_long(
            ns_stationary_time_ratio_by_time_bin_hm4[_stationary_time_bin],
            s_stationary_time_ratio_by_time_bin_hm4[_stationary_time_bin],
        )
        df_stationary_difference_hm3 = build_subject_day_difference(
            df_stationary_time_bin_hm3,
            summary_method=difference_summary_method,
        )
        df_stationary_difference_hm4 = build_subject_day_difference(
            df_stationary_time_bin_hm4,
            summary_method=difference_summary_method,
        )
        stationary_time_ratio_difference_figs_hm3[_stationary_time_bin] = (
            plot_subject_day_difference(
                df_stationary_difference_hm3,
                f"hM3Dq: first {_stationary_time_bin}s",
                "Stationary time ratio: sound - no sound",
                summary_method=difference_summary_method,
                figsize=(7, 4),
            )
        )
        stationary_time_ratio_difference_figs_hm4[_stationary_time_bin] = (
            plot_subject_day_difference(
                df_stationary_difference_hm4,
                f"hM4Di: first {_stationary_time_bin}s",
                "Stationary time ratio: sound - no sound",
                summary_method=difference_summary_method,
                figsize=(7, 4),
            )
        )

    stationary_difference_figure_rows = [
        mo.hstack(
            [
                stationary_time_ratio_difference_figs_hm3[time_bin],
                stationary_time_ratio_difference_figs_hm4[time_bin],
            ]
        )
        for time_bin in stationary_time_bins
    ]

    stationary_time_ratio_figures_to_save = [
        ("hM3Dq: first 15s", stationary_time_ratio_fig_hm3),
        ("hM4Di: first 15s", stationary_time_ratio_fig_hm4),
    ]
    for _stationary_time_bin in stationary_time_bins:
        stationary_time_ratio_figures_to_save.extend(
            [
                (
                    f"hM3Dq: first {_stationary_time_bin}s",
                    stationary_time_ratio_difference_figs_hm3[
                        _stationary_time_bin
                    ],
                ),
                (
                    f"hM4Di: first {_stationary_time_bin}s",
                    stationary_time_ratio_difference_figs_hm4[
                        _stationary_time_bin
                    ],
                ),
            ]
        )
    stationary_time_ratio_saved_figure_paths = save_figures_to_folder(
        stationary_time_ratio_figures_to_save,
        "stationary_time_ratio_sound-nosound",
    )

    mo.vstack(
        [
            mo.hstack(
                [stationary_time_ratio_fig_hm3, stationary_time_ratio_fig_hm4]
            ),
            *stationary_difference_figure_rows,
        ]
    )
    return (
        build_subject_day_difference,
        difference_summary_method,
        plot_ratio_boxplot,
        plot_subject_day_difference,
        ratio_dictionaries_to_long,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # compare trial number
    """)
    return


@app.cell
def _(
    mo,
    ns_speed_change_dic_hm3_prebin,
    ns_speed_change_dic_hm4_prebin,
    pd,
    plt,
    s_speed_change_dic_hm3_prebin,
    s_speed_change_dic_hm4_prebin,
    session_axis_lookup,
):
    from matplotlib.lines import Line2D as _Line2D
    from matplotlib.ticker import MaxNLocator as _MaxNLocator


    def build_trial_count_table(sound_dic, no_sound_dic):
        rows = []
        all_names = sorted(set(sound_dic) | set(no_sound_dic))

        for mouse in sorted({name[:6] for name in all_names}):
            mouse_names = sorted(
                name for name in all_names if name[:6] == mouse
            )
            dates = sorted({name.rsplit("_", 2)[-2] for name in mouse_names})
            fallback_start = max(
                [
                    axis_info["order"]
                    for session_name, axis_info in session_axis_lookup.items()
                    if str(session_name).startswith(mouse)
                ],
                default=0,
            )
            date_to_fallback = {
                date: (fallback_start + fallback_index, f"session{fallback_index}")
                for fallback_index, date in enumerate(dates, start=1)
            }

            for name in mouse_names:
                date = name.rsplit("_", 2)[-2]
                axis_info = session_axis_lookup.get(name)
                if axis_info is None:
                    day_order, day_label = date_to_fallback[date]
                else:
                    day_order = axis_info["order"]
                    day_label = axis_info["label"]
                sound_count = len(sound_dic.get(name, {}))
                no_sound_count = len(no_sound_dic.get(name, {}))
                rows.append(
                    {
                        "subject": mouse,
                        "day": day_label,
                        "day_order": day_order,
                        "sound": sound_count,
                        "no_sound": no_sound_count,
                        "total": sound_count + no_sound_count,
                    }
                )

        return pd.DataFrame(rows)


    def plot_trial_counts(trial_count_df, group_name):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        if trial_count_df.empty:
            ax.text(
                0.5,
                0.5,
                f"No trial-count data for {group_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return fig

        markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
        mice = sorted(trial_count_df["subject"].unique())
        mouse_markers = {
            mouse: markers[index % len(markers)]
            for index, mouse in enumerate(mice)
        }
        conditions = [
            ("sound", "firebrick", "sound"),
            ("no_sound", "steelblue", "no sound"),
            ("total", "black", "sound + no sound"),
        ]

        for mouse in mice:
            mouse_df = trial_count_df[
                trial_count_df["subject"] == mouse
            ].sort_values("day_order")
            for column, color, _ in conditions:
                ax.plot(
                    mouse_df["day_order"],
                    mouse_df[column],
                    color=color,
                    marker=mouse_markers[mouse],
                    markerfacecolor=color,
                    markeredgecolor="black",
                    linewidth=1.8,
                    markersize=7,
                )

        condition_handles = [
            _Line2D([0], [0], color=color, linewidth=2, label=label)
            for _, color, label in conditions
        ]
        condition_legend = ax.legend(
            handles=condition_handles,
            title="Trial type",
            frameon=False,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        ax.add_artist(condition_legend)

        mouse_handles = [
            _Line2D(
                [0],
                [0],
                color="gray",
                marker=mouse_markers[mouse],
                markerfacecolor="gray",
                markeredgecolor="black",
                linestyle="none",
                markersize=7,
                label=mouse,
            )
            for mouse in mice
        ]
        ax.legend(
            handles=mouse_handles,
            title="Mouse",
            frameon=False,
            bbox_to_anchor=(1.02, 0.58),
            loc="upper left",
        )

        day_ticks = (
            trial_count_df[["day_order", "day"]]
            .drop_duplicates()
            .sort_values("day_order")
        )
        ax.set_xticks(day_ticks["day_order"])
        ax.set_xticklabels(day_ticks["day"])
        ax.yaxis.set_major_locator(_MaxNLocator(integer=True))
        ax.set_xlabel("Injection")
        ax.set_ylabel("Number of trials")
        ax.set_title(group_name)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig


    trial_count_df_hm3 = build_trial_count_table(
        s_speed_change_dic_hm3_prebin,
        ns_speed_change_dic_hm3_prebin,
    )
    trial_count_df_hm4 = build_trial_count_table(
        s_speed_change_dic_hm4_prebin,
        ns_speed_change_dic_hm4_prebin,
    )
    trial_count_fig_hm3 = plot_trial_counts(
        trial_count_df_hm3,
        "hM3Dq",
    )
    trial_count_fig_hm4 = plot_trial_counts(
        trial_count_df_hm4,
        "hM4Di",
    )

    mo.vstack([trial_count_fig_hm3, trial_count_fig_hm4])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # x position change during trials
    """)
    return


@app.cell
def _(
    analyzed_video_names,
    behav_df_filtered_dic,
    hM3Dq_mice,
    hM4Di_mice,
    np,
    pd,
    pre_time_bin,
    session_df_dic,
):
    import ast as _ast

    x_position_time_bin = 11


    def build_x_position_dictionaries(mice, pre_time_bin=None):
        s_x_position_dic = {}
        ns_x_position_dic = {}

        for name in sorted(analyzed_video_names):
            if name[:6] not in mice or name not in session_df_dic:
                continue

            behav_df = behav_df_filtered_dic[name]
            session_df = session_df_dic[name]

            if ("Center", "x") not in behav_df.columns:
                raise KeyError(f"{name} does not contain Center x position")

            if ("timestamp", "") in behav_df.columns:
                timestamp = behav_df[("timestamp", "")]
            else:
                timestamp = behav_df["timestamp"]

            sound_x_by_trial = {}
            no_sound_x_by_trial = {}

            for _, trial_row in session_df.iterrows():
                if pd.isna(trial_row["trial"]):
                    continue

                trial = int(trial_row["trial"])

                if (
                    "sound_not_played" in session_df.columns
                    and pd.notna(trial_row["sound_not_played"])
                ):
                    no_sound_times = _ast.literal_eval(
                        trial_row["sound_not_played"]
                    )
                    for t0 in no_sound_times:
                        window_start = (
                            t0
                            if pre_time_bin is None
                            else t0 - pre_time_bin
                        )
                        x_position = behav_df.loc[
                            (timestamp >= window_start)
                            & (timestamp <= t0 + x_position_time_bin),
                            ("Center", "x"),
                        ]
                        x_position = pd.to_numeric(
                            x_position,
                            errors="coerce",
                        ).to_numpy(dtype=float)
                        if np.isfinite(x_position).sum() >= 2:
                            no_sound_x_by_trial.setdefault(
                                trial,
                                [],
                            ).append(
                                x_position.tolist()
                            )

                if (
                    "sound_played" in session_df.columns
                    and pd.notna(trial_row["sound_played"])
                ):
                    sound_times = _ast.literal_eval(
                        trial_row["sound_played"]
                    )
                    for t0 in sound_times:
                        window_start = (
                            t0
                            if pre_time_bin is None
                            else t0 - pre_time_bin
                        )
                        x_position = behav_df.loc[
                            (timestamp >= window_start)
                            & (timestamp <= t0 + x_position_time_bin),
                            ("Center", "x"),
                        ]
                        x_position = pd.to_numeric(
                            x_position,
                            errors="coerce",
                        ).to_numpy(dtype=float)
                        if np.isfinite(x_position).sum() >= 2:
                            sound_x_by_trial.setdefault(
                                trial,
                                [],
                            ).append(
                                x_position.tolist()
                            )

            s_x_position_dic[name] = sound_x_by_trial
            ns_x_position_dic[name] = no_sound_x_by_trial

        return s_x_position_dic, ns_x_position_dic


    s_x_position_dic_hm3_prebin, ns_x_position_dic_hm3_prebin = (
        build_x_position_dictionaries(
            hM3Dq_mice,
            pre_time_bin=pre_time_bin,
        )
    )
    s_x_position_dic_hm4_prebin, ns_x_position_dic_hm4_prebin = (
        build_x_position_dictionaries(
            hM4Di_mice,
            pre_time_bin=pre_time_bin,
        )
    )
    return (
        ns_x_position_dic_hm3_prebin,
        ns_x_position_dic_hm4_prebin,
        s_x_position_dic_hm3_prebin,
        s_x_position_dic_hm4_prebin,
    )


@app.cell
def _(
    frame_xy,
    mo,
    ns_x_position_dic_hm3_prebin,
    ns_x_position_dic_hm4_prebin,
    plot_mouse_trial_grid,
    s_x_position_dic_hm3_prebin,
    s_x_position_dic_hm4_prebin,
    trigger_zone_x_threshold,
    x_len_tran,
):
    x_position_trial_fig_hm3 = plot_mouse_trial_grid(
        s_x_position_dic_hm3_prebin,
        ns_x_position_dic_hm3_prebin,
        "hM3Dq",
        value_label="X position (pixels)",
        metric_name="x position",
        y_limits=(0, frame_xy[0]/x_len_tran),
        horizontal_reference=trigger_zone_x_threshold,
        save_folder_name="x_position change by different days",
    )
    x_position_trial_fig_hm4 = plot_mouse_trial_grid(
        s_x_position_dic_hm4_prebin,
        ns_x_position_dic_hm4_prebin,
        "hM4Di",
        value_label="X position (pixels)",
        metric_name="x position",
        y_limits=(0, frame_xy[0]/x_len_tran),
        horizontal_reference=trigger_zone_x_threshold,
        save_folder_name="x_position change by different days",
    )

    mo.vstack([x_position_trial_fig_hm3, x_position_trial_fig_hm4])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## first 5 trials mean position change combine all the days
    """)
    return


@app.cell
def _(
    analyzed_video_names,
    behav_df_filtered_dic,
    frame_xy,
    hM3Dq_mice,
    hM4Di_mice,
    mo,
    np,
    pd,
    plt,
    session_df_dic,
    trigger_zone_x_threshold,
    x_len_tran,
):
    import ast as _ast

    first5_position_pre = 2
    first5_position_time_bin = 15
    first5_position_dt = 0.001
    first5_position_time_grid = np.arange(
        -first5_position_pre,
        first5_position_time_bin,
        first5_position_dt,
    )

    def _first5_event_times(value):
        if value is None:
            return []
        if isinstance(value, str):
            if not value.strip():
                return []
            value = _ast.literal_eval(value)
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            return list(value)
        return [value]

    def _first5_timestamp(behav_df):
        if ("timestamp", "") in behav_df.columns:
            return behav_df[("timestamp", "")]
        timestamp = behav_df["timestamp"]
        if isinstance(timestamp, pd.DataFrame):
            return timestamp.iloc[:, 0]
        return timestamp

    def _first5_resample_x_trace(behav_df, trigger_time):
        timestamp = pd.to_numeric(
            _first5_timestamp(behav_df),
            errors="coerce",
        )
        window_df = behav_df[
            (timestamp >= trigger_time - first5_position_pre)
            & (timestamp <= trigger_time + first5_position_time_bin)
        ]
        tmp_df = (
            pd.DataFrame(
                {
                    "timestamp": _first5_timestamp(window_df).to_numpy(),
                    "x": window_df[("Center", "x")].to_numpy(),
                }
            )
            .dropna()
            .sort_values("timestamp")
            .drop_duplicates("timestamp")
        )
        if len(tmp_df) < 2:
            return None

        relative_time = tmp_df["timestamp"].to_numpy() - trigger_time
        x_position = tmp_df["x"].to_numpy()
        return np.interp(
            first5_position_time_grid,
            relative_time,
            x_position,
        )

    def _first5_mean_trace(traces):
        traces = [
            np.asarray(trace, dtype=float)
            for trace in traces
            if trace is not None and np.isfinite(trace).any()
        ]
        if not traces:
            return None
        return (
            pd.DataFrame(np.vstack(traces))
            .mean(axis=0, skipna=True)
            .to_numpy()
        )

    def _first5_session_x_traces(name):
        behav_df_filtered = behav_df_filtered_dic[name]
        session_df = session_df_dic[name]

        no_sound_traces = []
        sound_traces = []
        for _, trial_row in session_df.iterrows():
            if "sound_not_played" in session_df.columns:
                for t_ns in _first5_event_times(trial_row["sound_not_played"]):
                    no_sound_trace = _first5_resample_x_trace(
                        behav_df_filtered,
                        t_ns,
                    )
                    if no_sound_trace is not None:
                        no_sound_traces.append(no_sound_trace)

            if "sound_played" in session_df.columns:
                for sound_play in _first5_event_times(trial_row["sound_played"]):
                    sound_trace = _first5_resample_x_trace(
                        behav_df_filtered,
                        sound_play,
                    )
                    if sound_trace is not None:
                        sound_traces.append(sound_trace)

        return no_sound_traces, sound_traces

    def _first5_mouse_day_data(mice):
        mouse_day_data = {}
        for mouse in sorted(mice):
            mouse_names = sorted(
                name
                for name in analyzed_video_names
                if name[:6] == mouse and name in session_df_dic
            )
            if not mouse_names:
                continue

            dates = sorted({name.rsplit("_", 2)[-2] for name in mouse_names})
            date_to_day = {
                date: day for day, date in enumerate(dates, start=1)
            }
            mouse_day_data[mouse] = []
            for name in mouse_names:
                date = name.rsplit("_", 2)[-2]
                no_sound_traces, sound_traces = _first5_session_x_traces(name)
                mouse_day_data[mouse].append(
                    {
                        "name": name,
                        "day": date_to_day[date],
                        "no_sound_mean": _first5_mean_trace(no_sound_traces),
                        "sound_traces": sound_traces,
                    }
                )

            mouse_day_data[mouse] = sorted(
                mouse_day_data[mouse],
                key=lambda day_data: day_data["day"],
            )

        return mouse_day_data

    def _first5_plot_group(mouse_day_data, group_name):
        mice = sorted(mouse_day_data)
        if not mice:
            fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
            ax.text(
                0.5,
                0.5,
                f"No first-5-trial x-position data for {group_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return fig

        fig, axes = plt.subplots(
            len(mice),
            5,
            figsize=(20, max(3.5, 3.2 * len(mice))),
            dpi=150,
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for mouse_row, mouse in enumerate(mice):
            day_data_list = mouse_day_data[mouse]
            day_numbers = [day_data["day"] for day_data in day_data_list]
            blue_colors = dict(
                zip(
                    day_numbers,
                    plt.get_cmap("Blues")(
                        np.linspace(0.25, 0.9, len(day_numbers))
                    ),
                )
            )
            odd_days = [day for day in day_numbers if day % 2 == 1]
            even_days = [day for day in day_numbers if day % 2 == 0]
            red_colors = dict(
                zip(
                    odd_days,
                    plt.get_cmap("Reds")(
                        np.linspace(0.35, 0.9, max(len(odd_days), 1))
                    )[: len(odd_days)],
                )
            )
            yellow_colors = dict(
                zip(
                    even_days,
                    plt.get_cmap("YlOrBr")(
                        np.linspace(0.30, 0.9, max(len(even_days), 1))
                    )[: len(even_days)],
                )
            )

            for trial_count in range(1, 6):
                ax = axes[mouse_row, trial_count - 1]
                for day_data in day_data_list:
                    day = day_data["day"]
                    no_sound_mean = day_data["no_sound_mean"]
                    if no_sound_mean is not None:
                        ax.plot(
                            first5_position_time_grid,
                            no_sound_mean,
                            color=blue_colors[day],
                            lw=2.5,
                            alpha=0.75,
                            label=(
                                f"no sound day{day}"
                                if mouse_row == 0 and trial_count == 1
                                else None
                            ),
                        )

                    sound_mean = _first5_mean_trace(
                        day_data["sound_traces"][:trial_count]
                    )
                    if sound_mean is not None:
                        sound_color = (
                            red_colors[day]
                            if day % 2 == 1
                            else yellow_colors[day]
                        )
                        ax.plot(
                            first5_position_time_grid,
                            sound_mean,
                            color=sound_color,
                            lw=2.5,
                            alpha=0.9,
                            label=(
                                f"sound day{day}"
                                if mouse_row == 0 and trial_count == 1
                                else None
                            ),
                        )

                ax.axhline(
                    y=trigger_zone_x_threshold,
                    c="magenta",
                    linestyle="dotted",
                    linewidth=2,
                )
                ax.axvline(
                    x=0,
                    c="k",
                    linestyle="--",
                    linewidth=1.5,
                )
                ax.set_xlim(-first5_position_pre, first5_position_time_bin)
                ax.set_ylim(0, frame_xy[0]/x_len_tran)
                if mouse_row == 0:
                    ax.set_title(
                        f"sound first {trial_count} trial"
                        f"{'s' if trial_count > 1 else ''}",
                        fontsize=10,
                    )
                if trial_count == 1:
                    ax.set_ylabel(f"{mouse}\nX position (pixels)")
                if mouse_row == len(mice) - 1:
                    ax.set_xlabel("Time (s)")
                ax.grid(axis="y", alpha=0.2)

        axes[0, 0].legend(
            fontsize=6,
            frameon=False,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        fig.suptitle(
            f"{group_name}: average position (first 5 sound trials)",
            y=1.0,
        )
        fig.tight_layout()
        return fig

    first5_position_mouse_day_data_hm3 = _first5_mouse_day_data(hM3Dq_mice)
    first5_position_mouse_day_data_hm4 = _first5_mouse_day_data(hM4Di_mice)
    first5_position_fig_hm3 = _first5_plot_group(
        first5_position_mouse_day_data_hm3,
        "hM3Dq",
    )
    first5_position_fig_hm4 = _first5_plot_group(
        first5_position_mouse_day_data_hm4,
        "hM4Di",
    )

    mo.vstack([first5_position_fig_hm3, first5_position_fig_hm4])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## compare trigger zone time ratio by different time window
    """)
    return


@app.cell
def _(
    np,
    ns_behav_df_dic_hm3,
    ns_behav_df_dic_hm4,
    pd,
    s_behav_df_dic_hm3,
    s_behav_df_dic_hm4,
    trigger_zone_x_threshold,
):


    def get_trigger_zone_time_ratio(behav_df, time_bin):
        if ("timestamp", "") in behav_df.columns:
            timestamp = behav_df[("timestamp", "")]
        else:
            timestamp = behav_df["timestamp"]
        timestamp = pd.to_numeric(timestamp, errors="coerce")
        trigger_time = behav_df.attrs.get("trigger_time")
        if trigger_time is None:
            valid_timestamp = timestamp.dropna()
            if valid_timestamp.empty:
                return np.nan
            trigger_time = valid_timestamp.iloc[0]
        time_mask = (
            (timestamp >= 0)
            & (timestamp <= time_bin)
        )
        x = pd.to_numeric(
            behav_df.loc[time_mask, ("Center", "x")],
            errors="coerce",
        ).to_numpy()
        timestamp = timestamp.loc[time_mask].to_numpy()

        common_length = min(len(x), len(timestamp))
        x = x[:common_length]
        timestamp = timestamp[:common_length]

        valid_time = np.isfinite(timestamp)
        x = x[valid_time]
        timestamp = timestamp[valid_time]
        if len(timestamp) < 2:
            return np.nan

        dt = np.diff(timestamp)
        positive_dt = dt[np.isfinite(dt) & (dt > 0)]
        last_dt = np.median(positive_dt) if len(positive_dt) else 0.0
        time_per_frame = np.diff(
            np.append(timestamp, timestamp[-1] + last_dt)
        )
        time_per_frame = np.where(
            np.isfinite(time_per_frame),
            time_per_frame,
            0.0,
        )
        time_per_frame = np.clip(time_per_frame, a_min=0, a_max=None)
        total_time = time_per_frame.sum()
        if total_time <= 0:
            return np.nan

        in_trigger_zone = np.isfinite(x) & (x > trigger_zone_x_threshold)
        return float(time_per_frame[in_trigger_zone].sum() / total_time)

    def build_trigger_zone_time_ratio_dictionary(behav_df_dic, time_bin):
        return {
            name: {
                trial: [
                    get_trigger_zone_time_ratio(window_df, time_bin=time_bin)
                    for window_df in window_df_list
                ]
                for trial, window_df_list in trial_dic.items()
            }
            for name, trial_dic in behav_df_dic.items()
        }

    trigger_zone_time_bins = range(1, 16, 2)
    s_trigger_zone_time_ratio_by_time_bin_hm3 = {}
    ns_trigger_zone_time_ratio_by_time_bin_hm3 = {}
    s_trigger_zone_time_ratio_by_time_bin_hm4 = {}
    ns_trigger_zone_time_ratio_by_time_bin_hm4 = {}

    for _trigger_zone_time_bin in trigger_zone_time_bins:
        s_trigger_zone_time_ratio_by_time_bin_hm3[_trigger_zone_time_bin] = (
            build_trigger_zone_time_ratio_dictionary(
                s_behav_df_dic_hm3,
                time_bin=_trigger_zone_time_bin,
            )
        )
        ns_trigger_zone_time_ratio_by_time_bin_hm3[_trigger_zone_time_bin] = (
            build_trigger_zone_time_ratio_dictionary(
                ns_behav_df_dic_hm3,
                time_bin=_trigger_zone_time_bin,
            )
        )
        s_trigger_zone_time_ratio_by_time_bin_hm4[_trigger_zone_time_bin] = (
            build_trigger_zone_time_ratio_dictionary(
                s_behav_df_dic_hm4,
                time_bin=_trigger_zone_time_bin,
            )
        )
        ns_trigger_zone_time_ratio_by_time_bin_hm4[_trigger_zone_time_bin] = (
            build_trigger_zone_time_ratio_dictionary(
                ns_behav_df_dic_hm4,
                time_bin=_trigger_zone_time_bin,
            )
        )

    s_trigger_zone_time_ratio_dic_hm3 = (
        s_trigger_zone_time_ratio_by_time_bin_hm3[15]
    )
    ns_trigger_zone_time_ratio_dic_hm3 = (
        ns_trigger_zone_time_ratio_by_time_bin_hm3[15]
    )
    s_trigger_zone_time_ratio_dic_hm4 = (
        s_trigger_zone_time_ratio_by_time_bin_hm4[15]
    )
    ns_trigger_zone_time_ratio_dic_hm4 = (
        ns_trigger_zone_time_ratio_by_time_bin_hm4[15]
    )
    return (
        ns_trigger_zone_time_ratio_by_time_bin_hm3,
        ns_trigger_zone_time_ratio_by_time_bin_hm4,
        ns_trigger_zone_time_ratio_dic_hm3,
        ns_trigger_zone_time_ratio_dic_hm4,
        s_trigger_zone_time_ratio_by_time_bin_hm3,
        s_trigger_zone_time_ratio_by_time_bin_hm4,
        s_trigger_zone_time_ratio_dic_hm3,
        s_trigger_zone_time_ratio_dic_hm4,
        trigger_zone_time_bins,
    )


@app.cell
def _(
    build_subject_day_difference,
    difference_summary_method,
    mo,
    ns_trigger_zone_time_ratio_by_time_bin_hm3,
    ns_trigger_zone_time_ratio_by_time_bin_hm4,
    ns_trigger_zone_time_ratio_dic_hm3,
    ns_trigger_zone_time_ratio_dic_hm4,
    plot_ratio_boxplot,
    plot_subject_day_difference,
    ratio_dictionaries_to_long,
    s_trigger_zone_time_ratio_by_time_bin_hm3,
    s_trigger_zone_time_ratio_by_time_bin_hm4,
    s_trigger_zone_time_ratio_dic_hm3,
    s_trigger_zone_time_ratio_dic_hm4,
    save_figures_to_folder,
    trigger_zone_time_bins,
):
    df_trigger_zone_time_ratio_hm3 = ratio_dictionaries_to_long(
        ns_trigger_zone_time_ratio_dic_hm3,
        s_trigger_zone_time_ratio_dic_hm3,
    )
    df_trigger_zone_time_ratio_hm4 = ratio_dictionaries_to_long(
        ns_trigger_zone_time_ratio_dic_hm4,
        s_trigger_zone_time_ratio_dic_hm4,
    )

    trigger_zone_time_ratio_fig_hm3 = plot_ratio_boxplot(
        df_trigger_zone_time_ratio_hm3,
        "hM3Dq: first 15s",
        "Trigger-zone time ratio (x > trigger_zone_x_threshold)",
    )
    trigger_zone_time_ratio_fig_hm4 = plot_ratio_boxplot(
        df_trigger_zone_time_ratio_hm4,
        "hM4Di: first 15s",
        "Trigger-zone time ratio (x > trigger_zone_x_threshold)",
    )

    trigger_zone_time_ratio_difference_figs_hm3 = {}
    trigger_zone_time_ratio_difference_figs_hm4 = {}
    for _trigger_zone_time_bin in trigger_zone_time_bins:
        df_trigger_zone_time_bin_hm3 = ratio_dictionaries_to_long(
            ns_trigger_zone_time_ratio_by_time_bin_hm3[
                _trigger_zone_time_bin
            ],
            s_trigger_zone_time_ratio_by_time_bin_hm3[_trigger_zone_time_bin],
        )
        df_trigger_zone_time_bin_hm4 = ratio_dictionaries_to_long(
            ns_trigger_zone_time_ratio_by_time_bin_hm4[
                _trigger_zone_time_bin
            ],
            s_trigger_zone_time_ratio_by_time_bin_hm4[_trigger_zone_time_bin],
        )
        df_trigger_zone_difference_hm3 = build_subject_day_difference(
            df_trigger_zone_time_bin_hm3,
            summary_method=difference_summary_method,
        )
        df_trigger_zone_difference_hm4 = build_subject_day_difference(
            df_trigger_zone_time_bin_hm4,
            summary_method=difference_summary_method,
        )
        trigger_zone_time_ratio_difference_figs_hm3[
            _trigger_zone_time_bin
        ] = plot_subject_day_difference(
            df_trigger_zone_difference_hm3,
            f"hM3Dq: first {_trigger_zone_time_bin}s",
            "Trigger-zone time ratio: sound - no sound",
            summary_method=difference_summary_method,
            figsize=(7, 4),
        )
        trigger_zone_time_ratio_difference_figs_hm4[
            _trigger_zone_time_bin
        ] = plot_subject_day_difference(
            df_trigger_zone_difference_hm4,
            f"hM4Di: first {_trigger_zone_time_bin}s",
            "Trigger-zone time ratio: sound - no sound",
            summary_method=difference_summary_method,
            figsize=(7, 4),
        )

    trigger_zone_difference_figure_rows = [
        mo.hstack(
            [
                trigger_zone_time_ratio_difference_figs_hm3[time_bin],
                trigger_zone_time_ratio_difference_figs_hm4[time_bin],
            ]
        )
        for time_bin in trigger_zone_time_bins
    ]

    trigger_zone_figures_to_save = [
        ("hM3Dq: first 15s", trigger_zone_time_ratio_fig_hm3),
        ("hM4Di: first 15s", trigger_zone_time_ratio_fig_hm4),
    ]
    for _trigger_zone_time_bin in trigger_zone_time_bins:
        trigger_zone_figures_to_save.extend(
            [
                (
                    f"hM3Dq: first {_trigger_zone_time_bin}s",
                    trigger_zone_time_ratio_difference_figs_hm3[
                        _trigger_zone_time_bin
                    ],
                ),
                (
                    f"hM4Di: first {_trigger_zone_time_bin}s",
                    trigger_zone_time_ratio_difference_figs_hm4[
                        _trigger_zone_time_bin
                    ],
                ),
            ]
        )
    trigger_zone_saved_figure_paths = save_figures_to_folder(
        trigger_zone_figures_to_save,
        "trigger_zone_time_ratio_sound-nosound",
    )

    mo.vstack(
        [
            mo.hstack(
                [
                    trigger_zone_time_ratio_fig_hm3,
                    trigger_zone_time_ratio_fig_hm4,
                ]
            ),
            *trigger_zone_difference_figure_rows,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## crossing times
    """)
    return


@app.cell
def _(pd, session_df_dic):
    import ast as _ast

    _trigger_zone_transition_col = "_Transition_to_animal_inside_trigger_zone"

    def _extract_transition_times(value):
        if value is None:
            return []

        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            try:
                value = _ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return [value]

        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass

        if isinstance(value, (list, tuple, set)):
            times = []
            for item in value:
                times.extend(_extract_transition_times(item))
            return times

        return [value]

    trigger_zone_crossing_count_dic = {}
    for _session_name, _session_df in session_df_dic.items():

        _crossing_times = []
        for _value in _session_df[_trigger_zone_transition_col]:
            _crossing_times.extend(_extract_transition_times(_value))

        trigger_zone_crossing_count_dic[_session_name] = len(_crossing_times)
    return (trigger_zone_crossing_count_dic,)


@app.cell
def _(
    behav_pair_map,
    hM3Dq_mice,
    hM4Di_mice,
    np,
    pd,
    trigger_zone_crossing_count_dic,
):
    _crossing_rows = []

    _pair_map = behav_pair_map.copy()
    _pair_map["_DCZ_date"] = pd.to_datetime(_pair_map["DCZ_date"], errors="coerce")
    _pair_map = _pair_map.sort_values(["subject", "_DCZ_date", "pair_id"])

    for _subject, _subject_pair_map in _pair_map.groupby("subject"):
        for _pair_index, (_, _row) in enumerate(_subject_pair_map.iterrows(), start=1):
            for _condition, _key_col in [
                ("dcz", "DCZ_key"),
                ("saline", "saline_key"),
            ]:
                _session_key = _row[_key_col]

                if pd.isna(_session_key):
                    continue

                _crossing_rows.append(
                    {
                        "subject": _subject,
                        "pair_id": _row["pair_id"],
                        "pair_index": _pair_index,
                        "condition": _condition,
                        "session": _session_key,
                        "crossing_counts": trigger_zone_crossing_count_dic.get(
                            _session_key,
                            np.nan,
                        ),
                    }
                )

    crossing_count_pair_df = pd.DataFrame(_crossing_rows)
    crossing_count_pair_df["group"] = np.where(
        crossing_count_pair_df["subject"].isin(hM3Dq_mice),
        "hM3Dq",
        np.where(
            crossing_count_pair_df["subject"].isin(hM4Di_mice),
            "hM4Di",
            "unknown",
        ),
    )
    return (crossing_count_pair_df,)


@app.cell
def _(crossing_count_pair_df, figures_save_dir, os, plt, sns):
    _fig, _ax = plt.subplots(figsize=(6, 4), dpi=300)
    _g = sns.pointplot(
        data=crossing_count_pair_df,
        x="condition",
        y="crossing_counts",
        hue='group',
        markers=["o", "s"],
        order=["saline", "dcz"], 
        # linestyle=["-", "--"],
        palette={
            "hM3Dq": "orange",
            "hM4Di": "forestgreen",
        },
        linewidth=2,
        estimator='mean', 
        errwidth = 2,
        ci='sd',
        errorbar=('ci', 95),
        capsize=0.1,
        ax = _ax
    )
    _fig.savefig(os.path.join(figures_save_dir, "crossing_counts_between_different_conditions.svg"))
    _g
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # sound intensity in behavior
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## mice by mice, trial plot across time
    """)
    return


@app.cell
def _(
    analyzed_video_names,
    behav_df_filtered_dic,
    hM3Dq_mice,
    hM4Di_mice,
    mo,
    np,
    pd,
    plt,
    save_figures_to_folder,
    session_df_dic,
    sns,
):
    from scipy import stats as _stats
    from scipy.signal import hilbert as _hilbert
    import ast as _ast

    # auditory looming sounds
    def crescendo_looming_sound(
        amp_start: float = 20,
        amp_end: float = 90,
        ramp_duration: float = 0.4,
        ramp_down_duration: float = 0.005,
        hold_duration: float = 0.595,
        n_repeats: int = 10,
    ) -> np.ndarray:
        # convert n_repeats to an int
        n_repeats = int(n_repeats)
        fs = 192000  # Sampling frequency
        # Generate ramp + rampdown + hold for one repetition
        n_ramp = int(fs * ramp_duration)
        n_hold = int(fs * hold_duration)
        n_ramp_down = int(fs * ramp_down_duration)
        # convert amplitudes to dB
        db_start = 20 * np.log10(amp_start)
        db_end = 20 * np.log10(amp_end)
        # Linear amplitude ramp (in dB scale)
        db_ramp = np.linspace(db_start, db_end, n_ramp)
        db_ramp_down = np.linspace(db_end, db_start, n_ramp_down)
        # Convert back to amplitudes (exponential in linear scale)
        ramp_amplitudes = 10 ** (db_ramp / 20)
        ramp_down_amplitudes = 10 ** (db_ramp_down / 20)
        hold_amplitudes = np.ones(n_hold) * amp_start
        # Create one cycle of noise
        noise_ramp = np.random.randn(n_ramp) * ramp_amplitudes
        noise_hold = np.random.randn(n_hold) * hold_amplitudes
        noise_ramp_down = np.random.randn(n_ramp_down) * ramp_down_amplitudes
        cycle = np.concatenate([noise_ramp, noise_ramp_down, noise_hold])
        # Repeat 10 times
        stimulus = np.tile(cycle, n_repeats)

        return stimulus, np.arange(len(stimulus)) / fs

    def get_sound_event_times(value):
        if value is None:
            return []
        if isinstance(value, str):
            if not value.strip():
                return []
            value = _ast.literal_eval(value)
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            return list(value)
        return [value]

    def get_sound_timestamp(behav_df):
        if ("timestamp", "") in behav_df.columns:
            return behav_df[("timestamp", "")]
        timestamp = behav_df["timestamp"]
        if isinstance(timestamp, pd.DataFrame):
            return timestamp.iloc[:, 0]
        return timestamp

    def get_named_column(behav_df, column_name):
        if (column_name, "") in behav_df.columns:
            return behav_df[(column_name, "")]
        column = behav_df[column_name]
        if isinstance(column, pd.DataFrame):
            return column.iloc[:, 0]
        return column

    def build_session_title_lookup():
        session_title_lookup = {}
        for mouse in sorted({name[:6] for name in analyzed_video_names}):
            mouse_names = sorted(
                name for name in analyzed_video_names if name[:6] == mouse
            )
            dates = sorted({name.rsplit("_", 2)[-2] for name in mouse_names})
            date_to_day = {
                date: day for day, date in enumerate(dates, start=1)
            }
            if mouse in hM3Dq_mice:
                group_name = "hM3Dq"
            elif mouse in hM4Di_mice:
                group_name = "hM4Di"
            else:
                group_name = "unknown"
            for name in mouse_names:
                date = name.rsplit("_", 2)[-2]
                session_title_lookup[name] = (
                    f"{mouse} {group_name} day{date_to_day[date]}"
                )
        return session_title_lookup

    session_title_lookup = build_session_title_lookup()

    sound_play_dic = {}
    for _name in analyzed_video_names:
        sound_play_list = []
        if _name in session_df_dic and "sound_played" in session_df_dic[_name].columns:
            for sound_played in session_df_dic[_name]["sound_played"]:
                sound_play_list.extend(get_sound_event_times(sound_played))
        sound_play_dic[_name] = sound_play_list

    stimulus, t = crescendo_looming_sound()
    analytic_signal = _hilbert(stimulus)
    amplitude_envelope = np.abs(analytic_signal)

    window_size = 10000
    smooth_envelope = np.convolve(
        amplitude_envelope,
        np.ones(window_size) / window_size,
        mode="same",
    )

    def get_sound_intensity(behav_df, sound_play_list):
        behav_df_copy = behav_df.copy()
        timestamp = pd.to_numeric(
            get_sound_timestamp(behav_df_copy),
            errors="coerce",
        )

        sound_intensity = np.full(len(behav_df_copy), np.nan)
        trial_col = np.full(len(behav_df_copy), np.nan)

        i = 1
        for start in sound_play_list:
            end = start + t[-1]
            mask = (timestamp >= start) & (timestamp <= end)
            if not np.any(mask):
                continue

            rel_t = timestamp.loc[mask] - start
            env_interp = np.interp(rel_t, t, smooth_envelope)

            sound_intensity[mask.to_numpy()] = env_interp
            trial_col[mask.to_numpy()] = i
            i += 1

        behav_df_copy[("sound_intensity", "")] = sound_intensity
        behav_df_copy[("trial", "")] = trial_col

        return behav_df_copy

    behav_df_sound_intensity_dic = {}
    for _name in analyzed_video_names:
        behav_df_sound_intensity_dic[_name] = get_sound_intensity(
            behav_df_filtered_dic[_name],
            sound_play_dic[_name],
        )

    def plot_speed_change_with_sound(name):
        fig, ax = plt.subplots(1, 1, figsize=(15, 3.5), dpi=150)
        _behav_df_filtered = behav_df_sound_intensity_dic[name]
        timestamp = pd.to_numeric(
            get_sound_timestamp(_behav_df_filtered),
            errors="coerce",
        )
        mean_speed = pd.to_numeric(
            _behav_df_filtered[("Center", "mean_speed")],
            errors="coerce",
        )
        sound_play_list = sound_play_dic[name]

        sound_before_frames = 5  # 5 seconds before sound
        sound_after_frames = 10  # 10 seconds after sound
        colors = sns.color_palette("crest", len(sound_play_list))
        for sound, color in zip(sound_play_list[::-1], colors[::-1]):
            sound_mask = (
                (timestamp >= sound - sound_before_frames)
                & (timestamp <= sound + sound_after_frames)
            )
            sound_window_time = timestamp.loc[sound_mask] - sound
            sound_window_speed = mean_speed.loc[sound_mask]

            ax.plot(
                sound_window_time,
                sound_window_speed,
                color=color,
                alpha=0.3,
            )

        cbar = plt.colorbar(
            plt.cm.ScalarMappable(cmap=sns.color_palette("crest", as_cmap=True)),
            orientation="horizontal",
            ax=ax,
            shrink=0.3,
        )
        cbar.set_ticks([])
        cbar.set_label("before -> after")

        # plot the sound intensity on the speed plot, make it like shadow
        ax.plot(t, smooth_envelope, color="k", lw=1)
        ax.set_title(session_title_lookup[name])
        fig.tight_layout()
        return fig

    speed_change_with_sound_figures = [
        plot_speed_change_with_sound(_name)
        for _name in sorted(analyzed_video_names)
    ]
    speed_change_with_sound_saved_figure_paths = save_figures_to_folder(
        [
            (session_title_lookup[_name], fig)
            for _name, fig in zip(
                sorted(analyzed_video_names),
                speed_change_with_sound_figures,
            )
        ],
        "speed_change_with_sound",
    )

    corr_dic = {}
    p_dic = {}
    trial_ids_dic = {}
    for _name in analyzed_video_names:
        _behav_df_filtered = behav_df_sound_intensity_dic[_name]

        corr_list = []
        p_list = []
        trial_ids = []
        trial_values = get_named_column(_behav_df_filtered, "trial").dropna().unique()

        for i in trial_values:
            trial_df = _behav_df_filtered[
                get_named_column(_behav_df_filtered, "trial") == i
            ]

            # Ensure valid numeric arrays and remove NaNs
            x = pd.to_numeric(
                trial_df[("Center", "mean_speed")],
                errors="coerce",
            )
            y = pd.to_numeric(
                get_named_column(trial_df, "sound_intensity"),
                errors="coerce",
            )
            mask = (~np.isnan(x)) & (~np.isnan(y))
            x_valid = x[mask]
            y_valid = y[mask]

            if len(x_valid) > 2:
                corr, p = _stats.pearsonr(x_valid, y_valid)
                corr_list.append(corr)
                p_list.append(p)
                trial_ids.append(i)
            else:
                corr_list.append(np.nan)
                p_list.append(np.nan)
                trial_ids.append(i)
        corr_dic[_name] = corr_list
        p_dic[_name] = p_list
        trial_ids_dic[_name] = trial_ids

    def plot_correlation_between_speed_sound(name):
        fig, ax = plt.subplots(1, 1, figsize=(12, 3.5), dpi=150)
        corr_list = corr_dic[name]
        p_list = p_dic[name]
        bars = ax.bar(
            range(len(corr_list)),
            corr_list,
            color="lightgray",
            edgecolor="none",
        )

        # Add asterisks for p < 0.05
        for bar, p in zip(bars, p_list):
            if not np.isnan(p) and p < 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    (
                        bar.get_height() + 0.01
                        if bar.get_height() > 0
                        else bar.get_height() - 0.03
                    ),
                    "*",
                    ha="center",
                    va="bottom",
                    color="k",
                    fontsize=14,
                    fontweight="bold",
                )

        ax.axhline(0, color="k", lw=1)
        ax.set_xticks([])
        ax.set_xlabel("Trial")
        ax.set_ylabel("Pearson correlation (r)")
        ax.set_title(session_title_lookup[name])
        fig.tight_layout()
        return fig

    correlation_between_speed_sound_figures = [
        plot_correlation_between_speed_sound(_name)
        for _name in sorted(analyzed_video_names)
    ]
    correlation_between_speed_sound_saved_figure_paths = save_figures_to_folder(
        [
            (session_title_lookup[_name], fig)
            for _name, fig in zip(
                sorted(analyzed_video_names),
                correlation_between_speed_sound_figures,
            )
        ],
        "correlation_between_speed_sound",
    )

    mo.vstack(
        [
            mo.md("## speed change with sound"),
            *speed_change_with_sound_figures,
            mo.md("## correlation between speed and sound"),
            *correlation_between_speed_sound_figures,
        ]
    )
    return smooth_envelope, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## plot all the speed and x position trace with sound playing
    """)
    return


@app.cell
def _(
    home_zone_x_threshold,
    plt,
    smooth_envelope,
    t,
    trigger_zone_x_threshold,
):
    def plot_behavior_windows_with_sound(
        behav_df_dic,
        window_label="Sound-play",
        line_color="firebrick",
        line_alpha=0.25,
        line_width=1,
    ):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, dpi=150)

        for session_name, trial_dic in behav_df_dic.items():
            for _, window_df_list in trial_dic.items():
                for window_df in window_df_list:
                    axes[0].plot(
                        window_df[("timestamp", "")],
                        window_df[("Center", "mean_speed")],
                        color=line_color,
                        alpha=line_alpha,
                        lw=line_width,
                    )
                    axes[1].plot(
                        window_df[("timestamp", "")],
                        window_df[("Center", "x")],
                        color=line_color,
                        alpha=line_alpha,
                        lw=line_width,
                    )

        axes[0].plot(t, smooth_envelope, color="k", lw=2)
        axes[1].plot(t, smooth_envelope, color="k", lw=2)

        axes[0].axvline(0, color="k", linestyle="--", lw=1)
        axes[1].axvline(0, color="k", linestyle="--", lw=1)
        axes[1].axhline(
            trigger_zone_x_threshold,
            color="magenta",
            linestyle="--",
            lw=2,
            label="Trigger zone",
        )
        axes[1].axhline(
            home_zone_x_threshold,
            color="green",
            linestyle="--",
            lw=2,
            label="Trigger zone",
        )

        axes[0].set_ylabel("Mean speed")
        axes[1].set_ylabel("X position")
        axes[1].set_xlabel("Time from sound onset (s)")

        axes[0].set_title(f"{window_label} windows: Center mean_speed")
        axes[1].set_title(f"{window_label} windows: Center x position")

        axes[1].legend(frameon=False)

        plt.tight_layout()
        plt.show()

        return fig, axes

    return (plot_behavior_windows_with_sound,)


@app.cell
def _(
    plot_behavior_windows_with_sound,
    s_behav_df_dic_hm3_prebin,
    s_behav_df_dic_hm4_prebin,
):
    s_behav_df_dic_prebin = {
        **s_behav_df_dic_hm3_prebin,
        **s_behav_df_dic_hm4_prebin,
    }

    _fig, _axes = plot_behavior_windows_with_sound(
        s_behav_df_dic_prebin,
        window_label="Sound-play",
        line_color="firebrick",
    )
    return (s_behav_df_dic_prebin,)


@app.cell
def _(
    ns_behav_df_dic_hm3_prebin,
    ns_behav_df_dic_hm4_prebin,
    plot_behavior_windows_with_sound,
):
    ns_behav_df_dic_prebin = {
        **ns_behav_df_dic_hm3_prebin,
        **ns_behav_df_dic_hm4_prebin,
    }

    _fig, _axes = plot_behavior_windows_with_sound(
        ns_behav_df_dic_prebin,
        window_label="No-sound",
        line_color="steelblue",
    )
    return (ns_behav_df_dic_prebin,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # classify trials
    """)
    return


@app.cell
def _(home_zone_x_threshold, np, pd, trigger_zone_x_threshold):
    def classify_escape_trials(
        behav_df_dic,
        min_leave_frames=60,
        min_stayhome_frames=60,
        pk_spd_time_afterleave=0.5,
        aftersound_stayhome_timewindow=(10.5, 11),
        soundplay_finished_time=10,
        soundplay_delay_time=12,
        path_efficiency_sd_criteria=1,
    ):
        def first_stable_true_time(mask, timestamp, min_frames=60):
            mask = pd.Series(mask, index=timestamp.index).fillna(False).astype(bool)
            stable = mask.rolling(min_frames, min_periods=min_frames).sum() >= min_frames
            stable_positions = np.flatnonzero(stable.to_numpy())

            if len(stable_positions) == 0:
                return np.nan, None

            first_stable_position = stable_positions[0]
            first_true_position = first_stable_position - min_frames + 1
            first_true_index = timestamp.index[first_true_position]

            return float(timestamp.loc[first_true_index]), first_true_position

        def empty_escape_metrics():
            return {
                "real_leave": np.nan,
                "real_stayhome": np.nan,
                "straight_line_distance": np.nan,
                "actual_path_distance": np.nan,
                "path_efficiency": np.nan,
                "peak_speed": np.nan,
                "peak_speed_time": np.nan,
                "peak_speed_xposition": np.nan,
                "aftersound_stayhome": False,
            }

        def compute_path_efficiency(
            trial_df_delay,
            timestamp_delay,
            real_leave_time,
            real_stayhome_time,
        ):
            if (
                pd.isna(real_leave_time)
                or pd.isna(real_stayhome_time)
                or real_stayhome_time <= real_leave_time
            ):
                return np.nan, np.nan, np.nan

            path_mask = (
                (timestamp_delay >= real_leave_time)
                & (timestamp_delay <= real_stayhome_time)
            )
            path_df = trial_df_delay.loc[path_mask].copy()
            if path_df.shape[0] < 2:
                return np.nan, np.nan, np.nan

            xy = path_df.loc[:, [("Center", "x"), ("Center", "y")]].apply(
                pd.to_numeric,
                errors="coerce",
            )
            xy = xy.dropna()
            if xy.shape[0] < 2:
                return np.nan, np.nan, np.nan

            start_xy = xy.iloc[0].to_numpy(dtype=float)
            end_xy = xy.iloc[-1].to_numpy(dtype=float)
            straight_line_distance = float(np.linalg.norm(end_xy - start_xy))
            if (
                not np.isfinite(straight_line_distance)
                or straight_line_distance <= 0
            ):
                return straight_line_distance, np.nan, np.nan

            if ("Center", "distance") in path_df.columns:
                distance = pd.to_numeric(
                    path_df.loc[xy.index, ("Center", "distance")],
                    errors="coerce",
                ).iloc[1:]
                actual_path_distance = float(distance.dropna().sum())
            else:
                xy_diff = np.diff(xy.to_numpy(dtype=float), axis=0)
                actual_path_distance = float(
                    np.sqrt((xy_diff ** 2).sum(axis=1)).sum()
                )

            if (
                not np.isfinite(actual_path_distance)
                or actual_path_distance <= 0
            ):
                return straight_line_distance, actual_path_distance, np.nan

            path_efficiency = actual_path_distance / straight_line_distance
            return straight_line_distance, actual_path_distance, path_efficiency

        def classify_escape_window(window_df, soundplay_finished_time=soundplay_finished_time, soundplay_delay_time=soundplay_delay_time):
            metrics = empty_escape_metrics()

            timestamp = pd.to_numeric(window_df[("timestamp", "")], errors="coerce")

            time_mask_soundplay = (
                (timestamp >= 0)
                & (timestamp <= soundplay_finished_time)
            )
            time_mask_delay = (
                (timestamp >= 0)
                & (timestamp <= soundplay_delay_time)
            )

            trial_df_soundplay = window_df.loc[time_mask_soundplay].copy()
            timestamp_soundplay = timestamp.loc[time_mask_soundplay]

            trial_df_delay = window_df.loc[time_mask_delay].copy()
            timestamp_delay = timestamp.loc[time_mask_delay]

            if trial_df_soundplay.empty or trial_df_delay.empty:
                return metrics

            x = pd.to_numeric(
                trial_df_soundplay[("Center", "x")],
                errors="coerce",
            )

            outside = x < trigger_zone_x_threshold
            real_leave_time, _ = first_stable_true_time(
                outside,
                timestamp_soundplay,
                min_frames=min_leave_frames,
            )

            if pd.isna(real_leave_time):
                return metrics

            metrics["real_leave"] = real_leave_time

            after_leave_mask = timestamp_delay >= real_leave_time
            after_leave_df = trial_df_delay.loc[after_leave_mask].copy()
            after_leave_timestamp = timestamp_delay.loc[after_leave_mask]

            if after_leave_df.empty:
                return metrics

            after_leave_x = pd.to_numeric(
                after_leave_df[("Center", "x")],
                errors="coerce",
            )

            stay = after_leave_x < home_zone_x_threshold
            real_stayhome_time, _ = first_stable_true_time(
                stay,
                after_leave_timestamp,
                min_frames=min_stayhome_frames,
            )
            metrics["real_stayhome"] = real_stayhome_time
            (
                metrics["straight_line_distance"],
                metrics["actual_path_distance"],
                metrics["path_efficiency"],
            ) = compute_path_efficiency(
                trial_df_delay,
                timestamp_delay,
                real_leave_time,
                real_stayhome_time,
            )

            if aftersound_stayhome_timewindow is None:
                metrics["aftersound_stayhome"] = False
            else:
                aftersound_start, aftersound_end = aftersound_stayhome_timewindow
                aftersound_mask = (
                    (after_leave_timestamp >= aftersound_start)
                    & (after_leave_timestamp <= aftersound_end)
                )

                aftersound_x = pd.to_numeric(
                    after_leave_df.loc[aftersound_mask, ("Center", "x")],
                    errors="coerce",
                )

                metrics["aftersound_stayhome"] = (
                    not aftersound_x.empty
                    and aftersound_x.notna().all()
                    and (aftersound_x < home_zone_x_threshold).all()
                )

            speed = pd.to_numeric(
                after_leave_df[("Center", "mean_speed")],
                errors="coerce",
            )

            if speed.notna().any():
                peak_speed_index = speed.idxmax()
                metrics["peak_speed"] = float(speed.loc[peak_speed_index])
                metrics["peak_speed_time"] = float(
                    after_leave_timestamp.loc[peak_speed_index]
                )
                peak_speed_xposition = pd.to_numeric(
                    after_leave_df.loc[peak_speed_index, ("Center", "x")],
                    errors="coerce",
                )
                metrics["peak_speed_xposition"] = (
                    float(peak_speed_xposition)
                    if pd.notna(peak_speed_xposition)
                    else np.nan
                )

            return metrics

        classified_columns = [
            "session",
            "trial",
            "window_index",
            "trigger_time",
            "real_leave",
            "real_stayhome",
            "straight_line_distance",
            "actual_path_distance",
            "path_efficiency",
            "path_efficiency_log",
            "path_efficiency_pass_filter",
            "peak_speed",
            "peak_speed_time",
            "peak_speed_xposition",
            "peak_latency",
            "aftersound_stayhome",
            "escape",
        ]

        classified_df_dic = {}

        for name, trial_dic in behav_df_dic.items():
            rows = []

            for trial, window_df_list in trial_dic.items():
                for window_index, window_df in enumerate(window_df_list):
                    metrics = classify_escape_window(window_df)

                    real_leave = metrics["real_leave"]
                    real_stayhome = metrics["real_stayhome"]
                    peak_speed_time = metrics["peak_speed_time"]
                    aftersound_stayhome = metrics["aftersound_stayhome"]

                    if pd.notna(real_leave) and pd.notna(peak_speed_time):
                        peak_latency = peak_speed_time - real_leave
                    else:
                        peak_latency = np.nan

                    escape = (
                        pd.notna(real_leave)
                        and pd.notna(real_stayhome)
                        and pd.notna(peak_speed_time)
                        and (
                            aftersound_stayhome_timewindow is None
                            or aftersound_stayhome
                        )
                        and peak_latency >= 0
                        and peak_latency < pk_spd_time_afterleave
                    )

                    rows.append(
                        {
                            "session": name,
                            "trial": trial,
                            "window_index": window_index,
                            "trigger_time": window_df.attrs.get(
                                "trigger_time",
                                np.nan,
                            ),
                            "real_leave": real_leave,
                            "real_stayhome": real_stayhome,
                            "straight_line_distance": metrics["straight_line_distance"],
                            "actual_path_distance": metrics["actual_path_distance"],
                            "path_efficiency": metrics["path_efficiency"],
                            "path_efficiency_log": np.nan,
                            "path_efficiency_pass_filter": False,
                            "peak_speed": metrics["peak_speed"],
                            "peak_speed_time": peak_speed_time,
                            "peak_latency": peak_latency,
                            "peak_speed_xposition": metrics["peak_speed_xposition"],
                            "aftersound_stayhome": aftersound_stayhome,
                            "escape": bool(escape),
                        }
                    )

            classified_df = pd.DataFrame(
                rows,
                columns=classified_columns,
            )
            path_efficiency = pd.to_numeric(
                classified_df["path_efficiency"],
                errors="coerce",
            )
            path_efficiency = path_efficiency.replace([np.inf, -np.inf], np.nan)
            valid_path_efficiency = path_efficiency > 0

            classified_df.loc[
                valid_path_efficiency,
                "path_efficiency_log",
            ] = np.log(path_efficiency.loc[valid_path_efficiency])

            path_efficiency_log = classified_df["path_efficiency_log"]
            median_path_efficiency_log = path_efficiency_log.median()
            std_path_efficiency_log = path_efficiency_log.std()
            if pd.isna(std_path_efficiency_log):
                std_path_efficiency_log = 0

            path_efficiency_threshold = (
                median_path_efficiency_log
                + path_efficiency_sd_criteria * std_path_efficiency_log
            )
            classified_df["path_efficiency_pass_filter"] = (
                path_efficiency_log < path_efficiency_threshold
            )
            # add path efficiency as one of filters
            # classified_df["escape"] = (
            #     classified_df["escape"] & classified_df["path_efficiency_pass_filter"]
            # )
            classified_df_dic[name] = classified_df

        return classified_df_dic

    return (classify_escape_trials,)


@app.function
def split_escape_behavior_dict(
    behav_df_dic,
    session_df_dic_classified,
):
    s_behav_df_dic_prebin_escape = {}
    s_behav_df_dic_prebin_noreaction = {}

    for name, trial_dic in behav_df_dic.items():
        if name not in session_df_dic_classified:
            continue

        session_df = session_df_dic_classified[name]

        if "escape" not in session_df.columns:
            continue

        escape_rows = session_df.loc[session_df["escape"] == True].copy()
        escape_windows = set()
        if {"trial", "window_index"}.issubset(escape_rows.columns):
            escape_windows = {
                (str(row["trial"]), int(row["window_index"]))
                for _, row in escape_rows.iterrows()
            }
        else:
            escape_trials = set(escape_rows["trial"])
            escape_trials_str = {str(trial) for trial in escape_trials}

        s_behav_df_dic_prebin_escape[name] = {}
        s_behav_df_dic_prebin_noreaction[name] = {}

        for trial, window_df_list in trial_dic.items():
            for window_index, window_df in enumerate(window_df_list):
                if {"trial", "window_index"}.issubset(escape_rows.columns):
                    is_escape = (str(trial), window_index) in escape_windows
                else:
                    is_escape = (
                        trial in escape_trials
                        or str(trial) in escape_trials_str
                    )

                if is_escape:
                    s_behav_df_dic_prebin_escape[name].setdefault(
                        trial,
                        [],
                    ).append(window_df)
                else:
                    s_behav_df_dic_prebin_noreaction[name].setdefault(
                        trial,
                        [],
                    ).append(window_df)

    return s_behav_df_dic_prebin_escape, s_behav_df_dic_prebin_noreaction


@app.cell
def _(
    frame_xy,
    home_zone_x_threshold,
    mpl,
    pd,
    plot_test,
    plt,
    trigger_zone_x_threshold,
    x_len_tran,
    y_len_tran,
):
    def _get_timestamp_from_window_df(window_df):
        if ("timestamp", "") in window_df.columns:
            return pd.to_numeric(window_df[("timestamp", "")], errors="coerce")
        return pd.to_numeric(window_df["timestamp"], errors="coerce")


    def plot_window_dic_trajectories(
        behav_df_dic,
        title="Sound-play trajectories",
        cmap="inferno",
        start_color="w",
        time_after=10,
    ):
        # ========= 1) Unify the colorbar scale =========
        all_speed = []

        for _session_name, _trial_dic in behav_df_dic.items():
            for _trial, _window_df_list in _trial_dic.items():
                for _window_df in _window_df_list:
                    timestamp = _get_timestamp_from_window_df(_window_df)

                    time_mask = (timestamp >= 0) & (timestamp <= time_after)

                    speed = pd.to_numeric(
                        _window_df.loc[time_mask, ("Center", "mean_speed")],
                        errors="coerce",
                    ).dropna()

                    if not speed.empty:
                        all_speed.append(speed)

        all_speed = pd.concat(all_speed)
        vmin = all_speed.quantile(0.01)
        vmax = all_speed.quantile(0.99)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        # ========= 2) Plot trajectories =========
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))

        for _session_name, _trial_dic in behav_df_dic.items():
            for _trial, _window_df_list in _trial_dic.items():
                for _window_df in _window_df_list:
                    timestamp = _get_timestamp_from_window_df(_window_df)

                    time_mask = (timestamp >= 0) & (timestamp <= time_after)

                    df_traj = _window_df.loc[
                        time_mask,
                        "Center",
                    ][["x", "y", "mean_speed"]].copy()

                    df_traj[["x", "y", "mean_speed"]] = (
                        df_traj[["x", "y", "mean_speed"]]
                        .apply(pd.to_numeric, errors="coerce")
                        .interpolate(limit_direction="both")
                    )

                    df_traj = df_traj.dropna(subset=["x", "y", "mean_speed"])

                    if df_traj.empty:
                        continue

                    plot_test.plot_traj_speed(
                        df_traj,
                        cmap=cmap,
                        ax=ax,
                        norm=norm,
                    )

                    # mark trigger/start point, timestamp == 0
                    ax.scatter(
                        df_traj["x"].iloc[0],
                        df_traj["y"].iloc[0],
                        color=start_color,
                        s=80,
                        marker="o",
                        edgecolors="k",
                        zorder=3,
                    )

        # ========= 3) Axes =========
        ax.set_xlim(0, frame_xy[0]/x_len_tran)
        ax.set_ylim(0, frame_xy[1]/y_len_tran)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.axvline(x=trigger_zone_x_threshold, c="magenta", linestyle="--", linewidth=2)
        ax.axvline(x=home_zone_x_threshold, c="green", linestyle="--", linewidth=2)
        ax.set_title(title)

        # ========= 4) Colorbar =========
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(sm, orientation="vertical", ax=ax, shrink=0.4, pad=0.02)
        cbar.set_label("speed pixels/s", rotation=90)

        plt.tight_layout()
        plt.show()

        return fig

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## test the escape trials classify params
    """)
    return


@app.cell
def _(
    aftersound_stayhome_timewindow,
    classify_escape_trials,
    min_leave_frames,
    min_stayhome_frames,
    ns_behav_df_dic_prebin,
    pk_spd_time_afterleave,
    s_behav_df_dic_prebin,
    soundplay_delay_time,
    soundplay_finished_time,
):

    s_session_df_dic_classified = classify_escape_trials(
        s_behav_df_dic_prebin,
        min_leave_frames=min_leave_frames, 
        min_stayhome_frames=min_stayhome_frames, 
        pk_spd_time_afterleave=pk_spd_time_afterleave, 
        aftersound_stayhome_timewindow=aftersound_stayhome_timewindow,
        soundplay_finished_time=soundplay_finished_time,
        soundplay_delay_time=soundplay_delay_time,
    )
    ns_session_df_dic_classified = classify_escape_trials(
        ns_behav_df_dic_prebin,
        min_leave_frames=min_leave_frames, 
        min_stayhome_frames=min_stayhome_frames, 
        pk_spd_time_afterleave=pk_spd_time_afterleave,
        aftersound_stayhome_timewindow=aftersound_stayhome_timewindow,
        soundplay_finished_time=soundplay_finished_time,
        soundplay_delay_time=soundplay_delay_time,
    )
    return ns_session_df_dic_classified, s_session_df_dic_classified


@app.cell
def _(
    ns_behav_df_dic_prebin,
    ns_session_df_dic_classified,
    s_behav_df_dic_prebin,
    s_session_df_dic_classified,
):
    s_behav_df_dic_prebin_escape, s_behav_df_dic_prebin_noreaction = split_escape_behavior_dict(
            s_behav_df_dic_prebin,
            s_session_df_dic_classified,
        )
    ns_behav_df_dic_prebin_escape, ns_behav_df_dic_prebin_noreaction = split_escape_behavior_dict(
            ns_behav_df_dic_prebin,
            ns_session_df_dic_classified,
        )
    return (
        ns_behav_df_dic_prebin_escape,
        ns_behav_df_dic_prebin_noreaction,
        s_behav_df_dic_prebin_escape,
        s_behav_df_dic_prebin_noreaction,
    )


@app.cell
def _(
    behavior_utils,
    hM3Dq_mice,
    hM4Di_mice,
    ns_behav_df_dic_prebin_escape,
    ns_behav_df_dic_prebin_noreaction,
    ns_session_df_dic_classified,
    paired_dlc_dates,
    s_behav_df_dic_prebin_escape,
    s_behav_df_dic_prebin_noreaction,
    s_session_df_dic_classified,
):
    def split_by_group_and_condition(data_dic):
        data_dic_saline, data_dic_dcz, _pair_map, _missing_pairs = (
            behavior_utils.split_paired_behavior_dicts(
                data_dic,
                paired_dlc_dates,
            )
        )

        def filter_mice(input_dic, mice):
            mice = set(mice)
            return {
                name: value
                for name, value in input_dic.items()
                if name[:6] in mice
            }

        return {
            "hm3_saline": filter_mice(data_dic_saline, hM3Dq_mice),
            "hm3_dcz": filter_mice(data_dic_dcz, hM3Dq_mice),
            "hm4_saline": filter_mice(data_dic_saline, hM4Di_mice),
            "hm4_dcz": filter_mice(data_dic_dcz, hM4Di_mice),
        }

    s_session_df_dic_classified_split = split_by_group_and_condition(
        s_session_df_dic_classified
    )
    ns_session_df_dic_classified_split = split_by_group_and_condition(
        ns_session_df_dic_classified
    )
    s_behav_df_dic_prebin_escape_split = split_by_group_and_condition(
        s_behav_df_dic_prebin_escape
    )
    s_behav_df_dic_prebin_noreaction_split = split_by_group_and_condition(
        s_behav_df_dic_prebin_noreaction
    )
    ns_behav_df_dic_prebin_escape_split = split_by_group_and_condition(
        ns_behav_df_dic_prebin_escape
    )
    ns_behav_df_dic_prebin_noreaction_split = split_by_group_and_condition(
        ns_behav_df_dic_prebin_noreaction
    )
    return (
        ns_behav_df_dic_prebin_escape_split,
        ns_behav_df_dic_prebin_noreaction_split,
        ns_session_df_dic_classified_split,
        s_behav_df_dic_prebin_escape_split,
        s_behav_df_dic_prebin_noreaction_split,
        s_session_df_dic_classified_split,
    )


@app.cell
def _(home_zone_x_threshold, pd, smooth_envelope, t, trigger_zone_x_threshold):
    def plot_behavior_windows_with_sound_on_axes(
        behav_df_dic,
        axes,
        window_label,
        line_color,
        line_alpha=0.25,
        line_width=1,
        sound_linestyle="-",
    ):
        for _session_name, _trial_dic in behav_df_dic.items():
            for _, _window_df_list in _trial_dic.items():
                for _window_df in _window_df_list:
                    if ("timestamp", "") in _window_df.columns:
                        _timestamp = _window_df[("timestamp", "")]
                    else:
                        _timestamp = _window_df["timestamp"]

                    axes[0].plot(
                        _timestamp,
                        _window_df[("Center", "mean_speed")],
                        color=line_color,
                        alpha=line_alpha,
                        lw=line_width,
                    )
                    axes[1].plot(
                        _timestamp,
                        _window_df[("Center", "x")],
                        color=line_color,
                        alpha=line_alpha,
                        lw=line_width,
                    )

        axes[0].plot(t, smooth_envelope, color="k", linestyle=sound_linestyle, lw=2)
        axes[1].plot(t, smooth_envelope, color="k", linestyle=sound_linestyle, lw=2)

        axes[0].axvline(0, color="k", linestyle="--", lw=1)
        axes[1].axvline(0, color="k", linestyle="--", lw=1)
        axes[1].axhline(
            trigger_zone_x_threshold,
            color="magenta",
            linestyle="--",
            lw=2,
            label="Trigger zone",
        )
        axes[1].axhline(
            home_zone_x_threshold,
            color="green",
            linestyle="--",
            lw=2,
            label="Home zone",
        )

        axes[0].set_ylabel("Mean speed")
        axes[1].set_ylabel("X position")
        axes[1].set_xlabel("Time from trigger (s)")
        axes[0].set_title(f"{window_label}: Center mean_speed")
        axes[1].set_title(f"{window_label}: Center x position")
        axes[1].legend(frameon=False, fontsize=8)

    def build_escape_param_plot_dfs(session_df_dic_classified):
        _escape_rows = []
        _noreaction_rows = []
        _all_withoutescape_rows = []

        _columns = [
            "trial",
            "escape",
            "real_stayhome",
            "peak_speed",
            "peak_speed_xposition",
            "session",
        ]

        for _name, _session_df in session_df_dic_classified.items():
            # collect all non-escape trials, even if other columns are missing
            _all_withoutescape_plot_df = _session_df.copy()
            _all_withoutescape_plot_df["session"] = _name

            if "escape" in _all_withoutescape_plot_df.columns:
                _all_withoutescape_plot_df = _all_withoutescape_plot_df[
                    _all_withoutescape_plot_df["escape"] != True
                ]

            _all_withoutescape_rows.append(_all_withoutescape_plot_df)

            if not {
                "escape",
                "real_stayhome",
                "peak_speed",
                "peak_speed_xposition",
            }.issubset(_session_df.columns):
                continue

            _plot_df = _session_df[
                [
                    "trial",
                    "escape",
                    "real_stayhome",
                    "peak_speed",
                    "peak_speed_xposition",
                ]
            ].copy()

            _plot_df["session"] = _name
            _plot_df["real_stayhome"] = pd.to_numeric(
                _plot_df["real_stayhome"],
                errors="coerce",
            )
            _plot_df["peak_speed"] = pd.to_numeric(
                _plot_df["peak_speed"],
                errors="coerce",
            )
            _plot_df["peak_speed_xposition"] = pd.to_numeric(
                _plot_df["peak_speed_xposition"],
                errors="coerce",
            )
            _plot_df = _plot_df.dropna(
                subset=["real_stayhome", "peak_speed", "peak_speed_xposition"]
            )

            _escape_rows.append(_plot_df[_plot_df["escape"] == True])
            _noreaction_rows.append(_plot_df[_plot_df["escape"] == False])

        if _escape_rows:
            _escape_plot_df = pd.concat(_escape_rows, ignore_index=True)
        else:
            _escape_plot_df = pd.DataFrame(columns=_columns)

        if _noreaction_rows:
            _noreaction_plot_df = pd.concat(_noreaction_rows, ignore_index=True)
        else:
            _noreaction_plot_df = pd.DataFrame(columns=_columns)

        if _all_withoutescape_rows:
            _all_withoutescape_plot_df = pd.concat(
                _all_withoutescape_rows,
                ignore_index=True,
            )
        else:
            _all_withoutescape_plot_df = pd.DataFrame(columns=_columns)

        return (
            _escape_plot_df,
            _noreaction_plot_df,
            _all_withoutescape_plot_df,
        )

    def plot_escape_param_scatter(ax, plot_df, title, color, marker="o"):
        if plot_df.empty:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            ax.scatter(
                plot_df["peak_speed_xposition"],
                plot_df["peak_speed"],
                s=30,
                alpha=0.7,
                color=color,
                marker=marker,
            )

        ax.set_title(title)
        ax.set_xlabel("peak_speed_xposition")
        ax.set_ylabel("peak_speed")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def plot_escape_frequency_histogram(
        ax,
        escape_plot_df,
        all_withoutescape,
        title,
    ):
        _escape_count = len(escape_plot_df)
        _all_withoutescape_count = len(all_withoutescape)
        _total_count = _escape_count + _all_withoutescape_count

        if _total_count == 0:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)
            return

        _frequencies = [
            _escape_count / _total_count,
            _all_withoutescape_count / _total_count,
        ]
        _counts = [_escape_count, _all_withoutescape_count]

        _bars = ax.bar(
            [0, 1],
            _frequencies,
            color=["firebrick", "steelblue"],
            alpha=0.75,
            edgecolor="none",
        )

        for _bar, _freq, _count in zip(_bars, _frequencies, _counts):
            ax.text(
                _bar.get_x() + _bar.get_width() / 2,
                _freq + 0.03,
                f"{_freq:.2f}\nn={_count}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["escape", "non-escape"], rotation=30, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)




    return (
        build_escape_param_plot_dfs,
        plot_behavior_windows_with_sound_on_axes,
        plot_escape_frequency_histogram,
        plot_escape_param_scatter,
    )


@app.cell
def _(
    build_escape_param_plot_dfs,
    plot_behavior_windows_with_sound_on_axes,
    plot_escape_frequency_histogram,
    plot_escape_param_scatter,
    plt,
):
    def plot_escape_classification_overview(
        s_session_df_dic_classified,
        ns_session_df_dic_classified,
        s_behav_df_dic_prebin_escape,
        s_behav_df_dic_prebin_noreaction,
        ns_behav_df_dic_prebin_escape,
        ns_behav_df_dic_prebin_noreaction,
    ):
        escape_classification_overview_fig = plt.figure(
            figsize=(28, 18),
            dpi=150,
            constrained_layout=True,
        )
        _gs = escape_classification_overview_fig.add_gridspec(
            5,
            6,
            height_ratios=[1, 1, 1, 1, 1.1],
        )

        _s_escape_axes = [
            escape_classification_overview_fig.add_subplot(_gs[0, 0:3]),
            escape_classification_overview_fig.add_subplot(_gs[1, 0:3]),
        ]
        _s_noreaction_axes = [
            escape_classification_overview_fig.add_subplot(_gs[2, 0:3]),
            escape_classification_overview_fig.add_subplot(_gs[3, 0:3]),
        ]
        _ns_escape_axes = [
            escape_classification_overview_fig.add_subplot(_gs[0, 3:6]),
            escape_classification_overview_fig.add_subplot(_gs[1, 3:6]),
        ]
        _ns_noreaction_axes = [
            escape_classification_overview_fig.add_subplot(_gs[2, 3:6]),
            escape_classification_overview_fig.add_subplot(_gs[3, 3:6]),
        ]

        plot_behavior_windows_with_sound_on_axes(
            s_behav_df_dic_prebin_escape,
            _s_escape_axes,
            "Sound-play escape",
            "firebrick",
        )
        plot_behavior_windows_with_sound_on_axes(
            s_behav_df_dic_prebin_noreaction,
            _s_noreaction_axes,
            "Sound-play no reaction",
            "steelblue",
        )
        plot_behavior_windows_with_sound_on_axes(
            ns_behav_df_dic_prebin_escape,
            _ns_escape_axes,
            "No-sound escape",
            "firebrick",
            sound_linestyle="--",
        )
        plot_behavior_windows_with_sound_on_axes(
            ns_behav_df_dic_prebin_noreaction,
            _ns_noreaction_axes,
            "No-sound no reaction",
            "steelblue",
            sound_linestyle="--",
        )

        _s_escape_plot_df, _s_noreaction_plot_df, _s_all_withoutescape_plot_df = (
            build_escape_param_plot_dfs(s_session_df_dic_classified)
        )
        _ns_escape_plot_df, _ns_noreaction_plot_df, _ns_all_withoutescape_plot_df = (
            build_escape_param_plot_dfs(ns_session_df_dic_classified)
        )

        _scatter_axes = [
            escape_classification_overview_fig.add_subplot(_gs[4, 0]),
            escape_classification_overview_fig.add_subplot(_gs[4, 1]),
            escape_classification_overview_fig.add_subplot(_gs[4, 2]),
            escape_classification_overview_fig.add_subplot(_gs[4, 3]),
            escape_classification_overview_fig.add_subplot(_gs[4, 4]),
            escape_classification_overview_fig.add_subplot(_gs[4, 5]),
        ]
        plot_escape_param_scatter(
            _scatter_axes[0],
            _s_escape_plot_df,
            "Sound-play escape windows",
            "firebrick",
        )
        plot_escape_param_scatter(
            _scatter_axes[1],
            _s_all_withoutescape_plot_df,
            "Sound-play no-reaction windows",
            "steelblue",
        )
        plot_escape_frequency_histogram(
            _scatter_axes[2],
            _s_escape_plot_df,
            _s_all_withoutescape_plot_df,
            "Sound-play window frequency",
        )
        plot_escape_param_scatter(
            _scatter_axes[3],
            _ns_escape_plot_df,
            "No-sound escape windows",
            "firebrick",
            marker="x",
        )
        plot_escape_param_scatter(
            _scatter_axes[4],
            _ns_all_withoutescape_plot_df,
            "No-sound no-reaction windows",
            "steelblue",
            marker="x",
        )
        plot_escape_frequency_histogram(
            _scatter_axes[5],
            _ns_escape_plot_df,
            _ns_all_withoutescape_plot_df,
            "No-sound window frequency",
        )

        escape_classification_overview_fig.suptitle(
            "Escape trial classification overview",
            fontsize=16,
        )

        plt.show()

        _axes_dict = {
            "s_escape_speed": _s_escape_axes[0],
            "s_escape_x": _s_escape_axes[1],
            "s_noreaction_speed": _s_noreaction_axes[0],
            "s_noreaction_x": _s_noreaction_axes[1],
            "ns_escape_speed": _ns_escape_axes[0],
            "ns_escape_x": _ns_escape_axes[1],
            "ns_noreaction_speed": _ns_noreaction_axes[0],
            "ns_noreaction_x": _ns_noreaction_axes[1],
            "s_escape_scatter": _scatter_axes[0],
            "s_noreaction_scatter": _scatter_axes[1],
            "s_frequency": _scatter_axes[2],
            "ns_escape_scatter": _scatter_axes[3],
            "ns_noreaction_scatter": _scatter_axes[4],
            "ns_frequency": _scatter_axes[5],
        }
        return escape_classification_overview_fig, _axes_dict


    return (plot_escape_classification_overview,)


@app.cell
def _(
    build_escape_param_plot_dfs,
    plot_escape_frequency_histogram,
    plot_escape_param_scatter,
    plt,
):
    # scatter and frequency plot
    def plot_escape_classification_summary_plots(
        s_session_df_dic_classified,
        ns_session_df_dic_classified,
    ):
        escape_classification_summary_fig, _scatter_axes = plt.subplots(
            1,
            6,
            figsize=(28, 4),
            dpi=150,
            constrained_layout=True,
        )

        _s_escape_plot_df, _s_noreaction_plot_df, _s_all_withoutescape_plot_df = (
            build_escape_param_plot_dfs(s_session_df_dic_classified)
        )
        _ns_escape_plot_df, _ns_noreaction_plot_df, _ns_all_withoutescape_plot_df = (
            build_escape_param_plot_dfs(ns_session_df_dic_classified)
        )

        plot_escape_param_scatter(
            _scatter_axes[0],
            _s_escape_plot_df,
            "Sound-play escape windows",
            "firebrick",
        )
        plot_escape_param_scatter(
            _scatter_axes[1],
            _s_all_withoutescape_plot_df,
            "Sound-play no-reaction windows",
            "steelblue",
        )
        plot_escape_frequency_histogram(
            _scatter_axes[2],
            _s_escape_plot_df,
            _s_all_withoutescape_plot_df,
            "Sound-play window frequency",
        )
        plot_escape_param_scatter(
            _scatter_axes[3],
            _ns_escape_plot_df,
            "No-sound escape windows",
            "firebrick",
            marker="x",
        )
        plot_escape_param_scatter(
            _scatter_axes[4],
            _ns_all_withoutescape_plot_df,
            "No-sound no-reaction windows",
            "steelblue",
            marker="x",
        )
        plot_escape_frequency_histogram(
            _scatter_axes[5],
            _ns_escape_plot_df,
            _ns_all_withoutescape_plot_df,
            "No-sound window frequency",
        )

        escape_classification_summary_fig.suptitle(
            "Escape classification summary",
            fontsize=16,
        )

        plt.show()
        return escape_classification_summary_fig

    return (plot_escape_classification_summary_plots,)


@app.cell
def _():
    min_leave_frames=60
    min_stayhome_frames=15
    pk_spd_time_afterleave=2
    # aftersound_stayhome_timewindow=(10.5, 11)
    aftersound_stayhome_timewindow=None
    soundplay_finished_time=10
    soundplay_delay_time=12
    return (
        aftersound_stayhome_timewindow,
        min_leave_frames,
        min_stayhome_frames,
        pk_spd_time_afterleave,
        soundplay_delay_time,
        soundplay_finished_time,
    )


@app.cell
def _():
    figures_save_dir = "/mnt/e/data/LeciLab/behavioral_data/tmp/escape/classified_trials/"
    return (figures_save_dir,)


@app.cell
def _(
    figures_save_dir,
    ns_behav_df_dic_prebin_escape,
    ns_behav_df_dic_prebin_noreaction,
    ns_session_df_dic_classified,
    os,
    plot_escape_classification_overview,
    s_behav_df_dic_prebin_escape,
    s_behav_df_dic_prebin_noreaction,
    s_session_df_dic_classified,
):
    _escape_classification_overview_fig, _ = plot_escape_classification_overview(
        s_session_df_dic_classified,
        ns_session_df_dic_classified,
        s_behav_df_dic_prebin_escape,
        s_behav_df_dic_prebin_noreaction,
        ns_behav_df_dic_prebin_escape,
        ns_behav_df_dic_prebin_noreaction,
    )
    _escape_classification_overview_fig.savefig(
        os.path.join(
            figures_save_dir,
            "escape_classification_overview.svg",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## classified trials by condition
    """)
    return


@app.cell
def _(
    figures_save_dir,
    ns_behav_df_dic_prebin_escape_split,
    ns_behav_df_dic_prebin_noreaction_split,
    ns_session_df_dic_classified_split,
    os,
    plot_escape_classification_overview,
    s_behav_df_dic_prebin_escape_split,
    s_behav_df_dic_prebin_noreaction_split,
    s_session_df_dic_classified_split,
):
    # all the plots, the big overview figure
    for _inj_condition in ["hm3_saline", "hm3_dcz", "hm4_saline", "hm4_dcz"]:
        _s_session_df_dic_classified_split = s_session_df_dic_classified_split.get(
            _inj_condition, {}
        )
        _ns_session_df_dic_classified_split = ns_session_df_dic_classified_split.get(
            _inj_condition, {}
        )
        _s_behav_df_dic_prebin_escape_split = s_behav_df_dic_prebin_escape_split.get(
            _inj_condition, {}
        )
        _s_behav_df_dic_prebin_noreaction_split = s_behav_df_dic_prebin_noreaction_split.get(
            _inj_condition, {}
        )
        _ns_behav_df_dic_prebin_escape_split = ns_behav_df_dic_prebin_escape_split.get(
            _inj_condition, {}
        )
        _ns_behav_df_dic_prebin_noreaction_split = ns_behav_df_dic_prebin_noreaction_split.get(
            _inj_condition, {}
        )

        _escape_classification_overview_fig, _axes = plot_escape_classification_overview(
            _s_session_df_dic_classified_split,
            _ns_session_df_dic_classified_split,
            _s_behav_df_dic_prebin_escape_split,
            _s_behav_df_dic_prebin_noreaction_split,
            _ns_behav_df_dic_prebin_escape_split,
            _ns_behav_df_dic_prebin_noreaction_split,
        )
        _escape_classification_overview_fig.savefig(
            os.path.join(
                figures_save_dir,
                f"escape_classification_overview_{_inj_condition}.svg",
            )
        )
    return


@app.cell
def _(
    figures_save_dir,
    ns_session_df_dic_classified_split,
    os,
    plot_escape_classification_summary_plots,
    s_session_df_dic_classified_split,
):
    # all the plots, the big overview figure
    for _inj_condition in ["hm3_saline", "hm3_dcz", "hm4_saline", "hm4_dcz"]:
        _s_session_df_dic_classified_split = s_session_df_dic_classified_split.get(
            _inj_condition, {}
        )
        _ns_session_df_dic_classified_split = ns_session_df_dic_classified_split.get(
            _inj_condition, {}
        )

        _escape_classification_summary_fig = plot_escape_classification_summary_plots(
            _s_session_df_dic_classified_split,
            _ns_session_df_dic_classified_split,
        )
        _escape_classification_summary_fig.savefig(
            os.path.join(
                figures_save_dir,
                f"escape_classification_summary_{_inj_condition}.svg",
            )
        )
    return


@app.cell
def _(np, pd, session_axis_lookup):
    def build_escape_frequency_summary_df(
        session_df_dic_classified_split,
        sound_label,
    ):
        rows = []

        for group_condition, session_dic in session_df_dic_classified_split.items():
            group, condition = group_condition.split("_", 1)

            for dict_key, session_df in session_dic.items():
                if session_df.empty or "escape" not in session_df.columns:
                    continue

                if "session" in session_df.columns and session_df["session"].notna().any():
                    session_name = session_df["session"].dropna().iloc[0]
                else:
                    session_name = dict_key

                escape_count = int(session_df["escape"].sum())
                total_count = int(len(session_df))
                non_escape_count = total_count - escape_count

                axis_info = session_axis_lookup.get(session_name, {})

                rows.append(
                    {
                        "session": session_name,
                        "dict_key": dict_key,
                        "subject": session_name[:6],
                        "group": group,
                        "condition": condition,
                        "sound": sound_label,
                        "escape_count": escape_count,
                        "non_escape_count": non_escape_count,
                        "total_count": total_count,
                        "escape_frequency": (
                            escape_count / total_count
                            if total_count > 0
                            else np.nan
                        ),
                        "day_label": axis_info.get("label", np.nan),
                        "day_order": axis_info.get("order", np.nan),
                        "pair_index": axis_info.get("pair_index", np.nan),
                        "pair_id": axis_info.get("pair_id", np.nan),
                    }
                )

        return pd.DataFrame(rows)

    return (build_escape_frequency_summary_df,)


@app.cell
def _(
    build_escape_frequency_summary_df,
    np,
    ns_session_df_dic_classified_split,
    pd,
    s_session_df_dic_classified_split,
    session_axis_lookup,
):
    s_escape_frequency_df = build_escape_frequency_summary_df(
        s_session_df_dic_classified_split,
        sound_label="sound",
    )

    ns_escape_frequency_df = build_escape_frequency_summary_df(
        ns_session_df_dic_classified_split,
        sound_label="no_sound",
    )

    escape_frequency_df = pd.concat(
        [s_escape_frequency_df, ns_escape_frequency_df],
        ignore_index=True,
    )

    escape_frequency_df["day_label"] = escape_frequency_df["session"].map(
        lambda name: session_axis_lookup.get(name, {}).get("label", np.nan)
    )

    escape_frequency_df["day_order"] = escape_frequency_df["session"].map(
        lambda name: session_axis_lookup.get(name, {}).get("order", np.nan)
    )

    escape_frequency_df["pair_index"] = escape_frequency_df["session"].map(
        lambda name: session_axis_lookup.get(name, {}).get("pair_index", np.nan)
    )

    escape_frequency_df["pair_id"] = escape_frequency_df["session"].map(
        lambda name: session_axis_lookup.get(name, {}).get("pair_id", np.nan)
    )
    escape_frequency_df["frequency of (escape - noescape)"] = (escape_frequency_df['escape_count'] - escape_frequency_df['non_escape_count'])/escape_frequency_df['total_count']
    return (escape_frequency_df,)


@app.cell
def _(escape_frequency_df):
    pair_diff = (
        escape_frequency_df
        .pivot_table(
            index=["pair_id", "condition", "group"],
            columns="sound",
            values="escape_frequency",
        )
        .rename_axis(None, axis=1)
        .reset_index()
    )

    pair_diff["frequency_difference_between_sound_nosound"] = (
        pair_diff["sound"] - pair_diff["no_sound"]
    )

    pair_diff
    return (pair_diff,)


@app.cell
def _(escape_frequency_df, figures_save_dir, os, plt, sns):

    _g = sns.relplot(
        data=escape_frequency_df,
        x="day_order",
        y="escape_frequency",
        hue="sound",
        col="group",
        kind="line",
        marker="o",
        linewidth=2,
        height=4,
        aspect=1.2,
        palette={"sound": "firebrick", "no_sound": "steelblue"}
    )

    _g.set_axis_labels("Day order", "Escape frequency")
    _g.set_titles("{col_name}")

    for ax in _g.axes.flat:
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)

    _g.savefig(os.path.join(figures_save_dir, "escape_frequency_by_day.svg"))
    plt.show()
    return


@app.cell
def _(figures_save_dir, os, pair_diff, plt, sns):
    _fig, _ax = plt.subplots(figsize=(5, 4), dpi=300)
    _g = sns.pointplot(
        data=pair_diff,
        x="condition",
        y="frequency_difference_between_sound_nosound",
        hue='group',
        markers=["o", "s"],
        order=["saline", "dcz"], 
        # linestyle=["-", "--"],
        palette={
            "hm3": "orange",
            "hm4": "forestgreen",
        },
        dodge=0.1,
        linewidth=2,
        estimator='mean', 
        errwidth = 2,
        ci='sd',
        errorbar=('ci', 95),
        capsize=0.1,
        ax = _ax
    )
    _fig.savefig(os.path.join(figures_save_dir, "escape_frequency_difference_between_sound_nosound.svg"))
    _g
    return


@app.cell
def _(pair_diff, plt, sns, stats):
    _fig, _ax = plt.subplots(figsize=(6, 4), dpi=300)

    _g = sns.pointplot(
        data=pair_diff,
        x="condition",
        y="frequency_difference_between_sound_nosound",
        hue="group",
        markers=["o", "s"],
        order=["saline", "dcz"],
        hue_order=["hm3", "hm4"],
        palette={
            "hm3": "orange",
            "hm4": "forestgreen",
        },
        dodge=0.25,
        linewidth=2,
        estimator="mean",
        errorbar=("ci", 95),
        capsize=0.1,
        ax=_ax,
    )

    def p_to_stars(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    _y_col = "frequency_difference_between_sound_nosound"
    _hue_offsets = {
        "hm3": -0.12,
        "hm4": 0.12,
    }
    _colors = {
        "hm3": "orange",
        "hm4": "forestgreen",
    }

    _ymin, _ymax = _ax.get_ylim()
    _yrange = _ymax - _ymin
    _base_y = pair_diff[_y_col].max()
    _step = _yrange * 0.12
    _bar_h = _yrange * 0.03

    for _i, _group in enumerate(["hm3", "hm4"]):
        _group_df = pair_diff[pair_diff["group"] == _group].copy()

        _index_cols = ["pair_id"]
        if "subject" in _group_df.columns:
            _index_cols = ["subject", "pair_id"]

        _wide = (
            _group_df
            .pivot_table(
                index=_index_cols,
                columns="condition",
                values=_y_col,
                aggfunc="mean",
            )
            .dropna(subset=["saline", "dcz"])
        )

        if len(_wide) < 2:
            continue

        _stat, _p = stats.ttest_rel(_wide["saline"], _wide["dcz"])
        _stars = p_to_stars(_p)

        _x1 = 0 + _hue_offsets[_group]   # saline
        _x2 = 1 + _hue_offsets[_group]   # dcz
        _y = _base_y + (_i + 1) * _step

        _ax.plot(
            [_x1, _x1, _x2, _x2],
            [_y, _y + _bar_h, _y + _bar_h, _y],
            color=_colors[_group],
            linewidth=1.5,
        )
        _ax.text(
            (_x1 + _x2) / 2,
            _y + _bar_h,
            _stars,
            ha="center",
            va="bottom",
            color=_colors[_group],
            fontsize=12,
            fontweight="bold",
        )

    _ax.set_ylim(_ymin, _base_y + 3 * _step)
    _ax.set_xlabel("")
    _ax.set_ylabel("Sound - no sound escape frequency")
    _ax.set_title("Paired saline vs DCZ")

    plt.tight_layout()
    _g
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## classified trials by day
    """)
    return


@app.cell
def _(
    frame_xy,
    home_zone_x_threshold,
    mpl,
    ns_behav_df_dic_prebin_escape,
    ns_behav_df_dic_prebin_noreaction,
    pd,
    plot_test,
    plt,
    s_behav_df_dic_prebin_escape,
    s_behav_df_dic_prebin_noreaction,
    trigger_zone_x_threshold,
    x_len_tran,
    y_len_tran,
):
    def _get_timestamp_from_escape_window_df(window_df):
        if ("timestamp", "") in window_df.columns:
            return pd.to_numeric(window_df[("timestamp", "")], errors="coerce")
        return pd.to_numeric(window_df["timestamp"], errors="coerce")

    def _plot_window_dic_trajectories_on_ax(
        behav_df_dic,
        ax,
        title,
        cmap,
        start_color,
        time_after=10,
    ):
        _all_speed = []

        for _session_name, _trial_dic in behav_df_dic.items():
            for _trial, _window_df_list in _trial_dic.items():
                for _window_df in _window_df_list:
                    _timestamp = _get_timestamp_from_escape_window_df(
                        _window_df
                    )
                    _time_mask = (_timestamp >= 0) & (_timestamp <= time_after)
                    _speed = pd.to_numeric(
                        _window_df.loc[
                            _time_mask,
                            ("Center", "mean_speed"),
                        ],
                        errors="coerce",
                    ).dropna()
                    if not _speed.empty:
                        _all_speed.append(_speed)

        if not _all_speed:
            ax.text(
                0.5,
                0.5,
                "No trajectory data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)
            return

        _all_speed = pd.concat(_all_speed)
        _vmin = _all_speed.quantile(0.01)
        _vmax = _all_speed.quantile(0.99)
        _norm = mpl.colors.Normalize(vmin=_vmin, vmax=_vmax)

        for _session_name, _trial_dic in behav_df_dic.items():
            for _trial, _window_df_list in _trial_dic.items():
                for _window_df in _window_df_list:
                    _timestamp = _get_timestamp_from_escape_window_df(
                        _window_df
                    )
                    _time_mask = (_timestamp >= 0) & (_timestamp <= time_after)

                    _df_traj = _window_df.loc[
                        _time_mask,
                        "Center",
                    ][["x", "y", "mean_speed"]].copy()

                    _df_traj[["x", "y", "mean_speed"]] = (
                        _df_traj[["x", "y", "mean_speed"]]
                        .apply(pd.to_numeric, errors="coerce")
                        .interpolate(limit_direction="both")
                    )
                    _df_traj = _df_traj.dropna(
                        subset=["x", "y", "mean_speed"]
                    )

                    if _df_traj.empty:
                        continue

                    plot_test.plot_traj_speed(
                        _df_traj,
                        cmap=cmap,
                        ax=ax,
                        norm=_norm,
                    )
                    ax.scatter(
                        _df_traj["x"].iloc[0],
                        _df_traj["y"].iloc[0],
                        color=start_color,
                        s=80,
                        marker="o",
                        edgecolors="k",
                        zorder=3,
                    )

        ax.set_xlim(0, frame_xy[0] / x_len_tran)
        ax.set_ylim(0, frame_xy[1] / y_len_tran)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.axvline(
            x=trigger_zone_x_threshold,
            c="magenta",
            linestyle="--",
            linewidth=2,
        )
        ax.axvline(
            x=home_zone_x_threshold,
            c="green",
            linestyle="--",
            linewidth=2,
        )
        ax.set_title(title)

        _sm = mpl.cm.ScalarMappable(norm=_norm, cmap=cmap)
        _cbar = plt.colorbar(
            _sm,
            orientation="vertical",
            ax=ax,
            shrink=0.4,
            pad=0.02,
        )
        _cbar.set_label("speed cm/s", rotation=90)

    escape_trajectory_overview_fig, _axes = plt.subplots(
        2,
        2,
        figsize=(18, 10),
        dpi=150,
    )

    _plot_window_dic_trajectories_on_ax(
        s_behav_df_dic_prebin_escape,
        _axes[0, 0],
        "Sound-play escape trajectories: 0-10s",
        "inferno",
        "w",
        time_after=10,
    )
    _plot_window_dic_trajectories_on_ax(
        s_behav_df_dic_prebin_noreaction,
        _axes[0, 1],
        "Sound-play no-reaction trajectories: 0-10s",
        "inferno",
        "w",
        time_after=10,
    )
    _plot_window_dic_trajectories_on_ax(
        ns_behav_df_dic_prebin_escape,
        _axes[1, 0],
        "No-sound escape trajectories: 0-10s",
        "viridis",
        "k",
        time_after=10,
    )
    _plot_window_dic_trajectories_on_ax(
        ns_behav_df_dic_prebin_noreaction,
        _axes[1, 1],
        "No-sound no-reaction trajectories: 0-10s",
        "viridis",
        "k",
        time_after=10,
    )

    escape_trajectory_overview_fig.savefig("/mnt/e/data/LeciLab/behavioral_data/tmp/escape/classified_trials/classified_escapeor not_trajectory.png")
    escape_trajectory_overview_fig.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
