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

    from lecilab_behavior_analysis import utils as utils
    utils.IDIBAPS_TV_PROJECTS = "/storage/training_village/"

    from lecilab_behavior_analysis import df_transforms as dft
    from lecilab_behavior_analysis import plots
    from lecilab_behavior_analysis.figure_maker import (
        session_summary_figure,
        subject_progress_figure,
    )

    warnings.filterwarnings("ignore")
    return Path, behavior_utils, mpl, np, pd, plot_test, plt, sns, utils


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
        figure_folder = figure_export_root / _safe_figure_file_stem(folder_name)
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
    mouse_select = ['NUO003', 'NUO060', 'NUO061', 'NUO064', 'NUO065']
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

        local_path_dlc_sub = Path(local_path, "behavioral_data", project, "DLC_output", sub)
        local_path_session_sub = Path(local_path, "behavioral_data", project, "sessions", sub)
        local_path_video_sub = Path(local_path, "behavioral_data", project, "videos", sub)

        local_path_dlc_sub.mkdir(parents=True, exist_ok=True)
        local_path_session_sub.mkdir(parents=True, exist_ok=True)
        local_path_video_sub.mkdir(parents=True, exist_ok=True)

        if download_button.value:
            for f in behavior_files:
                if f.endswith(".csv") and (sub in f) and ("DLC" in f):
                    utils.rsync_specific_file(
                        credentials=credential,
                        file_path=parent_path + f,
                        local_path=local_path_dlc_sub,
                    )

        for csv_path in local_path_dlc_sub.glob("*DLC*.csv"):
            if (csv_path.name[:37] not in deleted_sessions_list) and (sub in csv_path.name):
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

        if download_button.value:
            session_data_path_subject = session_data_path + "/" + subject
            session_files = utils.get_folders_from_server(
                credentials=credential,
                path=session_data_path_subject,
            )

            for f in session_files:
                if f.endswith(".csv") and ("RAW" not in f) and (name in f):
                    utils.rsync_specific_file(
                        credentials=credential,
                        file_path=session_data_path_subject + "/" + f,
                        local_path=local_path_session_sub,
                    )

            video_data_path_subject = video_data_path + "/" + subject
            video_files = utils.get_folders_from_server(
                credentials=credential,
                path=video_data_path_subject,
            )

            for f in video_files:
                if f.endswith(".csv") and (name in f):
                    utils.rsync_specific_file(
                        credentials=credential,
                        file_path=video_data_path_subject + "/" + f,
                        local_path=local_path_video_sub,
                    )

        session_df = pd.read_csv(
            local_path_session_sub / f"{name}.csv",
            sep=";",
        )
        session_df_dic[name] = session_df

        video_df = pd.read_csv(
            local_path_video_sub / f"{name}.csv",
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
):
    behav_df_filtered_dic = {}
    for name1 in analyzed_video_names:
        df = behav_df_dic[name1]
        df['timestamp'] = video_df_dic[name1]['timestamp']
        df = behavior_utils.preprocess_positions(df, likelihood_thr=0.83, distance_thr=200, max_iter=100)
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
def _(
    analyzed_video_names,
    behav_df_filtered_dic,
    hM3Dq_mice,
    hM4Di_mice,
    pd,
    session_df_dic,
):
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

    def build_behavior_window_dataframes(mice, time_bin=15):
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
                        mask = (timestamp >= t0) & (timestamp <= t0 + time_bin)
                        window_df = behav_df.loc[mask].copy()
                        if not window_df.empty:
                            window_df.attrs["trigger_time"] = float(t0)
                            no_sound_by_trial.setdefault(trial, []).append(
                                window_df
                            )

                if "sound_played" in session_df.columns:
                    sound_times = get_event_times(trial_row["sound_played"])
                    for t0 in sound_times:
                        mask = (timestamp >= t0) & (timestamp <= t0 + time_bin)
                        window_df = behav_df.loc[mask].copy()
                        if not window_df.empty:
                            window_df.attrs["trigger_time"] = float(t0)
                            sound_by_trial.setdefault(trial, []).append(window_df)

            s_behav_df_dic[name] = sound_by_trial
            ns_behav_df_dic[name] = no_sound_by_trial

        return s_behav_df_dic, ns_behav_df_dic

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
    # plot trajectory
    """)
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    mo,
    mouse_dropdown,
    mpl,
    pd,
    plot_test,
    plt,
    session_df_dic,
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
        ax.set_xlim(0, 640)
        ax.set_ylim(0, 480)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.axvline(x=364, c="k", linestyle="--", linewidth=2)


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
                        (timestamp >= trigger_time)
                        & (timestamp <= trigger_time + time_bin)
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
    session_axis_lookup,
):
    # speed change trial by trial
    trial_speed_time_bin = 15
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

                for trial_index in range(n_mini):
                    trial_ax = fig.add_subplot(inner_grid[trial_index, 0])
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

                    if day_col == 0:
                        trial_ax.set_ylabel(
                            f"T{trial_data['trial']}",
                            rotation=0,
                            ha="right",
                            va="center",
                            fontsize=6,
                        )

                summary_ax = fig.add_subplot(inner_grid[-1, 0])
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
                if day_col == 0:
                    summary_ax.set_ylabel(
                        f"{mouse}\n{value_label}",
                        fontsize=8,
                    )
                else:
                    summary_ax.set_yticklabels([])
                summary_ax.tick_params(labelsize=7)
                if sound_traces or no_sound_traces:
                    summary_ax.legend(fontsize=6, frameon=False)

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
    )
    speed_trial_fig_hm4 = plot_mouse_trial_grid(
        s_speed_change_dic_hm4_prebin,
        ns_speed_change_dic_hm4_prebin,
        "hM4Di",
        value_label="Mean speed",
        metric_name="speed",
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
    stationary_speed_threshold = 10

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
                (timestamp >= trigger_time)
                & (timestamp <= trigger_time + time_bin)
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

    x_position_time_bin = 15


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
    mo,
    ns_x_position_dic_hm3_prebin,
    ns_x_position_dic_hm4_prebin,
    plot_mouse_trial_grid,
    s_x_position_dic_hm3_prebin,
    s_x_position_dic_hm4_prebin,
):
    x_position_trial_fig_hm3 = plot_mouse_trial_grid(
        s_x_position_dic_hm3_prebin,
        ns_x_position_dic_hm3_prebin,
        "hM3Dq",
        value_label="X position (pixels)",
        metric_name="x position",
        y_limits=(0, 640),
        horizontal_reference=364,
    )
    x_position_trial_fig_hm4 = plot_mouse_trial_grid(
        s_x_position_dic_hm4_prebin,
        ns_x_position_dic_hm4_prebin,
        "hM4Di",
        value_label="X position (pixels)",
        metric_name="x position",
        y_limits=(0, 640),
        horizontal_reference=364,
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
    hM3Dq_mice,
    hM4Di_mice,
    mo,
    np,
    pd,
    plt,
    session_df_dic,
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
                    y=364,
                    c="k",
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
                ax.set_ylim(0, 640)
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
):
    trigger_zone_x_threshold = 364

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
            (timestamp >= trigger_time)
            & (timestamp <= trigger_time + time_bin)
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
        "Trigger-zone time ratio (x > 364)",
    )
    trigger_zone_time_ratio_fig_hm4 = plot_ratio_boxplot(
        df_trigger_zone_time_ratio_hm4,
        "hM4Di: first 15s",
        "Trigger-zone time ratio (x > 364)",
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
    # sound intensity in behavior
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
    return


if __name__ == "__main__":
    app.run()
