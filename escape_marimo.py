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
    # mean spead comparison
    """)
    return


@app.cell
def _(
    analyzed_video_names,
    behav_df_filtered_dic,
    behavior_utils,
    hM3Dq_mice,
    hM4Di_mice,
    np,
    pd,
    session_df_dic,
):
    timeBin = 10


    def build_mean_speed_dictionaries(mice):
        ns_mean_speed_dic = {}
        s_mean_speed_dic = {}

        for name in sorted(analyzed_video_names):
            if name[:6] not in mice or name not in session_df_dic:
                continue

            behav_df_filtered = behav_df_filtered_dic[name]
            if ("Center", "mean_speed") not in behav_df_filtered.columns:
                behav_df_filtered = behavior_utils.compute_distance_speed(
                    behav_df_filtered,
                    bodyparts=["Center"],
                )

            if ("timestamp", "") in behav_df_filtered.columns:
                timestamp = behav_df_filtered[("timestamp", "")]
            else:
                timestamp = behav_df_filtered["timestamp"]

            session_df = session_df_dic[name]
            ns_mean_speed = []
            s_mean_speed = []

            for trial in session_df["trial"].dropna().unique():
                trial_row = session_df[session_df["trial"] == trial]

                # ---- no-sound windows
                if "sound_not_played" not in trial_row.columns:
                    no_sound_list = []
                elif pd.isna(trial_row["sound_not_played"].iloc[0]):
                    no_sound_list = []
                else:
                    no_sound_list = eval(
                        trial_row["sound_not_played"].iloc[0]
                    )

                for t0 in no_sound_list:
                    df_ns = behav_df_filtered[
                        (timestamp >= t0)
                        & (timestamp <= t0 + timeBin)
                    ]["Center"].dropna()
                    mean_speed = df_ns["mean_speed"].mean()
                    if np.isfinite(mean_speed):
                        ns_mean_speed.append(mean_speed)

                # ---- sound window
                if (
                    "sound_played" not in trial_row.columns
                    or pd.isna(trial_row["sound_played"].iloc[0])
                ):
                    continue

                sound_play_list = eval(trial_row["sound_played"].iloc[0])
                if not sound_play_list:
                    continue

                sound_play_start = sound_play_list[0]
                df_s = behav_df_filtered[
                    (timestamp >= sound_play_start)
                    & (timestamp <= sound_play_start + timeBin)
                ]["Center"].dropna()
                mean_speed = df_s["mean_speed"].mean()
                if np.isfinite(mean_speed):
                    s_mean_speed.append(mean_speed)

            ns_mean_speed_dic[name] = ns_mean_speed
            s_mean_speed_dic[name] = s_mean_speed

        return ns_mean_speed_dic, s_mean_speed_dic


    ns_meanSpeed_dic_hm3, s_meanSpeed_dic_hm3 = (
        build_mean_speed_dictionaries(hM3Dq_mice)
    )
    ns_meanSpeed_dic_hm4, s_meanSpeed_dic_hm4 = (
        build_mean_speed_dictionaries(hM4Di_mice)
    )


    def mean_speed_dictionaries_to_long(ns_dic, s_dic):
        session_names = sorted(set(ns_dic) | set(s_dic))
        session_day = {}

        for subject in sorted({name[:6] for name in session_names}):
            subject_sessions = sorted(
                name for name in session_names if name[:6] == subject
            )
            for day_index, name in enumerate(subject_sessions, start=1):
                session_day[name] = f"day{day_index}"

        rows = []
        for condition, speed_dic in [
            ("no_sound", ns_dic),
            ("sound", s_dic),
        ]:
            for name, speeds in speed_dic.items():
                for speed in speeds:
                    rows.append(
                        {
                            "session": name,
                            "subject": name[:6],
                            "day": session_day[name],
                            "cond": condition,
                            "speed": speed,
                        }
                    )

        return pd.DataFrame(rows)


    df_meanSpeed_long_hm3 = mean_speed_dictionaries_to_long(
        ns_meanSpeed_dic_hm3,
        s_meanSpeed_dic_hm3,
    )
    df_meanSpeed_long_hm4 = mean_speed_dictionaries_to_long(
        ns_meanSpeed_dic_hm4,
        s_meanSpeed_dic_hm4,
    )
    return df_meanSpeed_long_hm3, df_meanSpeed_long_hm4, timeBin


@app.cell
def _(df_meanSpeed_long_hm3, df_meanSpeed_long_hm4, mo, plt, sns, timeBin):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch


    def plot_mean_speed_comparison(df, group_name):
        df = df.copy()

        day_order = sorted(
            df["day"].unique(),
            key=lambda x: int(x.removeprefix("day")),
        )
        mouse_order = sorted(df["subject"].unique())
        cond_order = ["no_sound", "sound"]

        palette = {
            "no_sound": "#3CB371",
            "sound": "#FA8072",
        }
        markers = ["o", "s", "^", "D", "P", "X", "v"]
        mouse_markers = {
            mouse: markers[i % len(markers)]
            for i, mouse in enumerate(mouse_order)
        }

        # Each mouse-condition combination gets its own box
        df["mouse_cond"] = df["subject"] + "_" + df["cond"]

        hue_order = [
            f"{mouse}_{cond}"
            for mouse in mouse_order
            for cond in cond_order
        ]
        nested_palette = {
            f"{mouse}_{cond}": palette[cond]
            for mouse in mouse_order
            for cond in cond_order
        }

        fig, ax = plt.subplots(
            figsize=(max(10, len(day_order) * 2.5), 5),
            dpi=150,
        )

        sns.boxplot(
            data=df,
            x="day",
            y="speed",
            hue="mouse_cond",
            order=day_order,
            hue_order=hue_order,
            palette=nested_palette,
            fill=False,
            width=0.8,
            showfliers=False,
            ax=ax,
        )

        # Plot each mouse separately to give it a unique marker
        for mouse in mouse_order:
            mouse_df = df[df["subject"] == mouse]

            sns.stripplot(
                data=mouse_df,
                x="day",
                y="speed",
                hue="mouse_cond",
                order=day_order,
                hue_order=hue_order,
                palette=nested_palette,
                dodge=True,
                jitter=0.12,
                marker=mouse_markers[mouse],
                size=6,
                alpha=0.6,
                edgecolor="black",
                linewidth=0.4,
                legend=False,
                ax=ax,
            )

        # Condition color legend
        condition_handles = [
            Patch(
                facecolor="none",
                edgecolor=palette[cond],
                linewidth=2,
                label=cond,
            )
            for cond in cond_order
        ]
        condition_legend = ax.legend(
            handles=condition_handles,
            title="Condition",
            frameon=False,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        ax.add_artist(condition_legend)

        # Mouse marker legend
        mouse_handles = [
            Line2D(
                [0], [0],
                marker=mouse_markers[mouse],
                linestyle="none",
                markerfacecolor="gray",
                markeredgecolor="black",
                markersize=7,
                label=mouse,
            )
            for mouse in mouse_order
        ]
        ax.legend(
            handles=mouse_handles,
            title="Mouse",
            frameon=False,
            bbox_to_anchor=(1.02, 0.55),
            loc="upper left",
        )

        ax.set_xlabel("Day")
        ax.set_ylabel(
            f"Mean speed per trial in first {timeBin}s (pixels/s)"
        )
        ax.set_title(group_name)

        fig.tight_layout()
        return fig


    mean_speed_fig_hm3 = plot_mean_speed_comparison(
        df_meanSpeed_long_hm3,
        "hM3Dq",
    )
    mean_speed_fig_hm4 = plot_mean_speed_comparison(
        df_meanSpeed_long_hm4,
        "hM4Di",
    )

    mo.vstack([mean_speed_fig_hm3, mean_speed_fig_hm4])
    return


@app.cell
def _(df_meanSpeed_long_hm3, df_meanSpeed_long_hm4, mo, plt):
    def calculate_mouse_day_speed_diff(df_mean_speed_long):
        condition_mean = (
            df_mean_speed_long
            .groupby(["subject", "day", "cond"], as_index=False)["speed"]
            .mean()
        )

        speed_diff_df = (
            condition_mean
            .pivot(
                index=["subject", "day"],
                columns="cond",
                values="speed",
            )
            .reset_index()
        )

        speed_diff_df["speed_diff"] = (
            speed_diff_df["sound"]
            - speed_diff_df["no_sound"]
        )

        speed_diff_df["day_number"] = (
            speed_diff_df["day"]
            .str.extract(r"(\d+)", expand=False)
            .astype(int)
        )

        return speed_diff_df.sort_values(["subject", "day_number"])


    def plot_mouse_speed_diff(speed_diff_df, group_name):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

        for subject, mouse_df in speed_diff_df.groupby("subject"):
            mouse_df = mouse_df.dropna(subset=["speed_diff"])

            ax.plot(
                mouse_df["day_number"],
                mouse_df["speed_diff"],
                marker="o",
                linewidth=2,
                markersize=7,
                label=subject,
            )

        day_numbers = sorted(speed_diff_df["day_number"].unique())

        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(day_numbers)
        ax.set_xticklabels([f"day{day}" for day in day_numbers])
        ax.set_xlabel("Day")
        ax.set_ylabel("Mean speed difference (sound - no sound), pixels/s")
        ax.set_title(group_name)
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

        fig.tight_layout()
        return fig


    speed_diff_hm3 = calculate_mouse_day_speed_diff(
        df_meanSpeed_long_hm3
    )
    speed_diff_hm4 = calculate_mouse_day_speed_diff(
        df_meanSpeed_long_hm4
    )

    speed_diff_fig_hm3 = plot_mouse_speed_diff(
        speed_diff_hm3,
        "hM3Dq",
    )
    speed_diff_fig_hm4 = plot_mouse_speed_diff(
        speed_diff_hm4,
        "hM4Di",
    )

    mo.vstack([speed_diff_fig_hm3, speed_diff_fig_hm4])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # speed change during trials
    """)
    return


@app.cell
def _(session_df_dic):
    session_df_dic
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
    pre_time_bin = 2


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
):
    # speed change trial by trial
    trial_speed_time_bin = 15
    trial_speed_fps = 30
    trial_speed_grid = np.arange(-pre_time_bin, trial_speed_time_bin, 0.01)


    def organize_mouse_day_traces(sound_dic, no_sound_dic):
        organized = {}
        all_names = sorted(set(sound_dic) | set(no_sound_dic))

        for mouse in sorted({name[:6] for name in all_names}):
            mouse_names = sorted(
                name for name in all_names if name[:6] == mouse
            )
            dates = sorted({name.rsplit("_", 2)[-2] for name in mouse_names})
            date_to_day = {
                date: day for day, date in enumerate(dates, start=1)
            }

            organized[mouse] = {}
            for name in mouse_names:
                date = name.rsplit("_", 2)[-2]
                day = date_to_day[date]
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

        return organized


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
        organized = organize_mouse_day_traces(sound_dic, no_sound_dic)
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
                        trial_ax.set_title(f"day{day}", fontsize=10)
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
            date_to_day = {
                date: day for day, date in enumerate(dates, start=1)
            }
            counts_by_day = {
                day: {"sound": 0, "no_sound": 0}
                for day in date_to_day.values()
            }

            for name in mouse_names:
                date = name.rsplit("_", 2)[-2]
                day = date_to_day[date]
                counts_by_day[day]["sound"] += len(
                    sound_dic.get(name, {})
                )
                counts_by_day[day]["no_sound"] += len(
                    no_sound_dic.get(name, {})
                )

            for day, counts in counts_by_day.items():
                rows.append(
                    {
                        "subject": mouse,
                        "day": day,
                        "sound": counts["sound"],
                        "no_sound": counts["no_sound"],
                        "total": counts["sound"] + counts["no_sound"],
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
            ].sort_values("day")
            for column, color, _ in conditions:
                ax.plot(
                    mouse_df["day"],
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

        days = sorted(trial_count_df["day"].unique())
        ax.set_xticks(days)
        ax.set_xticklabels([f"day{day}" for day in days])
        ax.yaxis.set_major_locator(_MaxNLocator(integer=True))
        ax.set_xlabel("Day")
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


if __name__ == "__main__":
    app.run()
