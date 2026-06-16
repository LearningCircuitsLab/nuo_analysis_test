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
    from datetime import datetime
    import behavior_utils
    import plot_test

    from lecilab_behavior_analysis import utils as utils
    utils.IDIBAPS_TV_PROJECTS = "/archive/training_village/"

    from lecilab_behavior_analysis import df_transforms as dft
    from lecilab_behavior_analysis import plots
    from lecilab_behavior_analysis.figure_maker import (
        session_summary_figure,
        subject_progress_figure,
    )

    warnings.filterwarnings("ignore")
    return Path, behavior_utils, np, pd, plot_test, plt, utils


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
def _(Path, credential, download_button, mouse_select, pd, project, utils):
    behav_df_dic = {}

    parent_path = "/home/kudongdong/data/behavior_DLC/escape_auditory-LeciLab-2025-10-20/v2/"
    session_data_path = "/archive/training_village/auditory_escape_data/sessions"
    video_data_path = "/archive/training_village/auditory_escape_data/videos"

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
            if sub in csv_path.name:
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
    return (
        analyzed_video_names,
        behav_df_dic,
        name,
        session_df_dic,
        video_df_dic,
    )


@app.cell
def _(analyzed_video_names, name, session_df_dic):
    sound_play_dic = {}
    for _name in analyzed_video_names:
        sound_play_list = []
        for n in session_df_dic[_name]['sound_played']:
            sound_play_list.append(eval(n)[0])
        sound_play_dic[name] = sound_play_list
    return (sound_play_dic,)


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
def _(
    analyzed_video_names,
    behav_df_dic,
    bodyparts,
    np,
    preprocess_positions,
    sound_play_dic,
    video_df_dic,
):
    behav_df_filtered_dic = {}
    for name in analyzed_video_names:
        df = behav_df_dic[name]
        df['timestamp'] = video_df_dic[name]['timestamp']
        # input the sound play time in behavior data
        idx = np.searchsorted(df['timestamp'], sound_play_dic[name], side='right')
        df['sound_played'] = np.nan
        df.loc[idx, 'sound_played'] = sound_play_dic[name]
        df = preprocess_positions(df, likelihood_thr=0.65, distance_thr=200, max_iter=100)
        behav_df_filtered_dic[name] = df

    # interpolate the positions
    for name in analyzed_video_names:
        behav_df_filtered = behav_df_filtered_dic[name]
        for bp in bodyparts:
            for coord in ["x", "y"]:
                behav_df_filtered[(bp, coord)] = (
                    behav_df_filtered[(bp, coord)]
                    .interpolate(method="linear", limit_direction="both")
                )
        behav_df_filtered_dic[name] = behav_df_filtered
    return (name,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # manual mouse select
    """)
    return


@app.cell
def _():
    # hM3Dq_mice = ['NUO062', 'NUO063', 'NUO064', 'NUO065', 'NUO007', 'NUO008', 'NUO009', 'NUO010', 'NUO011', 'NUO012']
    # hM4Di_mice = ['NUO057', 'NUO058', 'NUO059', 'NUO060', 'NUO061', 'NUO001', 'NUO002', 'NUO003', 'NUO005', 'NUO006']
    hM3Dq_mice = ['NUO064', 'NUO065', 'NUO007', 'NUO008', 'NUO009', 'NUO010', 'NUO011', 'NUO012']
    hM4Di_mice = ['NUO060', 'NUO061', 'NUO001', 'NUO002', 'NUO003', 'NUO005', 'NUO006']
    return hM3Dq_mice, hM4Di_mice


@app.cell
def _(pd):
    injection_info_file_path = "/mnt/e/data/LeciLab/behavioral_data/data_test/escape_test_record_nuo.xlsx"
    injection_info_df = pd.read_excel(injection_info_file_path)
    injection_info_df
    return (injection_info_df,)


@app.cell
def _(injection_info_df, pd):
    date_col = injection_info_df.columns[0]

    injection_dates = pd.to_datetime(
        injection_info_df[date_col],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    injection_info_with_dates = injection_info_df.copy()
    injection_info_with_dates["_injection_date"] = injection_dates
    injection_mouse_columns = [
        column for column in injection_info_df.columns if column != date_col
    ]
    injection_lookup = (
        injection_info_with_dates.melt(
            id_vars="_injection_date",
            value_vars=injection_mouse_columns,
            var_name="_mouse",
            value_name="_observations",
        )
        .set_index(["_injection_date", "_mouse"])["_observations"]
        .to_dict()
    )
    injection_lookup
    # trial_dates = pd.to_datetime(
    #     df_test_aud["date"],
    #     errors="coerce",
    # ).dt.strftime("%Y-%m-%d")
    # trial_mice = (
    #     df_test_aud["subject"].astype(str)
    #     if "subject" in df_test_aud.columns
    #     else pd.Series(mouse, index=df_test_aud.index)
    # )
    # df_test_aud["observations"] = [
    #     injection_lookup.get((trial_date, trial_mouse), pd.NA)
    #     for trial_date, trial_mouse in zip(trial_dates, trial_mice)
    # ]

    # df_test_aud['observations']
    return


@app.cell
def _(pd):
    def flatten_dlc_columns(columns):
        flat_columns = []
        for column in columns:
            if isinstance(column, tuple) and len(column) >= 3:
                if column[0] == "scorer" and column[1] == "bodyparts":
                    column_name = "frame"
                else:
                    column_name = f"{column[1]}_{column[2]}"
            else:
                column_name = str(column)

            if column_name in flat_columns:
                column_name = f"{column_name}_{flat_columns.count(column_name)}"
            flat_columns.append(column_name)
        return flat_columns

    def read_dlc_csv(dlc_file):
        dlc_df = pd.read_csv(dlc_file, header=[0, 1, 2])
        dlc_df.columns = flatten_dlc_columns(dlc_df.columns)
        if "frame" in dlc_df.columns:
            dlc_df["frame"] = pd.to_numeric(
                dlc_df["frame"],
                errors="coerce",
            ).astype("Int64")
        return dlc_df


    return


@app.cell
def _(mo):
    download_dlcData_button = mo.ui.run_button(label="Download / update the mice dlc data")
    download_dlcData_button
    return (download_dlcData_button,)


@app.cell
def _():
    dlc_mice_selected = ['NUO005', 'NUO008', 'NUO010', 'NUO012']
    return (dlc_mice_selected,)


@app.cell
def _(mo):
    load_paired_dlc_button = mo.ui.run_button(label="Load paired DLC data")
    load_paired_dlc_button
    return


@app.cell
def _(
    Path,
    behavior_utils,
    dlc_mice_selected,
    download_dlcData_button,
    injection_info_df,
    pd,
    project,
    utils,
):
    behav_df_dic = {}
    video_df_dic = {}

    if download_dlcData_button.value:
        for dlc_mouse_selected in dlc_mice_selected:
            _local_path = Path(utils.get_outpath()) / project / "sessions" / dlc_mouse_selected / "DLC"
            _local_path.mkdir(parents=True, exist_ok=True)

            utils.rsync_specific_file(
                file_path="/home/kudongdong/data/behavior_DLC/TrainingVillage_2AFC_superanimal-LeciLab-2025-10-16/v2/{}_*.csv".format(dlc_mouse_selected),
                local_path=str(_local_path),
                credentials=utils.get_idibaps_cluster_credentials(),
            )

    else:
        for dlc_mouse_selected in dlc_mice_selected:
            _local_path = Path(utils.get_outpath()) / project / "sessions" / dlc_mouse_selected / "DLC"
            for dlc_file_ in sorted(_local_path.glob("*DLC*.csv")):
                random_df = pd.read_csv(dlc_file_, header=[1, 2])
                behav_df_dic[dlc_file_.name[:29]] = random_df

    video_data_path = f'/storage/training_village/{project}/videos/'
    if download_dlcData_button.value:
        for dlc_mouse_selected in dlc_mice_selected:
            _local_path = Path(utils.get_outpath()) / project / "sessions" / dlc_mouse_selected / "DLC"
            for dlc_file_ in sorted(_local_path.glob("*DLC*.csv")):
                utils.rsync_specific_file(
                    file_path=Path(video_data_path, dlc_mouse_selected, dlc_file_.name[:29]+".csv"),
                    local_path=str(_local_path),
                    credentials=utils.get_idibaps_cluster_credentials(),
                )
    else:
        video_df_dic = {}
        for video_file_ in behav_df_dic:
            _local_path = Path(utils.get_outpath()) / project / "sessions" / video_file_[:6] / "DLC"
            video_df = pd.read_csv(Path(_local_path, video_file_+".csv"), sep=';', index_col='frame')
            video_df = video_df[2:]
            video_df.index = range(0, len(video_df))
            video_df_dic[video_file_] = video_df

    paired_dlc_dates = behavior_utils.get_paired_injection_dates(
        injection_info_df,
        mice_selected=dlc_mice_selected,
    )
    (
        behav_df_dic_saline,
        behav_df_dic_dcz,
        behav_pair_map,
        missing_behav_pairs,
    ) = behavior_utils.split_paired_behavior_dicts(behav_df_dic, paired_dlc_dates)
    (
        video_df_dic_saline,
        video_df_dic_dcz,
        video_pair_map,
        missing_video_pairs,
    ) = behavior_utils.split_paired_behavior_dicts(video_df_dic, paired_dlc_dates)

    behav_pair_map
    return (
        behav_df_dic,
        behav_df_dic_dcz,
        behav_df_dic_saline,
        behav_pair_map,
        video_df_dic,
        video_df_dic_dcz,
        video_df_dic_saline,
    )


@app.cell
def _(behavior_utils, plot_test):
    import importlib
    importlib.reload(behavior_utils)
    importlib.reload(plot_test)
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_utils,
    np,
    video_df_dic_dcz,
    video_df_dic_saline,
):
    def add_timestamp_from_end(behav_df, video_df):
        behav_df[("timestamp", "")] = np.nan

        n = min(len(behav_df), len(video_df))

        behav_df.iloc[
            -n:,
            behav_df.columns.get_loc(("timestamp", "")),
        ] = video_df["timestamp"].to_numpy()[-n:]

        return behav_df


    for pair_id in behav_pair_map["pair_id"].unique():
        behav_df_dic_dcz[pair_id] = add_timestamp_from_end(
            behav_df_dic_dcz[pair_id],
            video_df_dic_dcz[pair_id],
        )
        behav_df_dic_dcz[pair_id].dropna(subset=[("timestamp", "")], inplace=True)
        behav_df_dic_dcz[pair_id] = behavior_utils.preprocess_positions(
            behav_df_dic_dcz[pair_id],
            likelihood_thr=0.7,
            distance_thr=200,
            max_iter=100,
            speed_thr=600
        )

        behav_df_dic_saline[pair_id] = add_timestamp_from_end(
            behav_df_dic_saline[pair_id],
            video_df_dic_saline[pair_id],
        )
        behav_df_dic_saline[pair_id].dropna(subset=[("timestamp", "")], inplace=True)
        behav_df_dic_saline[pair_id] = behavior_utils.preprocess_positions(
            behav_df_dic_saline[pair_id],
            likelihood_thr=0.7,
            distance_thr=200,
            max_iter=100,
            speed_thr=600
        )

        behav_df_dic_dcz[pair_id] = behavior_utils.compute_distance_speed(
            behav_df_dic_dcz[pair_id],
            window_size=5,
        )
        behav_df_dic_saline[pair_id] = behavior_utils.compute_distance_speed(
            behav_df_dic_saline[pair_id],
            window_size=5,
        )
    return


@app.cell
def _():
    roi_bottom = 80
    roi_top = 190
    roi_left = 235
    roi_right = 400
    return roi_bottom, roi_left, roi_right, roi_top


@app.cell
def _(behav_pair_map):
    behav_pair_map
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # save tmp svg
    """)
    return


@app.cell
def _(Path):
    save_behavior_svg = True
    behavior_svg_dir = Path("/mnt/e/data/LeciLab/behavioral_data/tmp")
    return behavior_svg_dir, save_behavior_svg


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    mo,
    plot_test,
    roi_bottom,
    roi_left,
    roi_right,
    roi_top,
    save_behavior_svg,
):
    pair_figures = plot_test.plot_paired_behavior_figures(
        pair_ids=behav_pair_map["pair_id"].unique(),
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,
        bodypart="Center",
    )

    plot_test.save_figures_svg(
        pair_figures,
        "paired_behavior",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack(pair_figures)
    return


@app.cell
def _(
    behav_df_dic_saline,
    behavior_utils,
    roi_bottom,
    roi_left,
    roi_right,
    roi_top,
):
    behavior_utils.get_roi_time_ratio(
        behav_df_dic_saline['NUO005_pair_01'],
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,)
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_utils,
    roi_bottom,
    roi_left,
    roi_right,
    roi_top,
    video_df_dic_dcz,
    video_df_dic_saline,
):
    roi_time_ratio_summary = behavior_utils.paired_roi_time_ratio_comparison(
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        video_df_dic_saline=video_df_dic_saline,
        video_df_dic_dcz=video_df_dic_dcz,
        pair_map=behav_pair_map,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,
        bodypart="Center",
    )

    roi_time_ratio_summary
    return (roi_time_ratio_summary,)


@app.cell
def _(
    behavior_svg_dir,
    hM3Dq_mice,
    hM4Di_mice,
    plot_test,
    plt,
    roi_time_ratio_summary,
    save_behavior_svg,
    stats,
):
    roi_group_df = roi_time_ratio_summary.copy()

    if "subject" not in roi_group_df.columns:
        roi_group_df["subject"] = roi_group_df["pair_id"].str[:6]


    def plot_roi_group(ax, df, mice, group_name):
        group_df = df[
            df["subject"].isin(mice)
        ].dropna(subset=["saline", "DCZ"]).copy()

        for _, roi_row in group_df.iterrows():
            ax.plot(
                [0, 1],
                [roi_row["saline"], roi_row["DCZ"]],
                color="gray",
                alpha=0.4,
            )

        ax.scatter(
            [0] * len(group_df),
            group_df["saline"],
            color="blue",
            edgecolor="black",
            label="saline",
        )

        ax.scatter(
            [1] * len(group_df),
            group_df["DCZ"],
            color="red",
            edgecolor="black",
            label="DCZ",
        )

        if len(group_df) > 0:
            try:
                roi_p = stats.wilcoxon(
                    group_df["saline"],
                    group_df["DCZ"],
                ).pvalue
                p_text = f"p={roi_p:.3g}"
            except ValueError:
                roi_p = float("nan")
                p_text = "p=NA"
        else:
            roi_p = float("nan")
            p_text = "p=NA"

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["saline", "DCZ"])
        ax.set_ylabel("fraction of time in ROI")
        ax.set_title(f"{group_name}, n={len(group_df)}, {p_text}")
        plot_test.add_paired_significance_label(
            ax,
            group_df["saline"],
            group_df["DCZ"],
            plot_test.p_to_star(roi_p),
        )
        ax.grid(axis="y", alpha=0.3)

        return group_df


    roi_group_fig, roi_group_axes = plt.subplots(
        1,
        2,
        figsize=(8, 5),
        sharey=True,
    )

    roi_hm4_df = plot_roi_group(
        roi_group_axes[0],
        roi_group_df,
        hM4Di_mice,
        "hM4Di",
    )

    roi_hm3_df = plot_roi_group(
        roi_group_axes[1],
        roi_group_df,
        hM3Dq_mice,
        "hM3Dq",
    )

    roi_group_fig.tight_layout()
    plot_test.save_figure_svg(
        roi_group_fig,
        "roi_time_ratio_groups",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )
    plt.show()
    return


@app.cell
def _(behav_df_dic_dcz, behav_df_dic_saline, behav_pair_map, behavior_utils):
    stationary_speed_threshold = 10  # pixels/s调

    stationary_time_ratio_summary = behavior_utils.paired_stationary_time_ratio_comparison(
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        speed_threshold=stationary_speed_threshold,
        pair_map=behav_pair_map,
        bodypart="Center",
        speed_col="mean_speed",
    )

    stationary_time_ratio_summary
    return stationary_speed_threshold, stationary_time_ratio_summary


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    plot_test,
    save_behavior_svg,
    stationary_speed_threshold,
):
    stationary_speed_figures = plot_test.plot_paired_stationary_speed_traces(
        pair_ids=behav_pair_map["pair_id"].unique(),
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        speed_threshold=stationary_speed_threshold,
        bodypart="Center",
        speed_col="mean_speed",
    )

    plot_test.save_figures_svg(
        stationary_speed_figures,
        "stationary_speed_trace",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    stationary_speed_figures
    return


@app.cell
def _(
    behavior_svg_dir,
    hM3Dq_mice,
    hM4Di_mice,
    plot_test,
    save_behavior_svg,
    stationary_time_ratio_summary,
):
    stationary_group_fig, stationary_hm4_df, stationary_hm3_df = (
        plot_test.plot_stationary_time_ratio_groups(
            stationary_time_ratio_summary,
            hM4Di_mice=hM4Di_mice,
            hM3Dq_mice=hM3Dq_mice,
        )
    )
    plot_test.save_figure_svg(
        stationary_group_fig,
        "stationary_time_ratio_groups",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )
    stationary_group_fig
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    mo,
    plot_test,
    save_behavior_svg,
    stationary_speed_threshold,
):
    speed_distribution_figures = plot_test.plot_paired_speed_distributions(
        pair_ids=behav_pair_map["pair_id"].unique(),
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        bodypart="Center",
        speed_col="mean_speed",
        speed_threshold=stationary_speed_threshold,  
        bins=60,
        kde=True,
    )

    plot_test.save_figures_svg(
        speed_distribution_figures,
        "speed_distribution",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack(speed_distribution_figures)
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    hM3Dq_mice,
    hM4Di_mice,
    mo,
    plot_test,
    save_behavior_svg,
):
    speed_acf_group_figures, hm4_acf_pair_ids, hm3_acf_pair_ids = (
        plot_test.plot_speed_acfs_by_group(
            pair_ids=behav_pair_map["pair_id"].unique(),
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            hM4Di_mice=hM4Di_mice,
            hM3Dq_mice=hM3Dq_mice,
            fps=30,
            max_lag_sec=60,
            bodypart="Center",
            speed_col="mean_speed",
        )
    )

    plot_test.save_figures_svg(
        speed_acf_group_figures,
        "speed_acf_group",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack(speed_acf_group_figures)
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    behavior_utils,
    hM3Dq_mice,
    hM4Di_mice,
    plot_test,
    save_behavior_svg,
):
    speed_acf_auc_summary = behavior_utils.paired_speed_acf_auc_comparison(
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        fps=30,
        max_lag_sec=60,
        pair_map=behav_pair_map,
        bodypart="Center",
        speed_col="mean_speed",
    )

    speed_acf_auc_fig, speed_acf_auc_hm4_df, speed_acf_auc_hm3_df = (
        plot_test.plot_speed_acf_auc_groups(
            speed_acf_auc_summary,
            hM4Di_mice=hM4Di_mice,
            hM3Dq_mice=hM3Dq_mice,
        )
    )

    plot_test.save_figure_svg(
        speed_acf_auc_fig,
        "speed_acf_auc_groups",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )
    speed_acf_auc_fig
    return


@app.cell
def _():
    noeffec_group = ['NUO005', 'NUO008']
    effec_group = ['NUO010', 'NUO012']
    return


@app.cell
def _(behav_df_dic_dcz, behav_df_dic_saline, pd):
    behav_df_effec = pd.DataFrame([])
    behav_df_noeffec = pd.DataFrame([])
    for behav_df_name_ in behav_df_dic_dcz:
        behav_df_subject = behav_df_dic_dcz[behav_df_name_]
        behav_df_subject[("session", "")] = behav_df_name_
        if ("NUO005" in behav_df_name_) or ("NUO008" in behav_df_name_):
            behav_df_noeffec = pd.concat([behav_df_noeffec, behav_df_subject])
        else:
            behav_df_effec = pd.concat([behav_df_effec, behav_df_subject])
    for behav_df_name_ in behav_df_dic_saline:
        behav_df_subject = behav_df_dic_saline[behav_df_name_]
        behav_df_subject[("session", "")] = behav_df_name_
        behav_df_noeffec = pd.concat([behav_df_noeffec, behav_df_subject])
    return


@app.cell
def _(behav_df_dic_dcz, behav_df_dic_saline, np, pd):
    import ssm

    def iter_speed_hmm_entries(model_name):
        if model_name == "effect":
            for pair_id, behav_df in behav_df_dic_dcz.items():
                if ("NUO005" in pair_id) or ("NUO008" in pair_id):
                    continue
                yield pair_id, "DCZ", behav_df
        else:
            for pair_id, behav_df in behav_df_dic_dcz.items():
                if ("NUO005" in pair_id) or ("NUO008" in pair_id):
                    yield pair_id, "DCZ", behav_df
            for pair_id, behav_df in behav_df_dic_saline.items():
                yield pair_id, "saline", behav_df

    def build_speed_hmm_datas_from_entries(
        entries,
        speed_col=("Center", "mean_speed"),
        min_len=20,
    ):
        datas = []
        data_info = []

        for pair_id, condition, behav_df in entries:
            if behav_df.empty or speed_col not in behav_df.columns:
                continue

            speed = pd.to_numeric(behav_df[speed_col], errors="coerce")
            valid_speed = speed.replace([np.inf, -np.inf], np.nan).notna()
            valid_positions = np.flatnonzero(valid_speed.to_numpy())
            if len(valid_positions) < min_len:
                continue

            data = speed.iloc[valid_positions].to_numpy(dtype=float).reshape(-1, 1)
            datas.append(data)
            data_info.append(
                {
                    "pair_id": pair_id,
                    "condition": condition,
                    "df": behav_df,
                    "valid_positions": valid_positions,
                }
            )

        return datas, data_info

    def state_mean_speeds(hmm, datas):
        state_rows = []
        for state in range(hmm.K):
            weighted_sum = 0.0
            weighted_count = 0.0
            for data in datas:
                posterior = hmm.expected_states(data)[0]
                state_weight = posterior[:, state]
                weighted_sum += np.sum(state_weight * data[:, 0])
                weighted_count += np.sum(state_weight)

            state_rows.append(
                {
                    "state": state,
                    "mean_speed": weighted_sum / weighted_count
                    if weighted_count > 0
                    else np.nan,
                    "state_weight": weighted_count,
                }
            )

        return pd.DataFrame(state_rows)

    def fit_speed_hmm(datas, num_states=2, n_iters=100):
        hmm = ssm.HMM(num_states, 1, observations="gaussian")
        hmm_lls = hmm.fit(
            datas,
            method="em",
            num_iters=n_iters,
            tolerance=1e-4,
        )

        state_summary_before = state_mean_speeds(hmm, datas)
        perm = [
            int(state)
            for state in state_summary_before.sort_values("mean_speed")[
                "state"
            ].to_list()
        ]
        hmm.permute(perm)

        state_summary_after = state_mean_speeds(hmm, datas)
        log_likelihood = sum(float(hmm.log_probability(data)) for data in datas)

        return hmm, log_likelihood, hmm_lls, perm, state_summary_after

    def add_hmm_states_to_dfs(hmm, datas, data_info, model_name):
        state_rows = []
        for data, info in zip(datas, data_info):
            posterior = hmm.expected_states(data)[0]
            states = np.argmax(posterior, axis=1)
            behav_df = info["df"]

            behav_df[("speed_hmm_state", "")] = np.nan
            behav_df[("speed_hmm_model", "")] = None
            state_col_loc = behav_df.columns.get_loc(("speed_hmm_state", ""))
            model_col_loc = behav_df.columns.get_loc(("speed_hmm_model", ""))

            behav_df.iloc[info["valid_positions"], state_col_loc] = states
            behav_df.iloc[info["valid_positions"], model_col_loc] = model_name

            state_rows.append(
                {
                    "model": model_name,
                    "pair_id": info["pair_id"],
                    "condition": info["condition"],
                    "n_valid_frames": len(states),
                    "state0_frames": int(np.sum(states == 0)),
                    "state1_frames": int(np.sum(states == 1)),
                    "state1_fraction": float(np.mean(states == 1)),
                }
            )

        return pd.DataFrame(state_rows)

    hmm_speed_models = {}
    hmm_speed_rows = []
    hmm_speed_state_summaries = []

    for model_name in ["effect", "noeffect"]:
        entries = list(iter_speed_hmm_entries(model_name))
        datas, data_info = build_speed_hmm_datas_from_entries(entries)
        if not datas:
            hmm_speed_rows.append(
                {
                    "model": model_name,
                    "n_sequences": 0,
                    "n_frames": 0,
                    "log_likelihood": np.nan,
                    "state0_mean_speed": np.nan,
                    "state1_mean_speed": np.nan,
                    "state_permutation": None,
                    "status": "no valid speed data",
                }
            )
            continue

        hmm, ll, hmm_lls, perm, state_summary = fit_speed_hmm(datas)
        state_summary_by_df = add_hmm_states_to_dfs(
            hmm,
            datas,
            data_info,
            model_name,
        )
        hmm_speed_state_summaries.append(state_summary_by_df)
        hmm_speed_models[model_name] = {
            "model": hmm,
            "datas": datas,
            "data_info": data_info,
            "log_likelihood": ll,
            "hmm_lls": hmm_lls,
            "state_permutation": perm,
            "state_summary": state_summary,
            "state_summary_by_df": state_summary_by_df,
        }

        hmm_speed_rows.append(
            {
                "model": model_name,
                "n_sequences": len(datas),
                "n_frames": sum(len(data) for data in datas),
                "log_likelihood": ll,
                "state0_mean_speed": state_summary.loc[
                    state_summary["state"] == 0,
                    "mean_speed",
                ].iloc[0],
                "state1_mean_speed": state_summary.loc[
                    state_summary["state"] == 1,
                    "mean_speed",
                ].iloc[0],
                "state_permutation": perm,
                "status": "ok",
            }
        )

    hmm_effec = hmm_speed_models.get("effect", {}).get("model")
    hmm_noeffec = hmm_speed_models.get("noeffect", {}).get("model")
    hmm_speed_summary = pd.DataFrame(hmm_speed_rows)
    hmm_speed_state_summary = (
        pd.concat(hmm_speed_state_summaries, ignore_index=True)
        if hmm_speed_state_summaries
        else pd.DataFrame()
    )
    hmm_speed_summary
    return hmm_effec, hmm_noeffec


@app.cell
def _(
    behavior_svg_dir,
    hmm_effec,
    hmm_noeffec,
    plot_test,
    plt,
    save_behavior_svg,
):
    import utils_test

    hmm_transition_fig, hmm_transition_axes = plt.subplots(
        1,
        2,
        figsize=(7, 3),
    )

    if hmm_effec is not None:
        utils_test.plot_transition_matrix(
            hmm_effec,
            title="effect speed HMM",
            ax=hmm_transition_axes[0],
            cmap="gray",
        )
    else:
        hmm_transition_axes[0].set_title("effect speed HMM missing")
        hmm_transition_axes[0].axis("off")

    if hmm_noeffec is not None:
        utils_test.plot_transition_matrix(
            hmm_noeffec,
            title="no-effect speed HMM",
            ax=hmm_transition_axes[1],
            cmap="gray",
        )
    else:
        hmm_transition_axes[1].set_title("no-effect speed HMM missing")
        hmm_transition_axes[1].axis("off")

    hmm_transition_fig.tight_layout()
    plot_test.save_figure_svg(
        hmm_transition_fig,
        "speed_hmm_model_transition_matrix",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )
    hmm_transition_fig
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    mo,
    plot_test,
    save_behavior_svg,
):

    hmm_speed_transition_graph_fig_dic = (
        plot_test.plot_paired_speed_hmm_transition_graphs(
            pair_ids=behav_pair_map["pair_id"].unique(),
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            state_col=("speed_hmm_state", ""),
            K=2,
        )
    )

    plot_test.save_figures_svg(
        hmm_speed_transition_graph_fig_dic,
        "speed_hmm_pair_transition_graph",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack(list(hmm_speed_transition_graph_fig_dic.values()))
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behavior_svg_dir,
    hmm_effec,
    hmm_noeffec,
    mo,
    plot_test,
    save_behavior_svg,
):
    hmm_speed_posterior_figures = (
        plot_test.plot_speed_hmm_posteriors_for_behavior_dicts(
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            hmm_effec=hmm_effec,
            hmm_noeffec=hmm_noeffec,
            speed_col=("Center", "mean_speed"),
            speed_quantiles=(0.01, 0.99),
        )
    )

    plot_test.save_figures_svg(
        hmm_speed_posterior_figures,
        "speed_hmm_posterior",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack(hmm_speed_posterior_figures)
    return (hmm_speed_posterior_figures,)


@app.cell
def _(hmm_speed_posterior_figures, plt):
    posterior_fig_idx = 0
    frame_start = 1000
    frame_window = 100
    frame_end = frame_start + frame_window

    source_fig = hmm_speed_posterior_figures[posterior_fig_idx]
    source_ax = source_fig.axes[0]

    posterior_30_frame_fig, posterior_30_frame_ax = plt.subplots(
        1,
        1,
        figsize=(8, 2.5),
        dpi=120,
    )

    for line in source_ax.lines:
        x = line.get_xdata()
        y = line.get_ydata()

        mask = (x >= frame_start) & (x < frame_end)

        posterior_30_frame_ax.plot(
            x[mask],
            y[mask],
            color=line.get_color(),
            linewidth=line.get_linewidth(),
            alpha=line.get_alpha() if line.get_alpha() is not None else 1,
            label=line.get_label(),
        )

    posterior_30_frame_ax.set_ylim(source_ax.get_ylim())
    posterior_30_frame_ax.set_xlim(frame_start, frame_end - 1)
    posterior_30_frame_ax.set_yticks([0, 0.5, 1])
    posterior_30_frame_ax.set_xlabel("frame #")
    posterior_30_frame_ax.set_ylabel("p(state)")
    posterior_30_frame_ax.set_title(
        source_ax.get_title()
        + f" | frames {frame_start}-{frame_end - 1}"
    )
    posterior_30_frame_ax.legend(frameon=False, loc="upper right")
    posterior_30_frame_ax.grid(alpha=0.25)

    plt.tight_layout()
    posterior_30_frame_fig
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    mo,
    plot_test,
    save_behavior_svg,
    stationary_speed_threshold,
):
    state_speed_distribution_figures = plot_test.plot_paired_state_speed_distributions(
        pair_ids=behav_pair_map["pair_id"].unique(),
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        bodypart="Center",
        speed_col="mean_speed",
        state_col=("speed_hmm_state", ""),
        states=(0, 1),
        speed_threshold=stationary_speed_threshold,
        bins=60,
        kde=True,
    )

    plot_test.save_figures_svg(
        state_speed_distribution_figures,
        "state_speed_distribution",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack(state_speed_distribution_figures)
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    plot_test,
    save_behavior_svg,
):
    (
        speed_hmm_state_fraction_fig,
        speed_hmm_state_fraction_summary,
        noeffec_group_state_fraction_df,
        effec_group_state_fraction_df,
    ) = plot_test.plot_speed_hmm_state_fraction_comparison(
        pair_ids=behav_pair_map["pair_id"].unique(),
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        state_col=("speed_hmm_state", ""),
        K=2,
        states=(0, 1),
        figsize=(8, 7),
    )

    plot_test.save_figure_svg(
        speed_hmm_state_fraction_fig,
        "speed_hmm_state_fraction_groups",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
        subfolder="speed_hmm_state_fraction_groups",
    )

    speed_hmm_state_fraction_fig
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    mo,
    plot_test,
    save_behavior_svg,
):
    hmm_speed_transition_fig_dic = (
        plot_test.plot_paired_speed_hmm_transition_matrices(
            pair_ids=behav_pair_map["pair_id"].unique(),
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            state_col=("speed_hmm_state", ""),
            K=2,
        )
    )

    plot_test.save_figures_svg(
        hmm_speed_transition_fig_dic,
        "speed_hmm_pair_transition_matrix",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack(list(hmm_speed_transition_fig_dic.values()))
    return (hmm_speed_transition_fig_dic,)


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    hmm_speed_transition_fig_dic,
    plot_test,
    save_behavior_svg,
):
    (
        speed_hmm_switch_probability_fig,
        speed_hmm_switch_probability_summary,
        noeffec_group_switch_df,
        effec_group_switch_df,
    ) = plot_test.plot_speed_hmm_switch_probability_comparison(
        pair_ids=behav_pair_map["pair_id"].unique(),
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        transition_fig_dic=hmm_speed_transition_fig_dic,
        K=2,
        figsize=(8, 7),
    )

    plot_test.save_figure_svg(
        speed_hmm_switch_probability_fig,
        "speed_hmm_switch_probability_groups",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
        subfolder="speed_hmm_switch_probability_groups",
    )

    speed_hmm_switch_probability_fig
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    plot_test,
    save_behavior_svg,
):
    speed_hmm_switch_rate_summary_df = plot_test.speed_hmm_switch_rate_summary(
        pair_ids=behav_pair_map["pair_id"].unique(),
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        fps=30,
        state_col=("speed_hmm_state", ""),
        K=2,
    )

    (
        speed_hmm_switch_rate_distribution_fig,
        _speed_hmm_switch_rate_distribution_df,
    ) = plot_test.plot_speed_hmm_switch_rate_distribution(
        speed_hmm_switch_rate_summary_df,
        states=(0, 1),
    )

    plot_test.save_figure_svg(
        speed_hmm_switch_rate_distribution_fig,
        "speed_hmm_switch_rate_by_session",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
        subfolder="speed_hmm_switch_rate_by_session",
    )

    speed_hmm_switch_rate_distribution_fig
    return (speed_hmm_switch_rate_summary_df,)


@app.cell
def _(
    behavior_svg_dir,
    plot_test,
    save_behavior_svg,
    speed_hmm_switch_rate_summary_df,
):
    (
        speed_hmm_state_switch_rate_fig,
        _speed_hmm_state_switch_rate_plot_df,
    ) = plot_test.plot_speed_hmm_state_switch_rate_groups(
        speed_hmm_switch_rate_summary_df,
        states=(0, 1),
    )

    plot_test.save_figure_svg(
        speed_hmm_state_switch_rate_fig,
        "speed_hmm_state_switch_rate_groups",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
        subfolder="speed_hmm_state_switch_rate",
    )
    speed_hmm_state_switch_rate_fig
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    mo,
    plot_test,
    save_behavior_svg,
):
    hmm_state_xy_heatmap_fig_dic = (
        plot_test.plot_paired_speed_hmm_state_xy_heatmaps(
            pair_ids=behav_pair_map["pair_id"].unique(),
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            bodypart="Center",
            state_col=("speed_hmm_state", ""),
            states=(0, 1),
            xbins=40,
            ybins=40,
            extent=(0, 640, 0, 480),
            normalize=True,
            difference=True,
        )
    )

    plot_test.save_figures_svg(
        hmm_state_xy_heatmap_fig_dic,
        "speed_hmm_state_xy_heatmap",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack(list(hmm_state_xy_heatmap_fig_dic.values()))
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    mo,
    plot_test,
    save_behavior_svg,
):
    hmm_state_occupancy_diff_fig_dic = (
        plot_test.plot_paired_speed_hmm_state_occupancy_diff_heatmaps(
            pair_ids=behav_pair_map["pair_id"].unique(),
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            bodypart="Center",
            state_col=("speed_hmm_state", ""),
            timestamp_col=("timestamp", ""),
            states=(0, 1),
            xbins=40,
            ybins=40,
            extent=(0, 640, 0, 480),
            normalize=True,
        )
    )

    plot_test.save_figures_svg(
        hmm_state_occupancy_diff_fig_dic,
        "speed_hmm_state_occupancy_diff_heatmap",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack(list(hmm_state_occupancy_diff_fig_dic.values()))
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    mo,
    plot_test,
    save_behavior_svg,
):

    condition_state_occupancy_diff_fig_dic = (
        plot_test.plot_paired_speed_hmm_condition_state_occupancy_diff_heatmaps(
            pair_ids=behav_pair_map["pair_id"].unique(),
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            bodypart="Center",
            state_col=("speed_hmm_state", ""),
            timestamp_col=("timestamp", ""),
            state_a=1,
            state_b=0,
            xbins=40,
            ybins=40,
            extent=(0, 640, 0, 480),
            normalize=True,
        )
    )

    plot_test.save_figures_svg(
        condition_state_occupancy_diff_fig_dic,
        "speed_hmm_condition_state_occupancy_diff_heatmap",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack(list(condition_state_occupancy_diff_fig_dic.values()))
    return


if __name__ == "__main__":
    app.run()
