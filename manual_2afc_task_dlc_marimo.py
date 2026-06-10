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
    import utils_test

    from lecilab_behavior_analysis import utils as utils
    from lecilab_behavior_analysis import df_transforms as dft
    from lecilab_behavior_analysis import plots
    from lecilab_behavior_analysis.figure_maker import (
        session_summary_figure,
        subject_progress_figure,
    )
    from scipy.optimize import minimize

    warnings.filterwarnings("ignore")
    return Path, behavior_utils, dft, np, pd, plot_test, plt, utils, utils_test


@app.cell
def _(behavior_utils, plot_test, utils_test):
    import importlib
    importlib.reload(behavior_utils)
    importlib.reload(plot_test)
    importlib.reload(utils_test)
    return


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

    return (safe_rsync_cluster_data,)


@app.cell
def _(utils):
    tv_projects = utils.get_server_projects()
    print(tv_projects)
    return (tv_projects,)


@app.cell
def _(mo, tv_projects):
    _default_project = 'COT_cannula_data'
    project_select = mo.ui.dropdown(
        options=tv_projects,
        value=_default_project,
        label="Project",
    )

    project_select
    return (project_select,)


@app.cell
def _(project_select, tv_projects):
    project = project_select.value
    project_idx = tv_projects.index(project)
    return (project,)


@app.cell
def _(project, utils):
    animals = utils.get_animals_in_project(project)
    print(animals)
    return (animals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # mouse select
    """)
    return


@app.cell
def _(animals, mo):
    mouse_select = mo.ui.dropdown(
        options=animals,
        value=animals[0],
        label="Mouse",
    )

    mouse_select
    return (mouse_select,)


@app.cell
def _(mo):
    download_button = mo.ui.run_button(label="Download / update this mouse")
    download_button
    return (download_button,)


@app.cell
def _(
    Path,
    animals,
    download_button,
    mouse_select,
    pd,
    project,
    safe_rsync_cluster_data,
    utils,
):
    mouse = mouse_select.value

    local_path = Path(utils.get_outpath()) / project / "sessions" / mouse
    local_path.mkdir(parents=True, exist_ok=True)

    if download_button.value:
        for _mouse in animals:
            _local_path = Path(utils.get_outpath()) / project / "sessions" / _mouse
            _local_path.mkdir(parents=True, exist_ok=True)

            safe_rsync_cluster_data(
                project_name=project,
                file_path="sessions/{}/{}.csv".format(_mouse, _mouse),
                local_path=str(_local_path),
                credentials=utils.get_idibaps_cluster_credentials(),
            )


    df_all = pd.read_csv(local_path / Path(f"{mouse}.csv"), sep=";")
    return df_all, mouse


@app.cell
def _(mo):
    run_mode_select = mo.ui.dropdown(
        options=["Auto", "Manual"],
        value="Manual",
        label="Run mode",
    )

    run_mode_select
    return (run_mode_select,)


@app.cell
def _(df_all, run_mode_select):
    run_mode = run_mode_select.value
    df = df_all[df_all["run_mode"] == run_mode]
    df.iloc[-10:]
    return (df,)


@app.cell
def _():
    # fig_subject_progress = subject_progress_figure(
    #     df,
    #     perf_window=100,
    #     summary_matrix_plot=False,
    # )
    # fig_subject_progress
    return


@app.cell
def _(df, dft, np):
    df_day = dft.add_day_column_to_df(df.copy())
    df_day = dft.add_trial_of_day_column_to_df(df_day)
    df_day["total_trial"] = np.arange(1, df_day.shape[0] + 1)

    available_dates = sorted(
        df_day["year_month_day"].dropna().astype(str).unique().tolist()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # manual mouse select
    """)
    return


@app.cell
def _(animals, mo):
    fit_mouse_select = mo.ui.multiselect(
        options=animals,
        value=animals[:-1],
        label="Mice for fit",
    )

    fit_mouse_select
    return (fit_mouse_select,)


@app.cell
def _(mo):
    update_processData_button = mo.ui.run_button(label="update selected mice processed data")
    update_processData_button
    return (update_processData_button,)


@app.cell
def _():
    # hM3Dq_mice = ['NUO062', 'NUO063', 'NUO064', 'NUO065', 'NUO007', 'NUO008', 'NUO009', 'NUO010', 'NUO011', 'NUO012']
    # hM4Di_mice = ['NUO057', 'NUO058', 'NUO059', 'NUO060', 'NUO061', 'NUO001', 'NUO002', 'NUO003', 'NUO005', 'NUO006']
    hM3Dq_mice = ['NUO064', 'NUO065', 'NUO007', 'NUO008', 'NUO009', 'NUO010', 'NUO011', 'NUO012']
    hM4Di_mice = ['NUO060', 'NUO061', 'NUO001', 'NUO002', 'NUO003', 'NUO005', 'NUO006']
    return hM3Dq_mice, hM4Di_mice


@app.cell
def _(
    Path,
    dft,
    fit_mouse_select,
    mouse,
    pd,
    project,
    safe_rsync_cluster_data,
    update_processData_button,
    utils,
):
    parameters_output_dir = (
        Path(utils.get_outpath()) / project / "processed" / "parameters_for_fit" / "manual"
    )
    stim_columns = {
        "df_test_vis": "visual",
        "df_test_aud": "auditory",
    }

    selected_mice = list(fit_mouse_select.value)
    if not selected_mice:
        selected_mice = [mouse]

    def load_or_compute_mouse_parameters(_mouse):
        parameters_output_path = parameters_output_dir / f"{_mouse}_parameters_for_fit.pkl"
        if parameters_output_path.exists() and not update_processData_button.value:
            print(f"Loaded saved parameters_for_fit data: {parameters_output_path}")
            return pd.read_pickle(parameters_output_path)

        _local_path = Path(utils.get_outpath()) / project / "sessions" / _mouse
        _local_path.mkdir(parents=True, exist_ok=True)
        _csv_path = _local_path / f"{_mouse}.csv"
        if not _csv_path.exists():
            safe_rsync_cluster_data(
                project_name=project,
                file_path="sessions/{}/{}.csv".format(_mouse, _mouse),
                local_path=str(_local_path),
                credentials=utils.get_idibaps_cluster_credentials(),
            )

        if not _csv_path.exists():
            print(f"No CSV found for {_mouse}; skipped parameters_for_fit.")
            return {output_name: pd.DataFrame() for output_name in stim_columns}

        df_all_mouse = pd.read_csv(_csv_path, sep=";")
        df_manual = df_all_mouse[df_all_mouse["run_mode"] == "Manual"]
        computed_results = {}
        for output_name, stim_name in stim_columns.items():
            df_stim = df_manual[df_manual["stimulus_modality"] == stim_name]
            if df_stim.empty:
                computed_results[output_name] = df_stim.copy()
                print(f"No {stim_name} trials found for {_mouse}; skipped parameters_for_fit.")
            else:
                computed_results[output_name] = dft.parameters_for_fit(df_stim)

        parameters_output_dir.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(computed_results, parameters_output_path)
        print(f"Saved parameters_for_fit data: {parameters_output_path}")
        return computed_results

    combined_results = {output_name: [] for output_name in stim_columns}
    for _mouse in selected_mice:
        mouse_results = load_or_compute_mouse_parameters(_mouse)
        for output_name in stim_columns:
            df_mouse_result = mouse_results.get(output_name, pd.DataFrame())
            if not df_mouse_result.empty:
                if "subject" not in df_mouse_result.columns:
                    df_mouse_result = df_mouse_result.copy()
                    df_mouse_result["subject"] = _mouse
                combined_results[output_name].append(df_mouse_result)

    df_test_raw_parts = [
        df_result
        for output_name in stim_columns
        for df_result in combined_results[output_name]
    ]
    df_test_raw = (
        pd.concat(df_test_raw_parts, ignore_index=True)
        if df_test_raw_parts
        else pd.DataFrame()
    )
    return (df_test_raw,)


@app.cell
def _():
    # df_test["observations"] = pd.NA

    # for session in df_test["session"].dropna().unique():
    #     df_session = df_test[df_test["session"] == session]
    #     date_values = df_session["date"].dropna().unique()
    #     if len(date_values) == 0:
    #         print(f"No date found for session {session}; skipped observation.")
    #         continue

    #     date_ = pd.to_datetime(date_values[0]).strftime("%Y%m%d_%H%M%S")
    #     session_info_file_name = mouse + "_TwoAFC_" + date_ + ".json"
    #     session_info_path = local_path / session_info_file_name
    #     safe_rsync_cluster_data(
    #         project_name=project,
    #         file_path="sessions/{}/{}".format(mouse, session_info_file_name),
    #         local_path=str(local_path),
    #         credentials=utils.get_idibaps_cluster_credentials(),
    #     )

    #     if not session_info_path.exists():
    #         print(f"No session info JSON found for session {session}: {session_info_path}")
    #         continue

    #     with session_info_path.open("r") as session_info_file:
    #         session_info = json.load(session_info_file)
    #     if isinstance(session_info, str):
    #         session_info = json.loads(session_info)

    #     observation = session_info.get("observations", pd.NA)
    #     df_test.loc[df_session.index, "observations"] = [
    #         observation
    #     ] * len(df_session)
    return


@app.cell
def _(pd):
    injection_info_file_path = "/mnt/e/data/LeciLab/behavioral_data/data_test/manual_test_record_nuo.xlsx"
    injection_info_df = pd.read_excel(injection_info_file_path)
    injection_info_df
    return (injection_info_df,)


@app.cell
def _(df_test_raw, injection_info_df, mouse, pd):
    df_test = df_test_raw.copy()
    df_test["observations"] = pd.NA

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

    trial_dates = pd.to_datetime(
        df_test["date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    trial_mice = (
        df_test["subject"].astype(str)
        if "subject" in df_test.columns
        else pd.Series(mouse, index=df_test.index)
    )
    df_test["observations"] = [
        injection_lookup.get((trial_date, trial_mouse), pd.NA)
        for trial_date, trial_mouse in zip(trial_dates, trial_mice)
    ]

    df_test['observations']
    return (df_test,)


@app.cell
def _(pd, utils):
    def add_number_of_pokes(df_raw: pd.DataFrame, port_number: int):
        df = df_raw.copy()
        port_hold_column = f"port{port_number}_holds"
        df[port_hold_column] = df.apply(lambda row: utils.get_trial_port_hold(row, port_number), axis=1)
        df[f"port{port_number}_pokes_num"] = df[port_hold_column].apply(lambda x: len(x))
        return df

    return (add_number_of_pokes,)


@app.cell
def _(add_number_of_pokes, df_test, dft, hM3Dq_mice, hM4Di_mice, pd):
    session_performance = df_test.groupby(
        ["subject", "session"]
    )["correct"].transform("mean")

    is_saline = df_test["observations"].str.contains(
        "saline",
        case=False,
        na=False,
    )

    drop_mask = is_saline & (session_performance < 0.7)

    df_test_upd = df_test[
        ~drop_mask
    ].copy()


    def add_trial_variables_by_subject(df):
        if df.empty:
            return df.copy()

        subject_dfs = []
        for _, df_subject in df.groupby("subject", sort=False):
            df_subject = dft.add_trial_duration_column_to_df(df_subject)
            df_subject = dft.add_engagement_column(
                df_subject,
                engagement_sd_criteria=2,
            )
            df_subject = dft.calculate_time_between_trials_and_reaction_time(dft.add_day_column_to_df(
                add_number_of_pokes(df_subject, port_number=2)
            ))
            subject_dfs.append(df_subject)
        return pd.concat(subject_dfs).sort_index()

    df_test_upd = add_trial_variables_by_subject(df_test_upd)
    # df_test_upd = df_test_upd[
    #     df_test_upd['year_month_day'].isin(
    #         ['2026-05-22', '2026-05-20']
    #     )
    # ]

    df_test_hm4 = df_test_upd[
        df_test_upd["subject"].isin(hM4Di_mice)
    ].copy()

    df_test_hm3 = df_test_upd[
        df_test_upd["subject"].isin(hM3Dq_mice)
    ].copy()
    return df_test_hm3, df_test_hm4, df_test_upd


@app.cell
def _(df_test_hm4):
    df_test_hm4['year_month_day']
    return


@app.cell
def _():
    noeffec_group = ['NUO005', 'NUO008']
    effec_group = ['NUO010', 'NUO012']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # import dlc data
    """)
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
    dlc_mice_selected = ['NUO001', 'NUO002', 'NUO003', 'NUO005', 'NUO007', 'NUO008', 'NUO009', 'NUO010', 'NUO012']
    return (dlc_mice_selected,)


@app.cell
def _(mo):
    load_paired_dlc_button = mo.ui.run_button(label="Load paired DLC data")
    load_paired_dlc_button
    return


@app.cell
def _(behavior_utils, dlc_mice_selected, injection_info_df):
    paired_dlc_dates = behavior_utils.get_paired_injection_dates(
        injection_info_df,
        mice_selected=dlc_mice_selected,
    )
    return (paired_dlc_dates,)


@app.cell
def _():
    # read names of all the behavior data
    import shlex
    import subprocess
    IDIBAPS_TV_PROJECTS = "/storage/training_village/"
    def list_cluster_files(project_name, remote_folder, credentials):
        remote_dir = f"{IDIBAPS_TV_PROJECTS}{project_name}/{remote_folder}".rstrip("/")
        ssh_target = f"{credentials['username']}@{credentials['host']}"

        command = f"ls -1 {shlex.quote(remote_dir)}"

        result = subprocess.run(
            ["ssh", ssh_target, command],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(result.stderr)
            return []

        return [name for name in result.stdout.splitlines() if name]

    return list_cluster_files, subprocess


@app.cell
def _(dlc_mice_selected, list_cluster_files, project, utils):
    all_session_files = []
    for dlc_mouse_selected_ in dlc_mice_selected:
        session_file = list_cluster_files(
            project_name=project,
            remote_folder="sessions/{}".format(dlc_mouse_selected_),
            credentials=utils.get_idibaps_cluster_credentials(),
        )
        all_session_files.extend(session_file)
    return (all_session_files,)


@app.cell
def _(all_session_files, paired_dlc_dates, pd):
    matched_session_files = []

    for _, row_ in paired_dlc_dates.iterrows():
        subject_ = str(row_["subject"])
        saline_date_ = pd.to_datetime(row_["saline_date"]).strftime("%Y%m%d")
        dcz_date_ = pd.to_datetime(row_["DCZ_date"]).strftime("%Y%m%d")

        saline_files_ = [
            file_name[:-4]
            for file_name in all_session_files
            if subject_ in file_name
            and saline_date_ in file_name
            and file_name.endswith(".csv")
            and not file_name.endswith("_RAW.csv")
        ]

        dcz_files_ = [
            file_name[:-4]
            for file_name in all_session_files
            if subject_ in file_name
            and dcz_date_ in file_name
            and file_name.endswith(".csv")
            and not file_name.endswith("_RAW.csv")
        ]

        matched_session_files.extend(saline_files_)
        matched_session_files.extend(dcz_files_)
    return (matched_session_files,)


@app.cell
def _(
    Path,
    behavior_utils,
    dlc_mice_selected,
    download_dlcData_button,
    matched_session_files,
    paired_dlc_dates,
    pd,
    project,
    subprocess,
    utils,
    utils_test,
):
    credentials = utils.get_idibaps_cluster_credentials()

    behav_df_dic = {}
    video_df_dic = {}

    matched_session_stems = set(matched_session_files)

    def is_matched_dlc_file(file_path):
        file_name = Path(file_path).name
        return any(session_stem in file_name for session_stem in matched_session_stems)

    def remote_glob_exists(remote_pattern, credentials):
        ssh_target = f"{credentials['username']}@{credentials['host']}"
        command = f"ls -1 {remote_pattern}"

        result = subprocess.run(
            ["ssh", ssh_target, command],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and bool(result.stdout.strip())

    dlc_data_path = "/home/kudongdong/data/behavior_DLC/TrainingVillage_2AFC_superanimal-LeciLab-2025-10-16/v2"

    if download_dlcData_button.value:
        for dlc_mouse_selected in dlc_mice_selected:
            _local_path = Path(utils.get_outpath()) / project / "sessions" / dlc_mouse_selected / "DLC"
            _local_path.mkdir(parents=True, exist_ok=True)

            mouse_session_stems = [
                session_stem
                for session_stem in matched_session_stems
                if session_stem.startswith(dlc_mouse_selected)
            ]

            for session_stem in mouse_session_stems:
                remote_pattern = f"{dlc_data_path}/{session_stem}*.csv"

                if not remote_glob_exists(remote_pattern, credentials):
                    print(f"Skip missing DLC: {remote_pattern}")
                    continue

                utils_test.rsync_specific_file(
                    file_path=remote_pattern,
                    local_path=str(_local_path),
                    credentials=credentials,
                )

    for dlc_mouse_selected in dlc_mice_selected:
        _local_path = Path(utils.get_outpath()) / project / "sessions" / dlc_mouse_selected / "DLC"

        for dlc_file_ in sorted(_local_path.glob("*DLC*.csv")):
            if not is_matched_dlc_file(dlc_file_):
                continue

            random_df = pd.read_csv(dlc_file_, header=[1, 2])
            behav_df_dic[dlc_file_.name[:29]] = random_df


    video_data_path = f"/storage/training_village/{project}/videos"

    if download_dlcData_button.value:
        for dlc_mouse_selected in dlc_mice_selected:
            _local_path = Path(utils.get_outpath()) / project / "sessions" / dlc_mouse_selected / "DLC"

            for dlc_file_ in sorted(_local_path.glob("*DLC*.csv")):
                if not is_matched_dlc_file(dlc_file_):
                    continue

                session_key = dlc_file_.name[:29]
                utils_test.rsync_specific_file(
                    file_path=Path(video_data_path, dlc_mouse_selected, session_key + ".csv"),
                    local_path=str(_local_path),
                    credentials=utils.get_idibaps_cluster_credentials(),
                )

    video_df_dic = {}
    missing_video_files = []

    if download_dlcData_button.value:
        for video_file_ in behav_df_dic:
            _local_path = Path(utils.get_outpath()) / project / "sessions" / video_file_[:6] / "DLC"
            video_csv_path = Path(_local_path, video_file_ + ".csv")

            if not video_csv_path.exists():
                print(f"Skip missing local video csv: {video_csv_path}")
                missing_video_files.append(video_file_)
                continue

            video_df = pd.read_csv(
                video_csv_path,
                sep=";",
                index_col="frame",
            )
            video_df = video_df[2:]
            video_df.index = range(0, len(video_df))
            video_df_dic[video_file_] = video_df

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
    return (
        behav_df_dic_dcz,
        behav_df_dic_saline,
        behav_pair_map,
        video_df_dic_dcz,
        video_df_dic_saline,
    )


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_utils,
    df_test_upd,
    np,
    pd,
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

    df_test_upd_pairing = df_test_upd.copy()
    df_test_upd_pairing["_pair_subject"] = df_test_upd_pairing["subject"].astype(str)
    df_test_upd_pairing["_pair_date"] = pd.to_datetime(
        df_test_upd_pairing["year_month_day"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    df_dic_dcz = {}
    df_dic_saline = {}

    for pair_id in behav_pair_map["pair_id"].unique():
        pair_row = behav_pair_map[behav_pair_map["pair_id"] == pair_id].iloc[0]
        subject = str(pair_row["subject"])
        saline_date = pd.to_datetime(pair_row["saline_date"]).strftime("%Y-%m-%d")
        dcz_date = pd.to_datetime(pair_row["DCZ_date"]).strftime("%Y-%m-%d")

        df_dic_dcz[pair_id] = df_test_upd_pairing[
            (df_test_upd_pairing["_pair_subject"] == subject)
            & (df_test_upd_pairing["_pair_date"] == dcz_date)
        ].drop(columns=["_pair_subject", "_pair_date"]).copy()
        df_dic_saline[pair_id] = df_test_upd_pairing[
            (df_test_upd_pairing["_pair_subject"] == subject)
            & (df_test_upd_pairing["_pair_date"] == saline_date)
        ].drop(columns=["_pair_subject", "_pair_date"]).copy()

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
    return df_dic_dcz, df_dic_saline


@app.cell
def _():
    roi_bottom = 80
    roi_top = 190
    roi_left = 235
    roi_right = 400
    return roi_bottom, roi_left, roi_right, roi_top


@app.cell
def _(Path):
    save_behavior_svg = True
    behavior_svg_dir = Path("/mnt/e/data/LeciLab/behavioral_data/tmp")
    return behavior_svg_dir, save_behavior_svg


@app.cell
def _(behavior_utils, mo, np, pd, plot_test, plt):
    from scipy import stats

    def analyze_paired_behavior_by_trial_column(
        split_column,
        df_dic_saline,
        df_dic_dcz,
        behav_df_dic_saline,
        behav_df_dic_dcz,
        behav_pair_map,
        roi_left,
        roi_right,
        roi_bottom,
        roi_top,
        hM4Di_mice,
        hM3Dq_mice,
        behavior_svg_dir,
        save_behavior_svg,
        bodypart="Center",
        true_label=None,
        false_label=None,
        include_stationary=False,
        include_occupancy=True,
        include_trajectory_speed=True,
        include_roi_time=True,
        include_stationary_speed=True,
        include_stationary_time=True,
        stationary_speed_threshold=10,
        speed_col="mean_speed",
    ):
        default_labels = {
            "engaged": ("engaged", "disengaged"),
            "correct": ("correct", "incorrect"),
            "previous_correct": ("previous correct", "previous incorrect"),
        }
        default_true_label, default_false_label = default_labels.get(
            split_column,
            (f"{split_column}_true", f"{split_column}_false"),
        )
        true_label = true_label or default_true_label
        false_label = false_label or default_false_label

        def split_behavior_by_trial_column(behav_df, trial_df):
            empty_behav_df = behav_df.iloc[0:0].copy()
            if behav_df.empty or trial_df.empty or split_column not in trial_df.columns:
                return empty_behav_df, empty_behav_df.copy()

            timestamp = pd.to_numeric(
                behavior_utils.get_behavior_column(behav_df, ("timestamp", "")),
                errors="coerce",
            )
            true_mask = pd.Series(False, index=behav_df.index)
            false_mask = pd.Series(False, index=behav_df.index)

            for _, trial_row in trial_df.iterrows():
                split_value = trial_row[split_column]
                if pd.isna(split_value):
                    continue

                try:
                    trial_start = float(trial_row["TRIAL_START"])
                    trial_end = float(trial_row["TRIAL_END"])
                except (TypeError, ValueError):
                    continue

                if not np.isfinite(trial_start) or not np.isfinite(trial_end):
                    continue

                trial_mask = timestamp.between(
                    trial_start,
                    trial_end,
                    inclusive="both",
                ).fillna(False)
                if split_value:
                    true_mask = true_mask | trial_mask
                else:
                    false_mask = false_mask | trial_mask

            return behav_df.loc[true_mask].copy(), behav_df.loc[false_mask].copy()

        def trial_df_for_split_value(trial_df, target_value):
            if trial_df.empty or split_column not in trial_df.columns:
                return trial_df.iloc[0:0].copy()

            split_mask = []
            for split_value in trial_df[split_column]:
                if pd.isna(split_value):
                    split_mask.append(False)
                else:
                    split_mask.append(bool(split_value) is target_value)
            return trial_df.loc[split_mask].copy()

        pair_ids = list(behav_pair_map["pair_id"].unique())
        behav_df_dic_dcz_true = {}
        behav_df_dic_dcz_false = {}
        behav_df_dic_saline_true = {}
        behav_df_dic_saline_false = {}

        for pair_id in pair_ids:
            if pair_id not in df_dic_dcz or pair_id not in df_dic_saline:
                continue

            (
                behav_df_dic_dcz_true[pair_id],
                behav_df_dic_dcz_false[pair_id],
            ) = split_behavior_by_trial_column(
                behav_df_dic_dcz[pair_id],
                df_dic_dcz[pair_id],
            )
            (
                behav_df_dic_saline_true[pair_id],
                behav_df_dic_saline_false[pair_id],
            ) = split_behavior_by_trial_column(
                behav_df_dic_saline[pair_id],
                df_dic_saline[pair_id],
            )

        def has_plot_data(behav_df):
            if behav_df.empty:
                return False
            if (bodypart, "x") not in behav_df.columns or (bodypart, "y") not in behav_df.columns:
                return False
            x = pd.to_numeric(behav_df[(bodypart, "x")], errors="coerce")
            y = pd.to_numeric(behav_df[(bodypart, "y")], errors="coerce")
            return x.notna().any() and y.notna().any()

        def pair_ids_with_data(behav_df_dic_saline_split, behav_df_dic_dcz_split):
            return [
                pair_id
                for pair_id in pair_ids
                if pair_id in behav_df_dic_saline_split
                and pair_id in behav_df_dic_dcz_split
                and has_plot_data(behav_df_dic_saline_split[pair_id])
                and has_plot_data(behav_df_dic_dcz_split[pair_id])
            ]

        true_pair_ids = pair_ids_with_data(
            behav_df_dic_saline_true,
            behav_df_dic_dcz_true,
        )
        false_pair_ids = pair_ids_with_data(
            behav_df_dic_saline_false,
            behav_df_dic_dcz_false,
        )

        def split_conditions_for_pair(pair_id):
            return [
                (f"{true_label} saline", behav_df_dic_saline_true[pair_id]),
                (f"{true_label} DCZ", behav_df_dic_dcz_true[pair_id]),
                (f"{false_label} saline", behav_df_dic_saline_false[pair_id]),
                (f"{false_label} DCZ", behav_df_dic_dcz_false[pair_id]),
            ]

        four_condition_pair_ids = [
            pair_id
            for pair_id in pair_ids
            if pair_id in behav_df_dic_saline_true
            and pair_id in behav_df_dic_dcz_true
            and pair_id in behav_df_dic_saline_false
            and pair_id in behav_df_dic_dcz_false
            and any(has_plot_data(behav_df) for _, behav_df in split_conditions_for_pair(pair_id))
        ]

        def occupancy_for_df(behav_df):
            if (
                behav_df.empty
                or (bodypart, "x") not in behav_df.columns
                or (bodypart, "y") not in behav_df.columns
                or ("timestamp", "") not in behav_df.columns
            ):
                return np.zeros((15, 15))
            return behavior_utils.occupancy_map(
                pd.to_numeric(behav_df[(bodypart, "x")], errors="coerce"),
                pd.to_numeric(behav_df[(bodypart, "y")], errors="coerce"),
                pd.to_numeric(behav_df[("timestamp", "")], errors="coerce"),
                xbins=15,
                ybins=15,
            )

        def plot_four_condition_occupancy(pair_id):
            condition_dfs = split_conditions_for_pair(pair_id)
            occ_maps = [occupancy_for_df(behav_df) for _, behav_df in condition_dfs]
            occ_values = np.concatenate([occ.ravel() for occ in occ_maps])
            occ_norm = plot_test._normalize_from_values(occ_values)

            x_values = []
            y_values = []
            for _, behav_df in condition_dfs:
                if (bodypart, "x") in behav_df.columns and (bodypart, "y") in behav_df.columns:
                    x_values.append(pd.to_numeric(behav_df[(bodypart, "x")], errors="coerce"))
                    y_values.append(pd.to_numeric(behav_df[(bodypart, "y")], errors="coerce"))

            if x_values and y_values:
                x_all = pd.concat(x_values, ignore_index=True).dropna()
                y_all = pd.concat(y_values, ignore_index=True).dropna()
            else:
                x_all = pd.Series(dtype=float)
                y_all = pd.Series(dtype=float)

            if x_all.empty or y_all.empty:
                extents = (0, 640, 0, 480)
            else:
                extents = (
                    float(x_all.min()),
                    float(x_all.max()),
                    float(y_all.min()),
                    float(y_all.max()),
                )

            fig, axes = plt.subplots(
                1,
                4,
                figsize=(18, 4.5),
                sharex=True,
                sharey=True,
            )
            last_im = None
            for ax, (condition_label, behav_df), occ in zip(axes, condition_dfs, occ_maps):
                last_im = ax.imshow(
                    occ.T,
                    origin="lower",
                    extent=extents,
                    cmap="viridis",
                    norm=occ_norm,
                    interpolation="gaussian",
                )
                if behav_df.empty:
                    ax.text(
                        0.5,
                        0.5,
                        "No frames",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        color="white",
                    )
                roi_rect = plt.Rectangle(
                    (roi_left, roi_bottom),
                    roi_right - roi_left,
                    roi_top - roi_bottom,
                    linewidth=1.5,
                    edgecolor="white",
                    facecolor="none",
                    linestyle="--",
                )
                ax.add_patch(roi_rect)
                ax.set_title(condition_label, fontsize=10)
                ax.set_aspect("equal")
                ax.set_xlim(0, 640)
                ax.set_ylim(0, 480)
                ax.set_xticks([])
                ax.set_yticks([])

            axes[0].invert_yaxis()
            fig.suptitle(f"{pair_id} occupancy by {split_column}", fontsize=12)
            if last_im is not None:
                fig.colorbar(
                    last_im,
                    ax=axes,
                    label="occupancy (s/pixels)",
                    shrink=0.7,
                    fraction=0.025,
                    pad=0.02,
                    extend="both",
                )
            return fig

        def plot_four_condition_traj_speed(pair_id):
            condition_dfs = split_conditions_for_pair(pair_id)
            traj_dfs = []
            for _, behav_df in condition_dfs:
                if behav_df.empty or bodypart not in behav_df.columns.get_level_values(0):
                    traj_dfs.append(pd.DataFrame(columns=["x", "y", "mean_speed"]))
                else:
                    traj_dfs.append(plot_test.prep_traj_speed_df(behav_df, bodypart=bodypart))

            speed_values = pd.concat(
                [traj_df["mean_speed"] for traj_df in traj_dfs],
                ignore_index=True,
            ).dropna()
            speed_norm = plot_test._normalize_from_values(speed_values)

            fig, axes = plt.subplots(
                1,
                4,
                figsize=(18, 4.5),
                sharex=True,
                sharey=True,
            )
            for ax, (condition_label, _), traj_df in zip(axes, condition_dfs, traj_dfs):
                if len(traj_df) >= 2:
                    plot_test.plot_traj_speed(
                        traj_df,
                        cmap="inferno",
                        ax=ax,
                        norm=speed_norm,
                    )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No trajectory",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                ax.set_title(condition_label, fontsize=10)
                ax.set_aspect("equal")
                ax.set_xlim(0, 640)
                ax.set_ylim(0, 480)
                ax.set_xticks([])
                ax.set_yticks([])

            axes[0].invert_yaxis()
            fig.suptitle(f"{pair_id} trajectory speed by {split_column}", fontsize=12)
            speed_sm = plt.cm.ScalarMappable(
                norm=speed_norm,
                cmap="inferno",
            )
            fig.colorbar(
                speed_sm,
                ax=axes,
                label="mean speed (pixels/s)",
                shrink=0.7,
                fraction=0.025,
                pad=0.02,
            )
            return fig

        four_condition_pair_figures = []
        if include_occupancy or include_trajectory_speed:
            for pair_id in four_condition_pair_ids:
                if include_occupancy:
                    four_condition_pair_figures.append(
                        plot_four_condition_occupancy(pair_id)
                    )
                if include_trajectory_speed:
                    four_condition_pair_figures.append(
                        plot_four_condition_traj_speed(pair_id)
                    )

            plot_test.save_figures_svg(
                four_condition_pair_figures,
                f"paired_behavior_{true_label}_vs_{false_label}",
                save_dir=behavior_svg_dir,
                enabled=save_behavior_svg,
            )

        true_roi_time_ratio_summary = pd.DataFrame()
        false_roi_time_ratio_summary = pd.DataFrame()
        roi_split_summary = pd.DataFrame()
        roi_time_ratio_four_condition_fig = None

        def roi_summary_for_merge(summary_df, split_label):
            if summary_df.empty:
                return pd.DataFrame(
                    columns=[
                        "pair_id",
                        "subject",
                        f"{split_label}_saline",
                        f"{split_label}_dcz",
                    ]
                )

            roi_df = summary_df.copy()
            if "subject" not in roi_df.columns and "pair_id" in roi_df.columns:
                roi_df["subject"] = roi_df["pair_id"].str[:6]
            for column in ["pair_id", "subject", "saline", "DCZ"]:
                if column not in roi_df.columns:
                    roi_df[column] = pd.Series(dtype=float)
            return roi_df[
                ["pair_id", "subject", "saline", "DCZ"]
            ].rename(
                columns={
                    "saline": f"{split_label}_saline",
                    "DCZ": f"{split_label}_dcz",
                }
            )

        def comparison_pvalue(group_df, left_col, right_col):
            paired_df = group_df[[left_col, right_col]].dropna()
            if paired_df.empty:
                return float("nan")
            try:
                return stats.wilcoxon(
                    paired_df[left_col],
                    paired_df[right_col],
                ).pvalue
            except ValueError:
                return float("nan")

        def add_sig_bar(ax, x0, x1, y, h, label):
            ax.plot(
                [x0, x0, x1, x1],
                [y, y + h, y + h, y],
                color="black",
                linewidth=1,
            )
            ax.text(
                (x0 + x1) / 2,
                y + h,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
            )

        def plot_roi_ratio_four_conditions(summary_df):
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(12, 5),
                sharey=True,
            )

            condition_cols = [
                "true_saline",
                "true_dcz",
                "false_saline",
                "false_dcz",
            ]
            condition_labels = [
                f"{true_label}\nsaline",
                f"{true_label}\nDCZ",
                f"{false_label}\nsaline",
                f"{false_label}\nDCZ",
            ]
            condition_colors = ["#2563eb", "#dc2626", "#93c5fd", "#fca5a5"]
            comparisons = [
                (0, 1, "true_saline", "true_dcz"),
                (2, 3, "false_saline", "false_dcz"),
                (0, 2, "true_saline", "false_saline"),
                (1, 3, "true_dcz", "false_dcz"),
            ]

            for ax, mice, group_name in zip(
                axes,
                [hM4Di_mice, hM3Dq_mice],
                ["hM4Di", "hM3Dq"],
            ):
                group_df = summary_df[
                    summary_df["subject"].isin(mice)
                ].copy()
                for column in condition_cols:
                    if column not in group_df.columns:
                        group_df[column] = pd.Series(dtype=float)

                for x0, x1, left_col, right_col in comparisons:
                    paired_df = group_df[[left_col, right_col]].dropna()
                    for _, row in paired_df.iterrows():
                        ax.plot(
                            [x0, x1],
                            [row[left_col], row[right_col]],
                            color="gray",
                            alpha=0.25,
                            linewidth=1,
                            zorder=1,
                        )

                for x, (column, color) in enumerate(zip(condition_cols, condition_colors)):
                    values = pd.to_numeric(group_df[column], errors="coerce").dropna()
                    ax.scatter(
                        [x] * len(values),
                        values,
                        color=color,
                        edgecolor="black",
                        zorder=3,
                    )

                all_values = pd.to_numeric(
                    group_df[condition_cols].stack(),
                    errors="coerce",
                ).dropna()
                if all_values.empty:
                    y_max = 1.0
                    y_range = 0.1
                else:
                    y_max = float(all_values.max())
                    y_min = float(all_values.min())
                    y_range = y_max - y_min
                    if not np.isfinite(y_range) or y_range == 0:
                        y_range = max(abs(y_max), 1.0) * 0.1

                h = y_range * 0.04
                for idx, (x0, x1, left_col, right_col) in enumerate(comparisons):
                    p_value = comparison_pvalue(group_df, left_col, right_col)
                    add_sig_bar(
                        ax,
                        x0,
                        x1,
                        y_max + y_range * (0.10 + 0.13 * idx),
                        h,
                        plot_test.p_to_star(p_value),
                    )

                ax.set_xticks(range(4))
                ax.set_xticklabels(condition_labels)
                ax.set_ylabel("fraction of time in ROI")
                ax.set_title(f"{group_name}, n={group_df['pair_id'].nunique()}")
                ax.grid(axis="y", alpha=0.3)
                ax.set_ylim(
                    bottom=0,
                    top=y_max + y_range * 0.75,
                )

            fig.tight_layout()
            return fig

        if include_roi_time:
            true_roi_time_ratio_summary = behavior_utils.paired_roi_time_ratio_comparison(
                behav_df_dic_saline=behav_df_dic_saline_true,
                behav_df_dic_dcz=behav_df_dic_dcz_true,
                pair_map=behav_pair_map,
                roi_left=roi_left,
                roi_right=roi_right,
                roi_bottom=roi_bottom,
                roi_top=roi_top,
                bodypart=bodypart,
            )
            false_roi_time_ratio_summary = behavior_utils.paired_roi_time_ratio_comparison(
                behav_df_dic_saline=behav_df_dic_saline_false,
                behav_df_dic_dcz=behav_df_dic_dcz_false,
                pair_map=behav_pair_map,
                roi_left=roi_left,
                roi_right=roi_right,
                roi_bottom=roi_bottom,
                roi_top=roi_top,
                bodypart=bodypart,
            )
            roi_split_summary = roi_summary_for_merge(
                true_roi_time_ratio_summary,
                "true",
            ).merge(
                roi_summary_for_merge(false_roi_time_ratio_summary, "false"),
                on=["pair_id", "subject"],
                how="outer",
            )
            roi_time_ratio_four_condition_fig = plot_roi_ratio_four_conditions(
                roi_split_summary,
            )

            plot_test.save_figure_svg(
                roi_time_ratio_four_condition_fig,
                f"roi_time_ratio_groups_{true_label}_vs_{false_label}",
                save_dir=behavior_svg_dir,
                enabled=save_behavior_svg,
            )

        stationary_speed_figures = []
        true_stationary_time_ratio_summary = pd.DataFrame()
        false_stationary_time_ratio_summary = pd.DataFrame()
        stationary_split_summary = pd.DataFrame()
        stationary_time_ratio_four_condition_fig = None

        def has_speed_data(behav_df):
            if behav_df.empty:
                return False
            if bodypart not in behav_df.columns.get_level_values(0):
                return False
            if speed_col not in behav_df[bodypart].columns:
                return False
            speed = pd.to_numeric(behav_df[(bodypart, speed_col)], errors="coerce")
            return speed.notna().any()

        def stationary_conditions_for_pair(pair_id):
            return [
                (
                    f"{true_label} saline",
                    behav_df_dic_saline_true[pair_id],
                    trial_df_for_split_value(df_dic_saline[pair_id], True),
                    "#2563eb",
                ),
                (
                    f"{true_label} DCZ",
                    behav_df_dic_dcz_true[pair_id],
                    trial_df_for_split_value(df_dic_dcz[pair_id], True),
                    "#dc2626",
                ),
                (
                    f"{false_label} saline",
                    behav_df_dic_saline_false[pair_id],
                    trial_df_for_split_value(df_dic_saline[pair_id], False),
                    "#93c5fd",
                ),
                (
                    f"{false_label} DCZ",
                    behav_df_dic_dcz_false[pair_id],
                    trial_df_for_split_value(df_dic_dcz[pair_id], False),
                    "#fca5a5",
                ),
            ]

        def frame_time_speed_df(behav_df):
            timestamp = pd.to_numeric(
                behavior_utils.get_behavior_column(behav_df, ("timestamp", "")),
                errors="coerce",
            )
            speed = pd.to_numeric(
                behavior_utils.get_behavior_column(behav_df, (bodypart, speed_col)),
                errors="coerce",
            )
            frame_df = pd.DataFrame(
                {
                    "timestamp": timestamp.to_numpy(),
                    "speed": speed.to_numpy(),
                }
            ).replace([np.inf, -np.inf], np.nan)
            return frame_df.dropna(subset=["timestamp"]).sort_values("timestamp")

        def frame_intervals(frame_df):
            t = frame_df["timestamp"].to_numpy(dtype=float)
            speed_values = frame_df["speed"].to_numpy(dtype=float)
            dt = np.diff(t)
            positive_dt = dt[np.isfinite(dt) & (dt > 0)]
            median_dt = float(np.median(positive_dt)) if len(positive_dt) else 0.0
            frame_end = np.empty_like(t)
            if len(t) > 1:
                frame_end[:-1] = t[1:]
            frame_end[-1] = t[-1] + median_dt
            return t, frame_end, speed_values

        def shade_stationary_segments_in_trials(ax, behav_df, trial_df):
            if behav_df.empty or trial_df.empty:
                return

            frame_df = frame_time_speed_df(behav_df)
            if frame_df.empty:
                return

            t, frame_end, speed_values = frame_intervals(frame_df)
            stationary = np.isfinite(speed_values) & (
                speed_values <= stationary_speed_threshold
            )

            for _, trial_row in trial_df.iterrows():
                try:
                    trial_start = float(trial_row["TRIAL_START"])
                    trial_end = float(trial_row["TRIAL_END"])
                except (TypeError, ValueError):
                    continue

                if (
                    not np.isfinite(trial_start)
                    or not np.isfinite(trial_end)
                    or trial_end <= trial_start
                ):
                    continue

                overlap_start = np.maximum(t, trial_start)
                overlap_end = np.minimum(frame_end, trial_end)
                draw_mask = stationary & (overlap_end > overlap_start)
                starts = overlap_start[draw_mask]
                ends = overlap_end[draw_mask]
                if len(starts) == 0:
                    continue

                run_start = starts[0]
                run_end = ends[0]
                for start, end in zip(starts[1:], ends[1:]):
                    if start <= run_end + 1e-9:
                        run_end = max(run_end, end)
                    else:
                        ax.axvspan(
                            run_start,
                            run_end,
                            color="gray",
                            alpha=0.18,
                            linewidth=0,
                        )
                        run_start = start
                        run_end = end

                ax.axvspan(
                    run_start,
                    run_end,
                    color="gray",
                    alpha=0.18,
                    linewidth=0,
                )

        def plot_four_condition_stationary_speed_trace(pair_id):
            condition_dfs = stationary_conditions_for_pair(pair_id)
            fig, axes = plt.subplots(
                1,
                4,
                figsize=(18, 4),
                sharey=True,
            )

            for ax, (condition_label, behav_df, trial_df, color) in zip(axes, condition_dfs):
                if has_speed_data(behav_df):
                    trace_df, x_label = plot_test._speed_trace_df(
                        behav_df,
                        bodypart=bodypart,
                        speed_col=speed_col,
                    )
                    if trace_df.empty:
                        ax.text(
                            0.5,
                            0.5,
                            "No speed",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                    else:
                        shade_stationary_segments_in_trials(ax, behav_df, trial_df)
                        ax.plot(
                            trace_df["x"],
                            trace_df["speed"],
                            color=color,
                            linewidth=1,
                            alpha=0.85,
                        )
                        ax.set_xlabel(x_label)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No speed",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

                ax.axhline(
                    stationary_speed_threshold,
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                )
                ax.set_title(condition_label, fontsize=10)
                ax.grid(axis="y", alpha=0.25)

            axes[0].set_ylabel(f"{bodypart} {speed_col} (pixels/s)")
            fig.suptitle(f"{pair_id} stationary speed trace by {split_column}", fontsize=12)
            fig.tight_layout()
            return fig

        def stationary_time_ratio_in_trials(behav_df, trial_df):
            if behav_df.empty or trial_df.empty:
                return np.nan, np.nan, 0.0

            frame_df = frame_time_speed_df(behav_df)
            if frame_df.empty:
                return np.nan, np.nan, 0.0

            t, frame_end, speed_values = frame_intervals(frame_df)
            stationary = np.isfinite(speed_values) & (
                speed_values <= stationary_speed_threshold
            )
            stationary_time = 0.0
            trial_time = 0.0
            for _, trial_row in trial_df.iterrows():
                try:
                    trial_start = float(trial_row["TRIAL_START"])
                    trial_end = float(trial_row["TRIAL_END"])
                except (TypeError, ValueError):
                    continue

                if (
                    not np.isfinite(trial_start)
                    or not np.isfinite(trial_end)
                    or trial_end <= trial_start
                ):
                    continue

                trial_time += trial_end - trial_start
                overlap_start = np.maximum(t, trial_start)
                overlap_end = np.minimum(frame_end, trial_end)
                overlap = np.clip(overlap_end - overlap_start, 0, None)
                stationary_time += float(overlap[stationary].sum())

            if trial_time <= 0:
                return np.nan, stationary_time, trial_time
            return stationary_time / trial_time, stationary_time, trial_time

        def paired_trial_stationary_time_ratio_comparison(target_value):
            rows = []
            for pair_id in pair_ids:
                if (
                    pair_id not in behav_df_dic_saline
                    or pair_id not in behav_df_dic_dcz
                    or pair_id not in df_dic_saline
                    or pair_id not in df_dic_dcz
                ):
                    continue

                saline_ratio, saline_stationary_time, saline_trial_time = (
                    stationary_time_ratio_in_trials(
                        behav_df_dic_saline[pair_id],
                        trial_df_for_split_value(df_dic_saline[pair_id], target_value),
                    )
                )
                dcz_ratio, dcz_stationary_time, dcz_trial_time = (
                    stationary_time_ratio_in_trials(
                        behav_df_dic_dcz[pair_id],
                        trial_df_for_split_value(df_dic_dcz[pair_id], target_value),
                    )
                )
                rows.append(
                    {
                        "pair_id": pair_id,
                        "saline": saline_ratio,
                        "DCZ": dcz_ratio,
                        "DCZ_minus_saline": dcz_ratio - saline_ratio,
                        "saline_stationary_time": saline_stationary_time,
                        "saline_trial_time": saline_trial_time,
                        "DCZ_stationary_time": dcz_stationary_time,
                        "DCZ_trial_time": dcz_trial_time,
                        "speed_threshold": stationary_speed_threshold,
                        "split_column": split_column,
                        "split_value": target_value,
                    }
                )

            summary_df = pd.DataFrame(rows)
            if not summary_df.empty:
                summary_df = summary_df.dropna(subset=["saline", "DCZ"])
            if (
                behav_pair_map is not None
                and not behav_pair_map.empty
                and not summary_df.empty
            ):
                summary_df = behav_pair_map.merge(
                    summary_df,
                    on="pair_id",
                    how="inner",
                )
            return summary_df

        def stationary_summary_for_merge(summary_df, split_label):
            if summary_df.empty:
                return pd.DataFrame(
                    columns=[
                        "pair_id",
                        "subject",
                        f"{split_label}_saline",
                        f"{split_label}_dcz",
                    ]
                )

            stationary_df = summary_df.copy()
            if "subject" not in stationary_df.columns and "pair_id" in stationary_df.columns:
                stationary_df["subject"] = stationary_df["pair_id"].str[:6]
            for column in ["pair_id", "subject", "saline", "DCZ"]:
                if column not in stationary_df.columns:
                    stationary_df[column] = pd.Series(dtype=float)
            return stationary_df[
                ["pair_id", "subject", "saline", "DCZ"]
            ].rename(
                columns={
                    "saline": f"{split_label}_saline",
                    "DCZ": f"{split_label}_dcz",
                }
            )

        def plot_stationary_ratio_four_conditions(summary_df):
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(12, 5),
                sharey=True,
            )

            condition_cols = [
                "true_saline",
                "true_dcz",
                "false_saline",
                "false_dcz",
            ]
            condition_labels = [
                f"{true_label}\nsaline",
                f"{true_label}\nDCZ",
                f"{false_label}\nsaline",
                f"{false_label}\nDCZ",
            ]
            condition_colors = ["#2563eb", "#dc2626", "#93c5fd", "#fca5a5"]
            comparisons = [
                (0, 1, "true_saline", "true_dcz"),
                (2, 3, "false_saline", "false_dcz"),
                (0, 2, "true_saline", "false_saline"),
                (1, 3, "true_dcz", "false_dcz"),
            ]

            for ax, mice, group_name in zip(
                axes,
                [hM4Di_mice, hM3Dq_mice],
                ["hM4Di", "hM3Dq"],
            ):
                group_df = summary_df[summary_df["subject"].isin(mice)].copy()
                for column in condition_cols:
                    if column not in group_df.columns:
                        group_df[column] = pd.Series(dtype=float)

                for x0, x1, left_col, right_col in comparisons:
                    paired_df = group_df[[left_col, right_col]].dropna()
                    for _, row in paired_df.iterrows():
                        ax.plot(
                            [x0, x1],
                            [row[left_col], row[right_col]],
                            color="gray",
                            alpha=0.25,
                            linewidth=1,
                            zorder=1,
                        )

                for x, (column, color) in enumerate(zip(condition_cols, condition_colors)):
                    values = pd.to_numeric(group_df[column], errors="coerce").dropna()
                    ax.scatter(
                        [x] * len(values),
                        values,
                        color=color,
                        edgecolor="black",
                        zorder=3,
                    )

                all_values = pd.to_numeric(
                    group_df[condition_cols].stack(),
                    errors="coerce",
                ).dropna()
                if all_values.empty:
                    y_max = 1.0
                    y_range = 0.1
                else:
                    y_max = float(all_values.max())
                    y_min = float(all_values.min())
                    y_range = y_max - y_min
                    if not np.isfinite(y_range) or y_range == 0:
                        y_range = max(abs(y_max), 1.0) * 0.1

                h = y_range * 0.04
                for idx, (x0, x1, left_col, right_col) in enumerate(comparisons):
                    p_value = comparison_pvalue(group_df, left_col, right_col)
                    add_sig_bar(
                        ax,
                        x0,
                        x1,
                        y_max + y_range * (0.10 + 0.13 * idx),
                        h,
                        plot_test.p_to_star(p_value),
                    )

                ax.set_xticks(range(4))
                ax.set_xticklabels(condition_labels)
                ax.set_ylabel("fraction of time stationary")
                ax.set_title(f"{group_name}, n={group_df['pair_id'].nunique()}")
                ax.grid(axis="y", alpha=0.3)
                ax.set_ylim(
                    bottom=0,
                    top=y_max + y_range * 0.75,
                )

            fig.tight_layout()
            return fig

        if include_stationary and include_stationary_speed:
            stationary_pair_ids = [
                pair_id
                for pair_id in pair_ids
                if pair_id in behav_df_dic_saline_true
                and pair_id in behav_df_dic_dcz_true
                and pair_id in behav_df_dic_saline_false
                and pair_id in behav_df_dic_dcz_false
                and any(
                    has_speed_data(behav_df)
                    for _, behav_df, _, _ in stationary_conditions_for_pair(pair_id)
                )
            ]
            stationary_speed_figures = [
                plot_four_condition_stationary_speed_trace(pair_id)
                for pair_id in stationary_pair_ids
            ]

            plot_test.save_figures_svg(
                stationary_speed_figures,
                f"stationary_speed_trace_{true_label}_vs_{false_label}",
                save_dir=behavior_svg_dir,
                enabled=save_behavior_svg,
            )

        if include_stationary and include_stationary_time:
            true_stationary_time_ratio_summary = (
                paired_trial_stationary_time_ratio_comparison(True)
            )
            false_stationary_time_ratio_summary = (
                paired_trial_stationary_time_ratio_comparison(False)
            )
            stationary_split_summary = stationary_summary_for_merge(
                true_stationary_time_ratio_summary,
                "true",
            ).merge(
                stationary_summary_for_merge(
                    false_stationary_time_ratio_summary,
                    "false",
                ),
                on=["pair_id", "subject"],
                how="outer",
            )
            stationary_time_ratio_four_condition_fig = (
                plot_stationary_ratio_four_conditions(stationary_split_summary)
            )

            plot_test.save_figure_svg(
                stationary_time_ratio_four_condition_fig,
                f"stationary_time_ratio_groups_{true_label}_vs_{false_label}",
                save_dir=behavior_svg_dir,
                enabled=save_behavior_svg,
            )

        view_items = []
        if four_condition_pair_figures:
            view_items.extend(
                [
                    mo.md(f"## {true_label} vs {false_label} paired behavior"),
                    *four_condition_pair_figures,
                ]
            )
        if roi_time_ratio_four_condition_fig is not None:
            view_items.extend(
                [
                    mo.md(f"## {true_label} vs {false_label} ROI time ratio groups"),
                    roi_time_ratio_four_condition_fig,
                ]
            )
        if stationary_speed_figures:
            view_items.extend(
                [
                    mo.md(f"## {true_label} vs {false_label} stationary speed traces"),
                    *stationary_speed_figures,
                ]
            )
        if stationary_time_ratio_four_condition_fig is not None:
            view_items.extend(
                [
                    mo.md(f"## {true_label} vs {false_label} stationary time ratio groups"),
                    stationary_time_ratio_four_condition_fig,
                ]
            )
        if not view_items:
            view_items.append(mo.md(f"## {true_label} vs {false_label}: no output selected"))

        return {
            "split_column": split_column,
            "true_label": true_label,
            "false_label": false_label,
            "true_pair_ids": true_pair_ids,
            "false_pair_ids": false_pair_ids,
            "include_stationary": include_stationary,
            "include_occupancy": include_occupancy,
            "include_trajectory_speed": include_trajectory_speed,
            "include_roi_time": include_roi_time,
            "include_stationary_speed": include_stationary_speed,
            "include_stationary_time": include_stationary_time,
            "stationary_speed_threshold": stationary_speed_threshold,
            "behav_df_dic_dcz_true": behav_df_dic_dcz_true,
            "behav_df_dic_dcz_false": behav_df_dic_dcz_false,
            "behav_df_dic_saline_true": behav_df_dic_saline_true,
            "behav_df_dic_saline_false": behav_df_dic_saline_false,
            "four_condition_pair_ids": four_condition_pair_ids,
            "four_condition_pair_figures": four_condition_pair_figures,
            "true_roi_time_ratio_summary": true_roi_time_ratio_summary,
            "false_roi_time_ratio_summary": false_roi_time_ratio_summary,
            "roi_split_summary": roi_split_summary,
            "roi_time_ratio_four_condition_fig": roi_time_ratio_four_condition_fig,
            "stationary_speed_figures": stationary_speed_figures,
            "true_stationary_time_ratio_summary": true_stationary_time_ratio_summary,
            "false_stationary_time_ratio_summary": false_stationary_time_ratio_summary,
            "stationary_split_summary": stationary_split_summary,
            "stationary_time_ratio_four_condition_fig": stationary_time_ratio_four_condition_fig,
            "view": mo.vstack(view_items),
        }

    return (analyze_paired_behavior_by_trial_column,)


@app.cell
def _(
    analyze_paired_behavior_by_trial_column,
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    df_dic_dcz,
    df_dic_saline,
    hM3Dq_mice,
    hM4Di_mice,
    roi_bottom,
    roi_left,
    roi_right,
    roi_top,
    save_behavior_svg,
):
    trial_split_analysis = analyze_paired_behavior_by_trial_column(
        split_column="engaged",
        df_dic_saline=df_dic_saline,
        df_dic_dcz=df_dic_dcz,
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        behav_pair_map=behav_pair_map,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,
        hM4Di_mice=hM4Di_mice,
        hM3Dq_mice=hM3Dq_mice,
        behavior_svg_dir=behavior_svg_dir,
        save_behavior_svg=save_behavior_svg,
        bodypart="Center",
        include_stationary=True,
        include_occupancy=False,
        include_trajectory_speed=False,
        include_roi_time=False,
        include_stationary_speed=True,
        include_stationary_time=False,
        stationary_speed_threshold=10,
        speed_col="mean_speed",
    )
    trial_split_analysis["view"]
    return (trial_split_analysis,)


@app.cell
def _(
    analyze_paired_behavior_by_trial_column,
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    df_dic_dcz,
    df_dic_saline,
    hM3Dq_mice,
    hM4Di_mice,
    mo,
    roi_bottom,
    roi_left,
    roi_right,
    roi_top,
    save_behavior_svg,
):
    analysis_by_column = {}
    view_list = []

    for column_name in ["engaged", "correct", "previous_correct"]:
        analysis = analyze_paired_behavior_by_trial_column(
            split_column=column_name,
            df_dic_saline=df_dic_saline,
            df_dic_dcz=df_dic_dcz,
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            behav_pair_map=behav_pair_map,
            roi_left=roi_left,
            roi_right=roi_right,
            roi_bottom=roi_bottom,
            roi_top=roi_top,
            hM4Di_mice=hM4Di_mice,
            hM3Dq_mice=hM3Dq_mice,
            behavior_svg_dir=behavior_svg_dir,
            save_behavior_svg=save_behavior_svg,
            bodypart="Center",
            include_stationary=True,
            include_occupancy=False,
            include_trajectory_speed=False,
            include_roi_time=False,
            include_stationary_speed=False,
            include_stationary_time=True,
            stationary_speed_threshold=10,
            speed_col="mean_speed",
        )

        analysis_by_column[column_name] = analysis
        view_list.append(mo.md(f"## split column: {column_name}"))
        view_list.append(analysis["view"])

    trial_split_analysis_compa_1 = analysis_by_column["engaged"]
    trial_split_analysis_compa_2 = analysis_by_column["correct"]
    trial_split_analysis_compa_3 = analysis_by_column["previous_correct"]

    mo.vstack(view_list)
    return


@app.cell
def _(
    analyze_paired_behavior_by_trial_column,
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    df_dic_dcz,
    df_dic_saline,
    hM3Dq_mice,
    hM4Di_mice,
    roi_bottom,
    roi_left,
    roi_right,
    roi_top,
    save_behavior_svg,
):

    trial_split_analysis_engage_roi = analyze_paired_behavior_by_trial_column(
        split_column="engaged",
        df_dic_saline=df_dic_saline,
        df_dic_dcz=df_dic_dcz,
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        behav_pair_map=behav_pair_map,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,
        hM4Di_mice=hM4Di_mice,
        hM3Dq_mice=hM3Dq_mice,
        behavior_svg_dir=behavior_svg_dir,
        save_behavior_svg=save_behavior_svg,
        bodypart="Center",
        include_stationary=False,
        include_occupancy=True,
        include_trajectory_speed=True,
        include_roi_time=False,
        include_stationary_speed=False,
        include_stationary_time=False,
        stationary_speed_threshold=10,
        speed_col="mean_speed",
    )
    trial_split_analysis_engage_roi["view"]
    return


@app.cell
def _(
    analyze_paired_behavior_by_trial_column,
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    df_dic_dcz,
    df_dic_saline,
    hM3Dq_mice,
    hM4Di_mice,
    mo,
    roi_bottom,
    roi_left,
    roi_right,
    roi_top,
    save_behavior_svg,
):
    analysis_by_column_roi = {}
    view_list_roi = []

    for column_name_roi in ["engaged", "correct", "previous_correct"]:
        analysis_roi = analyze_paired_behavior_by_trial_column(
            split_column=column_name_roi,
            df_dic_saline=df_dic_saline,
            df_dic_dcz=df_dic_dcz,
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            behav_pair_map=behav_pair_map,
            roi_left=roi_left,
            roi_right=roi_right,
            roi_bottom=roi_bottom,
            roi_top=roi_top,
            hM4Di_mice=hM4Di_mice,
            hM3Dq_mice=hM3Dq_mice,
            behavior_svg_dir=behavior_svg_dir,
            save_behavior_svg=save_behavior_svg,
            bodypart="Center",
            include_stationary=False,
            include_occupancy=False,
            include_trajectory_speed=False,
            include_roi_time=True,
            include_stationary_speed=False,
            include_stationary_time=False,
            stationary_speed_threshold=10,
            speed_col="mean_speed",
        )

        analysis_by_column_roi[column_name_roi] = analysis_roi
        view_list_roi.append(mo.md(f"## split column: {column_name_roi}"))
        view_list_roi.append(analysis_roi["view"])

    trial_split_analysis_roi_compa_1 = analysis_by_column_roi["engaged"]
    trial_split_analysis_roi_compa_2 = analysis_by_column_roi["correct"]
    trial_split_analysis_roi_compa_3 = analysis_by_column_roi["previous_correct"]

    mo.vstack(view_list_roi)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # extract trace after choice
    """)
    return


@app.cell
def _(df_test_hm3, np, pd, plt):
    import ast
    import re
    from matplotlib.lines import Line2D

    trial_state_cols = [
        "STATE_auto_reward_state_left_START",
        "STATE_hold_center_port_END",
        "STATE_hold_center_port_START",
        "STATE_hold_while_stimulus_END",
        "STATE_hold_while_stimulus_START",
        "STATE_iti_END",
        "STATE_iti_START",
        "STATE_punish_state_END",
        "STATE_punish_state_START",
        "STATE_ready_to_initiate_END",
        "STATE_ready_to_initiate_START",
        "STATE_reward_state_END",
        "STATE_reward_state_START",
        "STATE_reward_state_left_END",
        "STATE_reward_state_left_START",
        "STATE_reward_state_right_END",
        "STATE_reward_state_right_START",
        "STATE_start_of_trial_END",
        "STATE_start_of_trial_START",
        "STATE_stimulus_state_END",
        "STATE_stimulus_state_START",
        "TRIAL_END",
        "TRIAL_START",
    ]

    def as_event_times(value):
        if value is None:
            return []

        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            out = []
            for item in value:
                try:
                    item = float(item)
                    if np.isfinite(item):
                        out.append(item)
                except Exception:
                    pass
            return out

        if pd.isna(value):
            return []

        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
                return as_event_times(value)
            except Exception:
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
                return [float(num) for num in nums]

        try:
            value = float(value)
            return [value] if np.isfinite(value) else []
        except Exception:
            return []


    plot_df = df_test_hm3.head(5).copy()

    event_cols = [col for col in trial_state_cols if col in plot_df.columns]

    event_color_map = {
        col: plt.get_cmap("tab20")(idx % 20)
        for idx, col in enumerate(event_cols)
    }

    fig, ax = plt.subplots(figsize=(13, 4))

    for trial_idx, (_, row) in enumerate(plot_df.iterrows()):
        y = len(plot_df) - trial_idx

        trial_start_times = as_event_times(row.get("TRIAL_START", np.nan))
        t0 = trial_start_times[0]

        for col in event_cols:
            event_times = as_event_times(row.get(col, np.nan))
            for event_time in event_times:
                ax.vlines(
                    event_time,
                    y - 0.38,
                    y + 0.38,
                    color=event_color_map[col],
                    linewidth=2,
                    alpha=0.95,
                )

    ax.set_yticks(range(1, len(plot_df) + 1))
    ax.set_yticklabels([f"trial {i}" for i in range(len(plot_df), 0, -1)])
    ax.set_xlabel("time from TRIAL_START (s)")
    ax.set_ylabel("trial")
    ax.set_title("First 5 trials state event raster")
    ax.grid(axis="x", alpha=0.25)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=event_color_map[col],
            linewidth=2,
            label=col.replace("STATE_", ""),
        )
        for col in event_cols
    ]

    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=8,
    )

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(trial_split_analysis):
    trial_split_analysis["behav_df_dic_saline_true"].get("NUO005_pair_01")
    return


@app.cell
def _(trial_split_analysis):
    trial_split_analysis["behav_df_dic_saline_false"].get("NUO005_pair_01")
    return


@app.cell
def _(df_dic_saline):
    df_dic_saline['NUO005_pair_01']['previous_correct']
    return


@app.cell
def _(behav_df_dic_saline):
    behav_df_dic_saline['NUO005_pair_01'][('Center', 'distance')]
    return


if __name__ == "__main__":
    app.run()
