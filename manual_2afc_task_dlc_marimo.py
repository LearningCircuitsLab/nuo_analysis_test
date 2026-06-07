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

    def split_behavior_by_engagement(behav_df, trial_df):
        empty_behav_df = behav_df.iloc[0:0].copy()
        if behav_df.empty or trial_df.empty or "engaged" not in trial_df.columns:
            return empty_behav_df, empty_behav_df.copy()

        timestamp = pd.to_numeric(
            behavior_utils.get_behavior_column(behav_df, ("timestamp", "")),
            errors="coerce",
        )
        engage_mask = pd.Series(False, index=behav_df.index)
        disengage_mask = pd.Series(False, index=behav_df.index)

        for _, trial_row in trial_df.iterrows():
            trial_start = float(trial_row["TRIAL_START"])
            trial_end = float(trial_row["TRIAL_END"])

            if not np.isfinite(trial_start) or not np.isfinite(trial_end):
                continue

            trial_mask = timestamp.between(
                trial_start,
                trial_end,
                inclusive="both",
            ).fillna(False)
            if trial_row["engaged"]:
                engage_mask = engage_mask | trial_mask
            else:
                disengage_mask = disengage_mask | trial_mask

        return behav_df.loc[engage_mask].copy(), behav_df.loc[disengage_mask].copy()

    df_test_upd_pairing = df_test_upd.copy()
    df_test_upd_pairing["_pair_subject"] = df_test_upd_pairing["subject"].astype(str)
    df_test_upd_pairing["_pair_date"] = pd.to_datetime(
        df_test_upd_pairing["year_month_day"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    df_dic_dcz = {}
    df_dic_saline = {}
    behav_df_dic_dcz_engage = {}
    behav_df_dic_dcz_disengage = {}
    behav_df_dic_saline_engage = {}
    behav_df_dic_saline_disengage = {}

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
        (
            behav_df_dic_dcz_engage[pair_id],
            behav_df_dic_dcz_disengage[pair_id],
        ) = split_behavior_by_engagement(
            behav_df_dic_dcz[pair_id],
            df_dic_dcz[pair_id],
        )
        (
            behav_df_dic_saline_engage[pair_id],
            behav_df_dic_saline_disengage[pair_id],
        ) = split_behavior_by_engagement(
            behav_df_dic_saline[pair_id],
            df_dic_saline[pair_id],
        )
    return (
        behav_df_dic_dcz_disengage,
        behav_df_dic_dcz_engage,
        behav_df_dic_saline_disengage,
        behav_df_dic_saline_engage,
        df_dic_saline,
    )


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
def _(
    behav_df_dic_dcz_engage,
    behav_df_dic_saline_engage,
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
    engaged_pair_figures = plot_test.plot_paired_behavior_figures(
        pair_ids=behav_pair_map["pair_id"].unique(),
        behav_df_dic_saline=behav_df_dic_saline_engage,
        behav_df_dic_dcz=behav_df_dic_dcz_engage,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,
        bodypart="Center",
    )

    plot_test.save_figures_svg(
        engaged_pair_figures,
        "paired_behavior_engaged",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack(engaged_pair_figures)
    return


@app.cell
def _(
    behav_df_dic_dcz_disengage,
    behav_df_dic_saline_disengage,
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
    disengaged_pair_figures = plot_test.plot_paired_behavior_figures(
        pair_ids=behav_pair_map["pair_id"].unique(),
        behav_df_dic_saline=behav_df_dic_saline_disengage,
        behav_df_dic_dcz=behav_df_dic_dcz_disengage,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,
        bodypart="Center",
    )

    plot_test.save_figures_svg(
        disengaged_pair_figures,
        "paired_behavior_disengaged",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack(disengaged_pair_figures)
    return


@app.cell
def _(
    behav_df_dic_dcz_disengage,
    behav_df_dic_dcz_engage,
    behav_df_dic_saline_disengage,
    behav_df_dic_saline_engage,
    behav_pair_map,
    behavior_utils,
    roi_bottom,
    roi_left,
    roi_right,
    roi_top,
):
    roi_time_ratio_engaged_summary = behavior_utils.paired_roi_time_ratio_comparison(
        behav_df_dic_saline=behav_df_dic_saline_engage,
        behav_df_dic_dcz=behav_df_dic_dcz_engage,
        pair_map=behav_pair_map,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,
        bodypart="Center",
    )

    roi_time_ratio_disengaged_summary = behavior_utils.paired_roi_time_ratio_comparison(
        behav_df_dic_saline=behav_df_dic_saline_disengage,
        behav_df_dic_dcz=behav_df_dic_dcz_disengage,
        pair_map=behav_pair_map,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,
        bodypart="Center",
    )
    return roi_time_ratio_disengaged_summary, roi_time_ratio_engaged_summary


@app.cell
def _(
    behavior_svg_dir,
    hM3Dq_mice,
    hM4Di_mice,
    mo,
    plot_test,
    plt,
    roi_time_ratio_disengaged_summary,
    roi_time_ratio_engaged_summary,
    save_behavior_svg,
):
    from scipy import stats

    def plot_roi_group(ax, df, mice, group_name):
        roi_group_df = df.copy()
        if "subject" not in roi_group_df.columns:
            roi_group_df["subject"] = roi_group_df["pair_id"].str[:6]

        group_df = roi_group_df[
            roi_group_df["subject"].isin(mice)
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

    def plot_roi_ratio_groups(summary_df, title_prefix):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(8, 5),
            sharey=True,
        )

        plot_roi_group(
            axes[0],
            summary_df,
            hM4Di_mice,
            f"{title_prefix} hM4Di",
        )
        plot_roi_group(
            axes[1],
            summary_df,
            hM3Dq_mice,
            f"{title_prefix} hM3Dq",
        )

        fig.tight_layout()
        return fig

    roi_time_ratio_engaged_group_fig = plot_roi_ratio_groups(
        roi_time_ratio_engaged_summary,
        "engaged",
    )
    plot_test.save_figure_svg(
        roi_time_ratio_engaged_group_fig,
        "roi_time_ratio_groups_engaged",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    roi_time_ratio_disengaged_group_fig = plot_roi_ratio_groups(
        roi_time_ratio_disengaged_summary,
        "disengaged",
    )
    plot_test.save_figure_svg(
        roi_time_ratio_disengaged_group_fig,
        "roi_time_ratio_groups_disengaged",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack([
        roi_time_ratio_engaged_group_fig,
        roi_time_ratio_disengaged_group_fig,
    ])
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
def _(behav_df_dic_saline_engage):
    behav_df_dic_saline_engage["NUO005_pair_01"]
    return


@app.cell
def _(behav_df_dic_saline_disengage):
    behav_df_dic_saline_disengage["NUO005_pair_01"]
    return


@app.cell
def _(df_dic_saline):
    df_dic_saline['NUO005_pair_01']['TRIAL_END'][5449]
    return


@app.cell
def _(behav_df_dic_saline):
    behav_df_dic_saline['NUO005_pair_01'][('Center', 'distance')]
    return


if __name__ == "__main__":
    app.run()
