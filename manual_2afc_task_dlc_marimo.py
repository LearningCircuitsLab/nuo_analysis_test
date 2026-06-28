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
    return (
        Path,
        behavior_utils,
        dft,
        np,
        pd,
        plot_test,
        plots,
        plt,
        utils,
        utils_test,
    )


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
    hM3Dq_mice = ['NUO062', 'NUO063', 'NUO064', 'NUO065', 'NUO007', 'NUO008', 'NUO009', 'NUO010', 'NUO011', 'NUO012']
    hM4Di_mice = ['NUO057', 'NUO058', 'NUO059', 'NUO060', 'NUO061', 'NUO001', 'NUO002', 'NUO003', 'NUO005', 'NUO006']
    # hM3Dq_mice = ['NUO064', 'NUO065', 'NUO007', 'NUO008', 'NUO009', 'NUO010', 'NUO011', 'NUO012']
    # hM4Di_mice = ['NUO060', 'NUO061', 'NUO001', 'NUO002', 'NUO003', 'NUO005', 'NUO006']
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
def _(add_number_of_pokes, df_test, dft, hM3Dq_mice, hM4Di_mice, np, pd):
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

        session_dfs = []

        for (_, _), df_session in df.groupby(
            ["subject", "session"],
            sort=False,
        ):
            df_session = dft.add_trial_duration_column_to_df(
                df_session
            )

            # df_session = dft.add_engagement_column(
            #     df_session,
            #     engagement_sd_criteria=2,
            # )

            df_session = add_number_of_pokes(
                df_session,
                port_number=2,
            )

            df_session = dft.add_day_column_to_df(
                df_session
            )

            df_session = (
                dft.calculate_time_between_trials_and_reaction_time(
                    df_session
                )
            )

            session_dfs.append(df_session)

        return pd.concat(session_dfs).sort_index()

    def add_react_slow_column(df_in, reaction_sd_criteria=2):
        if df_in["subject"].nunique() > 1:
            raise ValueError("The dataframe should contain only one subject.")

        df = df_in.copy()
        reaction_time = pd.to_numeric(
            df["reaction_time"],
            errors="coerce",
        )
        reaction_time = reaction_time.where(reaction_time > 0)
        df["reaction_time_log"] = np.log(reaction_time)

        median_reaction_time_log = df["reaction_time_log"].median()
        std_reaction_time_log = df["reaction_time_log"].std()
        reaction_threshold = (
            median_reaction_time_log
            + reaction_sd_criteria * std_reaction_time_log
        )
        df["react_slow"] = (
            df["reaction_time_log"] > reaction_threshold
            if np.isfinite(reaction_threshold)
            else False
        )
        return df

    def add_break_more_column(df_in, break_sd_criteria=2):
        if df_in["subject"].nunique() > 1:
            raise ValueError("The dataframe should contain only one subject.")

        df = df_in.copy()
        port2_pokes = pd.to_numeric(
            df["port2_pokes_num"],
            errors="coerce",
        )
        df["port2_pokes_num_log"] = np.log(port2_pokes)

        median_port2_pokes_log = df["port2_pokes_num_log"].median()
        std_port2_pokes_log = df["port2_pokes_num_log"].std()
        break_threshold = (
            median_port2_pokes_log
            + break_sd_criteria * std_port2_pokes_log
        )
        df["break_more"] = (
            df["port2_pokes_num_log"] > break_threshold
            if np.isfinite(break_threshold)
            else False
        )
        return df

    def add_behavior_state_columns(df_subject):
        df_subject = dft.add_engagement_column(
            df_subject,
            engagement_sd_criteria=2,
        )
        df_subject = add_react_slow_column(
            df_subject,
            reaction_sd_criteria=2,
        )
        df_subject = add_break_more_column(
            df_subject,
            break_sd_criteria=2,
        )
        return df_subject

    df_test_upd = add_trial_variables_by_subject(df_test_upd)
    df_test_upd = (
        pd.concat(
            [
                add_behavior_state_columns(df_subject)
                for _, df_subject in df_test_upd.groupby(
                    "subject",
                    sort=False,
                )
            ]
        ).sort_index()
    )

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
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    df_dic_dcz,
    df_dic_saline,
    get_speed_per_mouse_sessions,
    hM3Dq_mice,
    hM4Di_mice,
    split_col,
    split_label,
):
    speed_per_animal_condition_hm4 = get_speed_per_mouse_sessions(
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        behav_pair_map=behav_pair_map,
        subjects=hM4Di_mice,
    )
    speed_per_animal_condition_hm4_diff = get_speed_per_mouse_sessions(
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        behav_pair_map=behav_pair_map,
        subjects=hM4Di_mice,
        difference=True,
    )
    speed_per_animal_condition_hm3 = get_speed_per_mouse_sessions(
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        behav_pair_map=behav_pair_map,
        subjects=hM3Dq_mice,
    )
    speed_per_animal_condition_hm3_diff = get_speed_per_mouse_sessions(
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        behav_pair_map=behav_pair_map,
        subjects=hM3Dq_mice,
        difference=True,
    )
    speed_per_animal_condition_hm4_split_col = (
        get_speed_per_mouse_sessions(
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            df_dic_saline=df_dic_saline,
            df_dic_dcz=df_dic_dcz,
            behav_pair_map=behav_pair_map,
            subjects=hM4Di_mice,
            split_column=split_col,
            split_labels=split_label,
        )
    )
    speed_per_animal_condition_hm4_diff_split_col = (
        get_speed_per_mouse_sessions(
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            df_dic_saline=df_dic_saline,
            df_dic_dcz=df_dic_dcz,
            behav_pair_map=behav_pair_map,
            subjects=hM4Di_mice,
            split_column=split_col,
            split_labels=split_label,
            difference=True,
        )
    )
    speed_per_animal_condition_hm3_split_col = (
        get_speed_per_mouse_sessions(
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            df_dic_saline=df_dic_saline,
            df_dic_dcz=df_dic_dcz,
            behav_pair_map=behav_pair_map,
            subjects=hM3Dq_mice,
            split_column=split_col,
            split_labels=split_label,
        )
    )
    speed_per_animal_condition_hm3_diff_split_col = (
        get_speed_per_mouse_sessions(
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            df_dic_saline=df_dic_saline,
            df_dic_dcz=df_dic_dcz,
            behav_pair_map=behav_pair_map,
            subjects=hM3Dq_mice,
            split_column=split_col,
            split_labels=split_label,
            difference=True,
        )
    )
    return (
        speed_per_animal_condition_hm3,
        speed_per_animal_condition_hm3_diff,
        speed_per_animal_condition_hm3_diff_split_col,
        speed_per_animal_condition_hm3_split_col,
        speed_per_animal_condition_hm4,
        speed_per_animal_condition_hm4_diff,
        speed_per_animal_condition_hm4_diff_split_col,
        speed_per_animal_condition_hm4_split_col,
    )


@app.cell
def _(mo):
    download_dlcData_button = mo.ui.run_button(label="Download / update the mice dlc data")
    download_dlcData_button
    return (download_dlcData_button,)


@app.cell
def _(project_select):
    project_select.value
    return


@app.cell
def _(project_select):
    if project_select.value == 'COT_cannula_GAD2_data':
        dlc_mice_selected = ['NUO057', 'NUO058', 'NUO059', 'NUO060', 'NUO061', 'NUO062', 'NUO063', 'NUO064', 'NUO012']
    if project_select.value == 'COT_cannula_data':
        dlc_mice_selected = ['NUO001', 'NUO002', 'NUO003', 'NUO005', 'NUO007', 'NUO008', 'NUO009', 'NUO010', 'NUO012']
    return (dlc_mice_selected,)


@app.cell
def _(mo):
    load_paired_dlc_button = mo.ui.run_button(label="Load paired DLC data")
    load_paired_dlc_button
    return


@app.cell
def _(
    behavior_utils,
    df_test_hm3,
    df_test_hm4,
    dlc_mice_selected,
    injection_info_df,
    pd,
):
    paired_dlc_dates = behavior_utils.get_paired_injection_dates(
        injection_info_df,
        mice_selected=dlc_mice_selected,
    )

    modality_source = pd.concat(
        [df_test_hm4, df_test_hm3],
        ignore_index=True,
    )
    modality_source["_modality_subject"] = modality_source["subject"].astype(str)
    modality_source["_modality_date"] = pd.to_datetime(
        modality_source["year_month_day"],
        errors="coerce",
    ).dt.normalize()

    modality_lookup = (
        modality_source.dropna(
            subset=[
                "_modality_subject",
                "_modality_date",
                "stimulus_modality",
            ]
        )
        .groupby(
            ["_modality_subject", "_modality_date"],
            sort=False,
        )["stimulus_modality"]
        .agg(lambda values: tuple(pd.unique(values.astype(str))))
        .to_dict()
    )

    def stimulus_modality_for_pair(pair_row):
        modalities = []
        subject = str(pair_row["subject"])
        for date_column in ["saline_date", "DCZ_date"]:
            pair_date = pd.to_datetime(
                pair_row[date_column],
                errors="coerce",
            )
            if pd.isna(pair_date):
                continue
            modalities.extend(
                modality_lookup.get(
                    (subject, pair_date.normalize()),
                    (),
                )
            )

        unique_modalities = list(dict.fromkeys(modalities))
        if not unique_modalities:
            return pd.NA
        return " / ".join(unique_modalities)

    paired_dlc_dates["stimulus_modality"] = paired_dlc_dates.apply(
        stimulus_modality_for_pair,
        axis=1,
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

    missing_video_files = []

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
        video_df = video_df.iloc[2:].reset_index(drop=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## extract video frame
    """)
    return


@app.cell
def _(behav_pair_map):
    behav_pair_map
    return


@app.cell
def _():
    # utils_test.rsync_specific_file(
    #     file_path="/home/kudongdong/data/training_village/2afc/image/NUO010/video2img/frame0.jpg",
    #     local_path="/mnt/e/data/LeciLab/behavioral_data/tmp/processing/video/",
    #     credentials=utils.get_idibaps_cluster_credentials(),
    # )
    return


@app.cell
def _(np):
    from PIL import Image
    import plotly.express as px

    img = np.asarray(
        Image.open(
            "/mnt/e/data/LeciLab/behavioral_data/tmp/processing/video/frame0.jpg"
        )
    )

    _fig = px.imshow(img, origin="upper")
    _fig.show()
    return


@app.cell
def _():
    zone_xy_1 = [135, 175]
    zone_xy_2 = [291, 110]
    zone_xy_3 = [351, 110]
    zone_xy_4 = [500, 175]
    zone_xy_5 = [500, 447]
    zone_xy_6 = [135, 447]
    port_xy_1 = [269, 111]
    port_xy_2 = [322, 105]
    port_xy_3 = [380, 113]
    return


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
def _(mo):
    stimulus_modality_select = mo.ui.dropdown(
        options=["visual", "auditory"],
        value=None,
        label="Stimulus modality",
    )
    stimulus_modality_select
    return (stimulus_modality_select,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # condition compare by animals
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## roi time
    """)
    return


@app.cell
def _(behavior_utils, np, pd):
    def _empty_roi_output(difference=False, by_engagement=False):
        condition_keys = (
            ["vis", "aud"]
            if difference
            else ["vis_saline", "vis_dcz", "aud_saline", "aud_dcz"]
        )
        return {condition_key: {} for condition_key in condition_keys}

    def _mouse_split_output(condition_dict, subject, split_labels):
        return condition_dict.setdefault(
            subject,
            {split_label: [] for split_label in split_labels},
        )

    def get_trial_roi_ratios_by_split_col(
        behav_df,
        trial_df,
        roi_left,
        roi_right,
        roi_bottom,
        roi_top,
        split_column="engaged",
        split_labels=("engaged", "disengaged"),
        bodypart="Center",
    ):
        true_label, false_label = split_labels
        trial_ratios = {true_label: [], false_label: []}
        roi_time_by_split = {true_label: 0.0, false_label: 0.0}
        trial_time_by_split = {true_label: 0.0, false_label: 0.0}

        timestamp = pd.to_numeric(
            behavior_utils.get_behavior_column(
                behav_df,
                ("timestamp", ""),
            ),
            errors="coerce",
        )
        x = pd.to_numeric(
            behavior_utils.get_behavior_column(
                behav_df,
                (bodypart, "x"),
            ),
            errors="coerce",
        )
        y = pd.to_numeric(
            behavior_utils.get_behavior_column(
                behav_df,
                (bodypart, "y"),
            ),
            errors="coerce",
        )

        frame_df = (
            pd.DataFrame(
                {
                    "timestamp": timestamp.to_numpy(),
                    "x": x.to_numpy(),
                    "y": y.to_numpy(),
                }
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["timestamp"])
            .sort_values("timestamp")
        )

        t = frame_df["timestamp"].to_numpy(dtype=float)
        x_values = frame_df["x"].to_numpy(dtype=float)
        y_values = frame_df["y"].to_numpy(dtype=float)
        dt = np.diff(t)
        positive_dt = dt[np.isfinite(dt) & (dt > 0)]
        median_dt = float(np.median(positive_dt)) if len(positive_dt) else 0.0

        frame_end = np.empty_like(t)
        if len(t) > 1:
            frame_end[:-1] = t[1:]
        frame_end[-1] = t[-1] + median_dt

        valid_position = np.isfinite(x_values) & np.isfinite(y_values)
        in_roi = (
            valid_position
            & (x_values >= roi_left)
            & (x_values <= roi_right)
            & (y_values >= roi_bottom)
            & (y_values <= roi_top)
        )

        for _, trial_row in trial_df.iterrows():
            split_label = true_label if bool(trial_row[split_column]) else false_label
            trial_start = pd.to_numeric(
                trial_row["TRIAL_START"],
                errors="coerce",
            )
            trial_end = pd.to_numeric(
                trial_row["TRIAL_END"],
                errors="coerce",
            )
            if (
                not np.isfinite(trial_start)
                or not np.isfinite(trial_end)
                or trial_end <= trial_start
            ):
                continue

            overlap_start = np.maximum(t, float(trial_start))
            overlap_end = np.minimum(frame_end, float(trial_end))
            overlap = np.clip(overlap_end - overlap_start, 0, None)
            trial_time = float(trial_end - trial_start)
            roi_time = float(overlap[in_roi].sum())
            roi_time_by_split[split_label] += roi_time
            trial_time_by_split[split_label] += trial_time

        for split_label in split_labels:
            if trial_time_by_split[split_label] > 0:
                trial_ratios[split_label].append(
                    roi_time_by_split[split_label]
                    / trial_time_by_split[split_label]
                )

        return trial_ratios

    def get_trial_roi_per_mouse_sessions(
        behav_df_dic_saline,
        behav_df_dic_dcz,
        df_dic_saline,
        df_dic_dcz,
        behav_pair_map,
        roi_left,
        roi_right,
        roi_bottom,
        roi_top,
        subjects=None,
        bodypart="Center",
        split_column="engaged",
        split_labels=("engaged", "disengaged"),
        difference=False,
    ):
        output = _empty_roi_output(
            difference=difference,
            by_engagement=True,
        )

        for _, pair_row in behav_pair_map.sort_values("pair_id").iterrows():
            pair_id = pair_row["pair_id"]
            subject = str(pair_row["subject"])
            if subjects is not None and subject not in subjects:
                continue
            if (
                pair_id not in behav_df_dic_saline
                or pair_id not in behav_df_dic_dcz
                or pair_id not in df_dic_saline
                or pair_id not in df_dic_dcz
            ):
                continue

            modality = str(pair_row.get("stimulus_modality", "")).lower()
            has_visual = "visual" in modality
            has_auditory = "auditory" in modality
            if has_visual == has_auditory:
                continue
            modality_prefix = "vis" if has_visual else "aud"

            saline_ratios = get_trial_roi_ratios_by_split_col(
                behav_df_dic_saline[pair_id],
                df_dic_saline[pair_id],
                roi_left=roi_left,
                roi_right=roi_right,
                roi_bottom=roi_bottom,
                roi_top=roi_top,
                split_column=split_column,
                split_labels=split_labels,
                bodypart=bodypart,
            )
            dcz_ratios = get_trial_roi_ratios_by_split_col(
                behav_df_dic_dcz[pair_id],
                df_dic_dcz[pair_id],
                roi_left=roi_left,
                roi_right=roi_right,
                roi_bottom=roi_bottom,
                roi_top=roi_top,
                split_column=split_column,
                split_labels=split_labels,
                bodypart=bodypart,
            )

            if difference:
                mouse_output = _mouse_split_output(
                    output[modality_prefix],
                    subject,
                    split_labels,
                )
                for split_label in split_labels:
                    saline_values = np.asarray(
                        saline_ratios[split_label],
                        dtype=float,
                    )
                    dcz_values = np.asarray(
                        dcz_ratios[split_label],
                        dtype=float,
                    )
                    if len(saline_values) == 0 or len(dcz_values) == 0:
                        continue
                    mouse_output[split_label].append(
                        {
                            "pair_id": pair_id,
                            "value": float(
                                np.nanmean(saline_values)
                                - np.nanmean(dcz_values)
                            ),
                        }
                    )
                continue

            saline_output = _mouse_split_output(
                output[f"{modality_prefix}_saline"],
                subject,
                split_labels,
            )
            dcz_output = _mouse_split_output(
                output[f"{modality_prefix}_dcz"],
                subject,
                split_labels,
            )
            for split_label in split_labels:
                saline_output[split_label].extend(saline_ratios[split_label])
                dcz_output[split_label].extend(dcz_ratios[split_label])

        return output


    def get_roi_per_mouse_sessions(
        behav_df_dic_saline,
        behav_df_dic_dcz,
        behav_pair_map,
        roi_left,
        roi_right,
        roi_bottom,
        roi_top,
        df_dic_saline=None,
        df_dic_dcz=None,
        subjects=None,
        bodypart="Center",
        split_column=None,
        split_labels=("engaged", "disengaged"),
        difference=False,
    ):
        if split_column is not None:
            return get_trial_roi_per_mouse_sessions(
                behav_df_dic_saline=behav_df_dic_saline,
                behav_df_dic_dcz=behav_df_dic_dcz,
                df_dic_saline=df_dic_saline,
                df_dic_dcz=df_dic_dcz,
                behav_pair_map=behav_pair_map,
                roi_left=roi_left,
                roi_right=roi_right,
                roi_bottom=roi_bottom,
                roi_top=roi_top,
                subjects=subjects,
                bodypart=bodypart,
                split_column=split_column,
                split_labels=split_labels,
                difference=difference,
            )

        output = _empty_roi_output(difference=difference)

        for _, pair_row in behav_pair_map.sort_values("pair_id").iterrows():
            pair_id = pair_row["pair_id"]
            subject = str(pair_row["subject"])
            if subjects is not None and subject not in subjects:
                continue
            if (
                pair_id not in behav_df_dic_saline
                or pair_id not in behav_df_dic_dcz
            ):
                continue

            modality = str(pair_row.get("stimulus_modality", "")).lower()
            has_visual = "visual" in modality
            has_auditory = "auditory" in modality
            if has_visual == has_auditory:
                continue
            modality_prefix = "vis" if has_visual else "aud"

            saline_ratio = behavior_utils.get_roi_time_ratio(
                behav_df_dic_saline[pair_id],
                roi_left=roi_left,
                roi_right=roi_right,
                roi_bottom=roi_bottom,
                roi_top=roi_top,
                bodypart=bodypart,
            )
            dcz_ratio = behavior_utils.get_roi_time_ratio(
                behav_df_dic_dcz[pair_id],
                roi_left=roi_left,
                roi_right=roi_right,
                roi_bottom=roi_bottom,
                roi_top=roi_top,
                bodypart=bodypart,
            )

            if difference:
                if pd.notna(saline_ratio) and pd.notna(dcz_ratio):
                    output[modality_prefix].setdefault(
                        subject,
                        [],
                    ).append(
                        {
                            "pair_id": pair_id,
                            "value": float(saline_ratio - dcz_ratio),
                        }
                    )
                continue

            if pd.notna(saline_ratio):
                output[f"{modality_prefix}_saline"].setdefault(
                    subject,
                    [],
                ).append({"pair_id": pair_id, "value": float(saline_ratio)})
            if pd.notna(dcz_ratio):
                output[f"{modality_prefix}_dcz"].setdefault(
                    subject,
                    [],
                ).append({"pair_id": pair_id, "value": float(dcz_ratio)})

        return output

    return (get_roi_per_mouse_sessions,)


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    get_roi_per_mouse_sessions,
    hM3Dq_mice,
    hM4Di_mice,
    roi_bottom,
    roi_left,
    roi_right,
    roi_top,
):
    roi_per_animal_condition_hm4 = get_roi_per_mouse_sessions(
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        behav_pair_map=behav_pair_map,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,
        subjects=hM4Di_mice,
    )
    roi_per_animal_condition_hm4_diff = get_roi_per_mouse_sessions(
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        behav_pair_map=behav_pair_map,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,
        subjects=hM4Di_mice,
        difference=True,
    )
    roi_per_animal_condition_hm3 = get_roi_per_mouse_sessions(
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        behav_pair_map=behav_pair_map,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,
        subjects=hM3Dq_mice,
    )
    roi_per_animal_condition_hm3_diff = get_roi_per_mouse_sessions(
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        behav_pair_map=behav_pair_map,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,
        subjects=hM3Dq_mice,
        difference=True,
    )
    return (
        roi_per_animal_condition_hm3,
        roi_per_animal_condition_hm3_diff,
        roi_per_animal_condition_hm4,
        roi_per_animal_condition_hm4_diff,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## condition select
    """)
    return


@app.cell
def _(mo):
    compare_value_settings = {
        "engagement": {
            "split_col": "engaged",
            "split_label": ["engaged", "disengaged"]
        },
        "previous correct": {
            "split_col": "previous_correct",
            "split_label": ["previous correct", "previous incorrect"]
        },
    }
    compare_value_select = mo.ui.dropdown(
        options=list(compare_value_settings),
        value="engagement",
        label="Compare value",
    )
    compare_value_select
    return compare_value_select, compare_value_settings


@app.cell
def _(compare_value_select, compare_value_settings):
    _compare_value_setting = compare_value_settings[compare_value_select.value]
    compare_value_name = compare_value_select.value
    split_col = _compare_value_setting["split_col"]
    split_label = _compare_value_setting["split_label"]
    return compare_value_name, split_col, split_label


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    df_dic_dcz,
    df_dic_saline,
    get_roi_per_mouse_sessions,
    hM3Dq_mice,
    hM4Di_mice,
    roi_bottom,
    roi_left,
    roi_right,
    roi_top,
    split_col,
    split_label,
):
    roi_per_animal_condition_hm4_split_col = get_roi_per_mouse_sessions(
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        df_dic_saline=df_dic_saline,
        df_dic_dcz=df_dic_dcz,
        behav_pair_map=behav_pair_map,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,
        subjects=hM4Di_mice,
        split_column=split_col,
        split_labels=split_label,
    )
    roi_per_animal_condition_hm4_diff_split_col = get_roi_per_mouse_sessions(
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        df_dic_saline=df_dic_saline,
        df_dic_dcz=df_dic_dcz,
        behav_pair_map=behav_pair_map,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,
        subjects=hM4Di_mice,
        split_column=split_col,
        split_labels=split_label,
        difference=True,
    )
    roi_per_animal_condition_hm3_split_col = get_roi_per_mouse_sessions(
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        df_dic_saline=df_dic_saline,
        df_dic_dcz=df_dic_dcz,
        behav_pair_map=behav_pair_map,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,
        subjects=hM3Dq_mice,
        split_column=split_col,
        split_labels=split_label,
    )
    roi_per_animal_condition_hm3_diff_split_col = get_roi_per_mouse_sessions(
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        df_dic_saline=df_dic_saline,
        df_dic_dcz=df_dic_dcz,
        behav_pair_map=behav_pair_map,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,
        subjects=hM3Dq_mice,
        split_column=split_col,
        split_labels=split_label,
        difference=True,
    )
    return (
        roi_per_animal_condition_hm3_diff_split_col,
        roi_per_animal_condition_hm3_split_col,
        roi_per_animal_condition_hm4_diff_split_col,
        roi_per_animal_condition_hm4_split_col,
    )


@app.cell
def _(
    mo,
    plot_test,
    roi_per_animal_condition_hm3,
    roi_per_animal_condition_hm3_diff,
    roi_per_animal_condition_hm4,
    roi_per_animal_condition_hm4_diff,
):
    hm4_roi_condition_session_fig = (
        plot_test.plot_condition_session_values_by_mouse(
            roi_per_animal_condition_hm4,
            "hM4Di",
            ylabel="Session ROI time ratio",
            title="hM4Di: ROI time ratio by condition",
        )
    )
    hm3_roi_condition_session_fig = (
        plot_test.plot_condition_session_values_by_mouse(
            roi_per_animal_condition_hm3,
            "hM3Dq",
            ylabel="Session ROI time ratio",
            title="hM3Dq: ROI time ratio by condition",
        )
    )
    roi_condition_diff_session_fig = (
        plot_test.plot_group_condition_values_by_mouse(
            roi_per_animal_condition_hm3_diff,
            roi_per_animal_condition_hm4_diff,
            ylabel="saline - DCZ (session ROI time ratio)",
            title="ROI time ratio saline - DCZ",
        )
    )

    mo.vstack(
        [
            hm4_roi_condition_session_fig,
            hm3_roi_condition_session_fig,
            roi_condition_diff_session_fig,
        ]
    )
    return


@app.cell
def _(
    compare_value_name,
    mo,
    plot_test,
    roi_per_animal_condition_hm3_diff_split_col,
    roi_per_animal_condition_hm3_split_col,
    roi_per_animal_condition_hm4_diff_split_col,
    roi_per_animal_condition_hm4_split_col,
    split_col,
    split_label,
):
    hm4_roi_engagement_condition_session_fig = (
        plot_test.plot_condition_session_values_by_mouse(
            roi_per_animal_condition_hm4_split_col,
            "hM4Di",
            ylabel="Session ROI time ratio",
            title=(
                f"hM4Di: ROI time ratio by condition and "
                f"{compare_value_name}"
            ),
            split_column=split_col,
            split_labels=split_label,
        )
    )
    hm3_roi_engagement_condition_session_fig = (
        plot_test.plot_condition_session_values_by_mouse(
            roi_per_animal_condition_hm3_split_col,
            "hM3Dq",
            ylabel="Session ROI time ratio",
            title=(
                f"hM3Dq: ROI time ratio by condition and "
                f"{compare_value_name}"
            ),
            split_column=split_col,
            split_labels=split_label,
        )
    )
    roi_engagement_condition_diff_session_fig = (
        plot_test.plot_group_condition_values_by_mouse(
            roi_per_animal_condition_hm3_diff_split_col,
            roi_per_animal_condition_hm4_diff_split_col,
            ylabel="saline - DCZ (session ROI time ratio)",
            title=f"ROI time ratio saline - DCZ by {compare_value_name}",
            split_column=split_col,
            split_labels=split_label,
        )
    )
    mo.vstack(
        [
            hm4_roi_engagement_condition_session_fig,
            hm3_roi_engagement_condition_session_fig,
            roi_engagement_condition_diff_session_fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### combine the subjects
    """)
    return


@app.cell
def _(
    compare_value_name,
    mo,
    np,
    pd,
    plt,
    roi_per_animal_condition_hm3_diff_split_col,
    roi_per_animal_condition_hm4_diff_split_col,
    split_label,
):
    from scipy import stats as _stats

    _split_labels = list(split_label)

    def _p_value_to_label(_p_value):
        if pd.isna(_p_value):
            return "n/a"
        if _p_value < 0.001:
            return "***"
        if _p_value < 0.01:
            return "**"
        if _p_value < 0.05:
            return "*"
        return "ns"

    def _add_significance_bar(_ax, _x1, _x2, _y, _h, _label):
        _ax.plot(
            [_x1, _x1, _x2, _x2],
            [_y, _y + _h, _y + _h, _y],
            color="black",
            linewidth=1,
        )
        _ax.text(
            (_x1 + _x2) / 2,
            _y + _h,
            _label,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    def _combined_diff_rows(_output_dic, _group_name):
        _rows = []

        def _entry_to_pair_value(_entry, _fallback_idx):
            if isinstance(_entry, dict):
                return (
                    _entry.get("pair_id", f"row_{_fallback_idx}"),
                    _entry.get("value", np.nan),
                )
            return f"row_{_fallback_idx}", _entry

        for _modality in ["vis", "aud"]:
            for _subject, _split_values in (
                _output_dic.get(_modality, {}).items()
            ):
                if not isinstance(_split_values, dict):
                    continue
                if any(
                    _label not in _split_values
                    for _label in _split_labels
                ):
                    continue

                for _label in _split_labels:
                    for _entry_idx, _entry in enumerate(
                        _split_values[_label]
                    ):
                        _pair_id, _value = _entry_to_pair_value(
                            _entry,
                            _entry_idx,
                        )
                        _value = pd.to_numeric(
                            pd.Series([_value]),
                            errors="coerce",
                        ).iloc[0]
                        if pd.isna(_value):
                            continue
                        _rows.append(
                            {
                                "group": _group_name,
                                "subject": _subject,
                                "modality": _modality,
                                "pair_id": _pair_id,
                                "split_label": _label,
                                "roi_diff": float(_value),
                            }
                        )
        return _rows

    combined_roi_diff_split_df = pd.DataFrame(
        _combined_diff_rows(
            roi_per_animal_condition_hm3_diff_split_col,
            "hM3Dq",
        )
        + _combined_diff_rows(
            roi_per_animal_condition_hm4_diff_split_col,
            "hM4Di",
        )
    )

    def _plot_combined_group(_df, _group_name):
        _fig, _ax = plt.subplots(figsize=(5, 5), dpi=150)
        _df_group = _df[_df["group"] == _group_name].copy()

        if _df_group.empty:
            _ax.text(
                0.5,
                0.5,
                "No paired values to plot.",
                ha="center",
                va="center",
                transform=_ax.transAxes,
            )
            _ax.set_axis_off()
            _fig.tight_layout()
            return _fig

        _data = [
            _df_group.loc[
                _df_group["split_label"] == _label,
                "roi_diff",
            ].dropna()
            for _label in _split_labels
        ]
        _colors = ["#4C78A8", "#F58518"]
        _x_positions = np.arange(1, len(_split_labels) + 1)

        _boxplot = _ax.boxplot(
            _data,
            labels=_split_labels,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
        )
        for _patch, _color in zip(_boxplot["boxes"], _colors):
            _patch.set_facecolor(_color)
            _patch.set_alpha(0.22)
            _patch.set_edgecolor(_color)

        _paired_df = _df_group.pivot_table(
            index=["subject", "modality", "pair_id"],
            columns="split_label",
            values="roi_diff",
            aggfunc="mean",
        )
        _paired_df = _paired_df.dropna(subset=_split_labels)

        for _, _row in _paired_df.iterrows():
            _ax.plot(
                _x_positions,
                [_row[_label] for _label in _split_labels],
                color="gray",
                alpha=0.3,
                linewidth=1,
                zorder=2,
            )

        for _idx, (_label, _color) in enumerate(
            zip(_split_labels, _colors),
            start=1,
        ):
            _values = _df_group.loc[
                _df_group["split_label"] == _label,
                "roi_diff",
            ].dropna().to_numpy(dtype=float)
            if len(_values) == 0:
                continue
            _jitter = np.linspace(-0.06, 0.06, len(_values))
            _ax.scatter(
                _idx + _jitter,
                _values,
                color=_color,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.75,
                s=35,
                zorder=3,
            )

        if len(_split_labels) == 2 and len(_paired_df) >= 2:
            try:
                _p_value = _stats.ttest_rel(
                    _paired_df[_split_labels[0]],
                    _paired_df[_split_labels[1]],
                    nan_policy="omit",
                ).pvalue
            except ValueError:
                _p_value = np.nan

            _y_values = pd.to_numeric(
                _df_group["roi_diff"],
                errors="coerce",
            ).dropna()
            if not _y_values.empty:
                _y_min = _y_values.min()
                _y_max = _y_values.max()
                _y_range = _y_max - _y_min
                if _y_range == 0:
                    _y_range = abs(_y_max) * 0.1 if _y_max != 0 else 1
                _bar_y = _y_max + _y_range * 0.12
                _bar_h = _y_range * 0.06
                _add_significance_bar(
                    _ax,
                    _x_positions[0],
                    _x_positions[1],
                    _bar_y,
                    _bar_h,
                    _p_value_to_label(_p_value),
                )
                _ax.set_ylim(top=_bar_y + _bar_h + _y_range * 0.15)

        _ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        _ax.set_xlabel(compare_value_name)
        _ax.set_ylabel("saline - DCZ (ROI time ratio)")
        _ax.set_title(f"{_group_name}: combined aud/vis subjects")
        _ax.grid(True, axis="y", alpha=0.3)
        _fig.tight_layout()
        return _fig

    combined_roi_diff_split_figures = [
        _plot_combined_group(combined_roi_diff_split_df, "hM3Dq"),
        _plot_combined_group(combined_roi_diff_split_df, "hM4Di"),
    ]
    _plot_combined_group(combined_roi_diff_split_df, "hM3Dq").savefig('/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/hm3_bypreviouschoice_roitime.svg')
    _plot_combined_group(combined_roi_diff_split_df, "hM4Di").savefig('/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/hm4_bypreviouschoice_roitime.svg')
    mo.hstack(combined_roi_diff_split_figures)
    return


@app.cell
def _(
    mo,
    np,
    pd,
    plt,
    roi_per_animal_condition_hm3,
    roi_per_animal_condition_hm4,
):
    from scipy import stats as _stats

    def _p_value_to_label(_p_value):
        if pd.isna(_p_value):
            return "n/a"
        if _p_value < 0.001:
            return "***"
        if _p_value < 0.01:
            return "**"
        if _p_value < 0.05:
            return "*"
        return "ns"

    def _add_significance_bar(_ax, _x1, _x2, _y, _h, _label):
        _ax.plot(
            [_x1, _x1, _x2, _x2],
            [_y, _y + _h, _y + _h, _y],
            color="black",
            linewidth=1,
        )
        _ax.text(
            (_x1 + _x2) / 2,
            _y + _h,
            _label,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    def plot_combined_saline_dcz_by_group(
        hm3_condition_dic,
        hm4_condition_dic,
        metric_col,
        ylabel,
        title_prefix,
    ):
        def _entry_to_pair_value(_entry, _fallback_idx):
            if isinstance(_entry, dict):
                return (
                    _entry.get("pair_id", f"row_{_fallback_idx}"),
                    _entry.get("value", np.nan),
                )
            return f"row_{_fallback_idx}", _entry

        def _condition_rows(_condition_dic, _group_name):
            _rows = []
            for _modality in ["vis", "aud"]:
                for _condition_name in ["saline", "dcz"]:
                    _condition_key = f"{_modality}_{_condition_name}"
                    for _subject, _values in (
                        _condition_dic.get(_condition_key, {}).items()
                    ):
                        for _entry_idx, _entry in enumerate(_values):
                            _pair_id, _value = _entry_to_pair_value(
                                _entry,
                                _entry_idx,
                            )
                            _value = pd.to_numeric(
                                pd.Series([_value]),
                                errors="coerce",
                            ).iloc[0]
                            if pd.isna(_value):
                                continue
                            _rows.append(
                                {
                                    "group": _group_name,
                                    "subject": _subject,
                                    "modality": _modality,
                                    "pair_id": _pair_id,
                                    "condition": _condition_name,
                                    metric_col: float(_value),
                                }
                            )
            return _rows

        combined_df = pd.DataFrame(
            _condition_rows(hm3_condition_dic, "hM3Dq")
            + _condition_rows(hm4_condition_dic, "hM4Di")
        )

        def _plot_group(_group_name):
            _fig, _ax = plt.subplots(figsize=(5, 5), dpi=150)
            _df_group = combined_df[
                combined_df["group"] == _group_name
            ].copy()

            if _df_group.empty:
                _ax.text(
                    0.5,
                    0.5,
                    "No paired values to plot.",
                    ha="center",
                    va="center",
                    transform=_ax.transAxes,
                )
                _ax.set_axis_off()
                _fig.tight_layout()
                return _fig

            _conditions = ["saline", "dcz"]
            _colors = {"saline": "blue", "dcz": "red"}
            _data = [
                _df_group.loc[
                    _df_group["condition"] == _condition_name,
                    metric_col,
                ].dropna()
                for _condition_name in _conditions
            ]
            _x_positions = np.arange(1, len(_conditions) + 1)

            _boxplot = _ax.boxplot(
                _data,
                labels=["saline", "DCZ"],
                patch_artist=True,
                showmeans=True,
                showfliers=False,
            )
            for _patch, _condition_name in zip(
                _boxplot["boxes"],
                _conditions,
            ):
                _patch.set_facecolor(_colors[_condition_name])
                _patch.set_alpha(0.20)
                _patch.set_edgecolor(_colors[_condition_name])

            _paired_df = _df_group.pivot_table(
                index=["subject", "modality", "pair_id"],
                columns="condition",
                values=metric_col,
                aggfunc="mean",
            ).dropna(subset=_conditions)

            for _, _row in _paired_df.iterrows():
                _ax.plot(
                    _x_positions,
                    [_row[_condition] for _condition in _conditions],
                    color="gray",
                    alpha=0.35,
                    linewidth=1,
                    zorder=2,
                )

            for _idx, _condition_name in enumerate(_conditions, start=1):
                _values = _df_group.loc[
                    _df_group["condition"] == _condition_name,
                    metric_col,
                ].dropna().to_numpy(dtype=float)
                if len(_values) == 0:
                    continue
                _jitter = np.linspace(-0.06, 0.06, len(_values))
                _ax.scatter(
                    _idx + _jitter,
                    _values,
                    color=_colors[_condition_name],
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.75,
                    s=35,
                    zorder=3,
                )

            if len(_paired_df) >= 1:
                try:
                    _p_value = _stats.wilcoxon(
                        _paired_df["saline"],
                        _paired_df["dcz"],
                    ).pvalue
                except ValueError:
                    _p_value = np.nan

                _y_values = pd.to_numeric(
                    _df_group[metric_col],
                    errors="coerce",
                ).dropna()
                if not _y_values.empty:
                    _y_min = _y_values.min()
                    _y_max = _y_values.max()
                    _y_range = _y_max - _y_min
                    if _y_range == 0:
                        _y_range = abs(_y_max) * 0.1 if _y_max != 0 else 1
                    _bar_y = _y_max + _y_range * 0.12
                    _bar_h = _y_range * 0.06
                    _add_significance_bar(
                        _ax,
                        _x_positions[0],
                        _x_positions[1],
                        _bar_y,
                        _bar_h,
                        _p_value_to_label(_p_value),
                    )
                    _ax.set_ylim(top=_bar_y + _bar_h + _y_range * 0.15)

            _ax.set_xlabel("Observation")
            _ax.set_ylabel(ylabel)
            _ax.set_title(f"{title_prefix} in {_group_name}")
            _ax.grid(True, axis="y", alpha=0.3)
            _fig.tight_layout()
            return _fig

        return combined_df, [
            _plot_group("hM3Dq"),
            _plot_group("hM4Di"),
        ]

    combined_roi_condition_df, combined_roi_condition_figures = (
        plot_combined_saline_dcz_by_group(
            roi_per_animal_condition_hm3,
            roi_per_animal_condition_hm4,
            metric_col="roi_time_ratio",
            ylabel="ROI time ratio",
            title_prefix="ROI time ratio",
        )
    )
    combined_roi_condition_figures[0].savefig("/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/hm3_roi_time_byobserve_fig.svg")
    combined_roi_condition_figures[1].savefig("/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/hm4_roi_time_byobserve_fig.svg")
    mo.hstack(combined_roi_condition_figures)
    return (plot_combined_saline_dcz_by_group,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## speed
    """)
    return


@app.cell
def _(behavior_utils, np, pd):
    def _empty_speed_output(difference=False):
        condition_keys = (
            ["vis", "aud"]
            if difference
            else ["vis_saline", "vis_dcz", "aud_saline", "aud_dcz"]
        )
        return {condition_key: {} for condition_key in condition_keys}

    def _mouse_speed_split_output(condition_dict, subject, split_labels):
        return condition_dict.setdefault(
            subject,
            {split_label: [] for split_label in split_labels},
        )

    def get_mean_speed(behav_df, bodypart="Center", speed_col="mean_speed"):
        speed = pd.to_numeric(
            behavior_utils.get_behavior_column(
                behav_df,
                (bodypart, speed_col),
            ),
            errors="coerce",
        ).to_numpy(dtype=float)
        speed = speed[np.isfinite(speed)]
        if len(speed) == 0:
            return np.nan
        return float(np.nanmean(speed))

    def get_trial_speed_by_split_col(
        behav_df,
        trial_df,
        split_column="engaged",
        split_labels=("engaged", "disengaged"),
        bodypart="Center",
        speed_col="mean_speed",
    ):
        true_label, false_label = split_labels
        speed_values_by_split = {true_label: [], false_label: []}
        speed_time_by_split = {true_label: 0.0, false_label: 0.0}
        time_by_split = {true_label: 0.0, false_label: 0.0}

        timestamp = pd.to_numeric(
            behavior_utils.get_behavior_column(
                behav_df,
                ("timestamp", ""),
            ),
            errors="coerce",
        )
        speed = pd.to_numeric(
            behavior_utils.get_behavior_column(
                behav_df,
                (bodypart, speed_col),
            ),
            errors="coerce",
        )

        frame_df = (
            pd.DataFrame(
                {
                    "timestamp": timestamp.to_numpy(),
                    "speed": speed.to_numpy(),
                }
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["timestamp"])
            .sort_values("timestamp")
        )

        t = frame_df["timestamp"].to_numpy(dtype=float)
        speed_values = frame_df["speed"].to_numpy(dtype=float)
        dt = np.diff(t)
        positive_dt = dt[np.isfinite(dt) & (dt > 0)]
        median_dt = float(np.median(positive_dt)) if len(positive_dt) else 0.0

        frame_end = np.empty_like(t)
        if len(t) > 1:
            frame_end[:-1] = t[1:]
        frame_end[-1] = t[-1] + median_dt

        valid_speed = np.isfinite(speed_values)

        for _, trial_row in trial_df.iterrows():
            split_label = true_label if bool(trial_row[split_column]) else false_label
            trial_start = pd.to_numeric(
                trial_row["TRIAL_START"],
                errors="coerce",
            )
            trial_end = pd.to_numeric(
                trial_row["TRIAL_END"],
                errors="coerce",
            )
            if (
                not np.isfinite(trial_start)
                or not np.isfinite(trial_end)
                or trial_end <= trial_start
            ):
                continue

            overlap_start = np.maximum(t, float(trial_start))
            overlap_end = np.minimum(frame_end, float(trial_end))
            overlap = np.clip(overlap_end - overlap_start, 0, None)
            overlap = np.where(valid_speed, overlap, 0.0)

            speed_time_by_split[split_label] += float(
                np.sum(speed_values[valid_speed] * overlap[valid_speed])
            )
            time_by_split[split_label] += float(np.sum(overlap[valid_speed]))

        for split_label in split_labels:
            if time_by_split[split_label] > 0:
                speed_values_by_split[split_label].append(
                    speed_time_by_split[split_label]
                    / time_by_split[split_label]
                )

        return speed_values_by_split

    def get_trial_speed_per_mouse_sessions(
        behav_df_dic_saline,
        behav_df_dic_dcz,
        df_dic_saline,
        df_dic_dcz,
        behav_pair_map,
        subjects=None,
        bodypart="Center",
        speed_col="mean_speed",
        split_column="engaged",
        split_labels=("engaged", "disengaged"),
        difference=False,
    ):
        output = _empty_speed_output(difference=difference)

        for _, pair_row in behav_pair_map.sort_values("pair_id").iterrows():
            pair_id = pair_row["pair_id"]
            subject = str(pair_row["subject"])
            if subjects is not None and subject not in subjects:
                continue
            if (
                pair_id not in behav_df_dic_saline
                or pair_id not in behav_df_dic_dcz
                or pair_id not in df_dic_saline
                or pair_id not in df_dic_dcz
            ):
                continue

            modality = str(pair_row.get("stimulus_modality", "")).lower()
            has_visual = "visual" in modality
            has_auditory = "auditory" in modality
            if has_visual == has_auditory:
                continue
            modality_prefix = "vis" if has_visual else "aud"

            saline_speed = get_trial_speed_by_split_col(
                behav_df_dic_saline[pair_id],
                df_dic_saline[pair_id],
                split_column=split_column,
                split_labels=split_labels,
                bodypart=bodypart,
                speed_col=speed_col,
            )
            dcz_speed = get_trial_speed_by_split_col(
                behav_df_dic_dcz[pair_id],
                df_dic_dcz[pair_id],
                split_column=split_column,
                split_labels=split_labels,
                bodypart=bodypart,
                speed_col=speed_col,
            )

            if difference:
                mouse_output = _mouse_speed_split_output(
                    output[modality_prefix],
                    subject,
                    split_labels,
                )
                for split_label in split_labels:
                    saline_values = np.asarray(
                        saline_speed[split_label],
                        dtype=float,
                    )
                    dcz_values = np.asarray(
                        dcz_speed[split_label],
                        dtype=float,
                    )
                    if len(saline_values) == 0 or len(dcz_values) == 0:
                        continue
                    mouse_output[split_label].append(
                        {
                            "pair_id": pair_id,
                            "value": float(
                                np.nanmean(saline_values)
                                - np.nanmean(dcz_values)
                            ),
                        }
                    )
                continue

            saline_output = _mouse_speed_split_output(
                output[f"{modality_prefix}_saline"],
                subject,
                split_labels,
            )
            dcz_output = _mouse_speed_split_output(
                output[f"{modality_prefix}_dcz"],
                subject,
                split_labels,
            )
            for split_label in split_labels:
                saline_output[split_label].extend(saline_speed[split_label])
                dcz_output[split_label].extend(dcz_speed[split_label])

        return output

    def get_speed_per_mouse_sessions(
        behav_df_dic_saline,
        behav_df_dic_dcz,
        behav_pair_map,
        df_dic_saline=None,
        df_dic_dcz=None,
        subjects=None,
        bodypart="Center",
        speed_col="mean_speed",
        split_column=None,
        split_labels=("engaged", "disengaged"),
        difference=False,
    ):
        if split_column is not None:
            return get_trial_speed_per_mouse_sessions(
                behav_df_dic_saline=behav_df_dic_saline,
                behav_df_dic_dcz=behav_df_dic_dcz,
                df_dic_saline=df_dic_saline,
                df_dic_dcz=df_dic_dcz,
                behav_pair_map=behav_pair_map,
                subjects=subjects,
                bodypart=bodypart,
                speed_col=speed_col,
                split_column=split_column,
                split_labels=split_labels,
                difference=difference,
            )

        output = _empty_speed_output(difference=difference)

        for _, pair_row in behav_pair_map.sort_values("pair_id").iterrows():
            pair_id = pair_row["pair_id"]
            subject = str(pair_row["subject"])
            if subjects is not None and subject not in subjects:
                continue
            if (
                pair_id not in behav_df_dic_saline
                or pair_id not in behav_df_dic_dcz
            ):
                continue

            modality = str(pair_row.get("stimulus_modality", "")).lower()
            has_visual = "visual" in modality
            has_auditory = "auditory" in modality
            if has_visual == has_auditory:
                continue
            modality_prefix = "vis" if has_visual else "aud"

            saline_speed = get_mean_speed(
                behav_df_dic_saline[pair_id],
                bodypart=bodypart,
                speed_col=speed_col,
            )
            dcz_speed = get_mean_speed(
                behav_df_dic_dcz[pair_id],
                bodypart=bodypart,
                speed_col=speed_col,
            )

            if difference:
                if pd.notna(saline_speed) and pd.notna(dcz_speed):
                    output[modality_prefix].setdefault(
                        subject,
                        [],
                    ).append(
                        {
                            "pair_id": pair_id,
                            "value": float(saline_speed - dcz_speed),
                        }
                    )
                continue

            if pd.notna(saline_speed):
                output[f"{modality_prefix}_saline"].setdefault(
                    subject,
                    [],
                ).append({"pair_id": pair_id, "value": float(saline_speed)})
            if pd.notna(dcz_speed):
                output[f"{modality_prefix}_dcz"].setdefault(
                    subject,
                    [],
                ).append({"pair_id": pair_id, "value": float(dcz_speed)})

        return output


    return (get_speed_per_mouse_sessions,)


@app.cell
def _(
    mo,
    plot_test,
    speed_per_animal_condition_hm3,
    speed_per_animal_condition_hm3_diff,
    speed_per_animal_condition_hm4,
    speed_per_animal_condition_hm4_diff,
):
    hm4_speed_condition_session_fig = (
        plot_test.plot_condition_session_values_by_mouse(
            speed_per_animal_condition_hm4,
            "hM4Di",
            ylabel="Session mean speed",
            title="hM4Di: mean speed by condition",
        )
    )
    hm3_speed_condition_session_fig = (
        plot_test.plot_condition_session_values_by_mouse(
            speed_per_animal_condition_hm3,
            "hM3Dq",
            ylabel="Session mean speed",
            title="hM3Dq: mean speed by condition",
        )
    )
    speed_condition_diff_session_fig = (
        plot_test.plot_group_condition_values_by_mouse(
            speed_per_animal_condition_hm3_diff,
            speed_per_animal_condition_hm4_diff,
            ylabel="saline - DCZ (session mean speed)",
            title="mean speed saline - DCZ",
        )
    )

    mo.vstack(
        [
            hm4_speed_condition_session_fig,
            hm3_speed_condition_session_fig,
            speed_condition_diff_session_fig,
        ]
    )
    return


@app.cell
def _(
    compare_value_name,
    mo,
    plot_test,
    speed_per_animal_condition_hm3_diff_split_col,
    speed_per_animal_condition_hm3_split_col,
    speed_per_animal_condition_hm4_diff_split_col,
    speed_per_animal_condition_hm4_split_col,
    split_col,
    split_label,
):
    hm4_speed_engagement_condition_session_fig = (
        plot_test.plot_condition_session_values_by_mouse(
            speed_per_animal_condition_hm4_split_col,
            "hM4Di",
            ylabel="Session mean speed",
            title=(
                f"hM4Di: mean speed by condition and "
                f"{compare_value_name}"
            ),
            split_column=split_col,
            split_labels=split_label,
        )
    )
    hm3_speed_engagement_condition_session_fig = (
        plot_test.plot_condition_session_values_by_mouse(
            speed_per_animal_condition_hm3_split_col,
            "hM3Dq",
            ylabel="Session mean speed",
            title=(
                f"hM3Dq: mean speed by condition and "
                f"{compare_value_name}"
            ),
            split_column=split_col,
            split_labels=split_label,
        )
    )
    speed_engagement_condition_diff_session_fig = (
        plot_test.plot_group_condition_values_by_mouse(
            speed_per_animal_condition_hm3_diff_split_col,
            speed_per_animal_condition_hm4_diff_split_col,
            ylabel="saline - DCZ (session mean speed)",
            title=f"mean speed saline - DCZ by {compare_value_name}",
            split_column=split_col,
            split_labels=split_label,
        )
    )
    mo.vstack(
        [
            hm4_speed_engagement_condition_session_fig,
            hm3_speed_engagement_condition_session_fig,
            speed_engagement_condition_diff_session_fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### combine the subjects
    """)
    return


@app.cell
def _(
    compare_value_name,
    mo,
    np,
    pd,
    plt,
    speed_per_animal_condition_hm3_diff_split_col,
    speed_per_animal_condition_hm4_diff_split_col,
    split_label,
):
    from scipy import stats as _stats

    _split_labels = list(split_label)

    def _p_value_to_label(_p_value):
        if pd.isna(_p_value):
            return "n/a"
        if _p_value < 0.001:
            return "***"
        if _p_value < 0.01:
            return "**"
        if _p_value < 0.05:
            return "*"
        return "ns"

    def _add_significance_bar(_ax, _x1, _x2, _y, _h, _label):
        _ax.plot(
            [_x1, _x1, _x2, _x2],
            [_y, _y + _h, _y + _h, _y],
            color="black",
            linewidth=1,
        )
        _ax.text(
            (_x1 + _x2) / 2,
            _y + _h,
            _label,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    def _combined_speed_diff_rows(_output_dic, _group_name):
        _rows = []

        def _entry_to_pair_value(_entry, _fallback_idx):
            if isinstance(_entry, dict):
                return (
                    _entry.get("pair_id", f"row_{_fallback_idx}"),
                    _entry.get("value", np.nan),
                )
            return f"row_{_fallback_idx}", _entry

        for _modality in ["vis", "aud"]:
            for _subject, _split_values in (
                _output_dic.get(_modality, {}).items()
            ):
                if not isinstance(_split_values, dict):
                    continue
                if any(
                    _label not in _split_values
                    for _label in _split_labels
                ):
                    continue

                for _label in _split_labels:
                    for _entry_idx, _entry in enumerate(
                        _split_values[_label]
                    ):
                        _pair_id, _value = _entry_to_pair_value(
                            _entry,
                            _entry_idx,
                        )
                        _value = pd.to_numeric(
                            pd.Series([_value]),
                            errors="coerce",
                        ).iloc[0]
                        if pd.isna(_value):
                            continue
                        _rows.append(
                            {
                                "group": _group_name,
                                "subject": _subject,
                                "modality": _modality,
                                "pair_id": _pair_id,
                                "split_label": _label,
                                "speed_diff": float(_value),
                            }
                        )
        return _rows

    combined_speed_diff_split_df = pd.DataFrame(
        _combined_speed_diff_rows(
            speed_per_animal_condition_hm3_diff_split_col,
            "hM3Dq",
        )
        + _combined_speed_diff_rows(
            speed_per_animal_condition_hm4_diff_split_col,
            "hM4Di",
        )
    )

    def _plot_combined_speed_group(_df, _group_name):
        _fig, _ax = plt.subplots(figsize=(5, 5), dpi=150)
        _df_group = _df[_df["group"] == _group_name].copy()

        if _df_group.empty:
            _ax.text(
                0.5,
                0.5,
                "No paired values to plot.",
                ha="center",
                va="center",
                transform=_ax.transAxes,
            )
            _ax.set_axis_off()
            _fig.tight_layout()
            return _fig

        _data = [
            _df_group.loc[
                _df_group["split_label"] == _label,
                "speed_diff",
            ].dropna()
            for _label in _split_labels
        ]
        _colors = ["#4C78A8", "#F58518"]
        _x_positions = np.arange(1, len(_split_labels) + 1)

        _boxplot = _ax.boxplot(
            _data,
            labels=_split_labels,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
        )
        for _patch, _color in zip(_boxplot["boxes"], _colors):
            _patch.set_facecolor(_color)
            _patch.set_alpha(0.22)
            _patch.set_edgecolor(_color)

        _paired_df = _df_group.pivot_table(
            index=["subject", "modality", "pair_id"],
            columns="split_label",
            values="speed_diff",
            aggfunc="mean",
        )
        _paired_df = _paired_df.dropna(subset=_split_labels)

        for _, _row in _paired_df.iterrows():
            _ax.plot(
                _x_positions,
                [_row[_label] for _label in _split_labels],
                color="gray",
                alpha=0.3,
                linewidth=1,
                zorder=2,
            )

        for _idx, (_label, _color) in enumerate(
            zip(_split_labels, _colors),
            start=1,
        ):
            _values = _df_group.loc[
                _df_group["split_label"] == _label,
                "speed_diff",
            ].dropna().to_numpy(dtype=float)
            if len(_values) == 0:
                continue
            _jitter = np.linspace(-0.06, 0.06, len(_values))
            _ax.scatter(
                _idx + _jitter,
                _values,
                color=_color,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.75,
                s=35,
                zorder=3,
            )

        if len(_split_labels) == 2 and len(_paired_df) >= 2:
            try:
                _p_value = _stats.ttest_rel(
                    _paired_df[_split_labels[0]],
                    _paired_df[_split_labels[1]],
                    nan_policy="omit",
                ).pvalue
            except ValueError:
                _p_value = np.nan

            _y_values = pd.to_numeric(
                _df_group["speed_diff"],
                errors="coerce",
            ).dropna()
            if not _y_values.empty:
                _y_min = _y_values.min()
                _y_max = _y_values.max()
                _y_range = _y_max - _y_min
                if _y_range == 0:
                    _y_range = abs(_y_max) * 0.1 if _y_max != 0 else 1
                _bar_y = _y_max + _y_range * 0.12
                _bar_h = _y_range * 0.06
                _add_significance_bar(
                    _ax,
                    _x_positions[0],
                    _x_positions[1],
                    _bar_y,
                    _bar_h,
                    _p_value_to_label(_p_value),
                )
                _ax.set_ylim(top=_bar_y + _bar_h + _y_range * 0.15)

        _ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        _ax.set_xlabel(compare_value_name)
        _ax.set_ylabel("saline - DCZ (mean speed)")
        _ax.set_title(f"{_group_name}: combined aud/vis subjects")
        _ax.grid(True, axis="y", alpha=0.3)
        _fig.tight_layout()
        return _fig

    combined_speed_diff_split_figures = [
        _plot_combined_speed_group(combined_speed_diff_split_df, "hM3Dq"),
        _plot_combined_speed_group(combined_speed_diff_split_df, "hM4Di"),
    ]

    mo.hstack(combined_speed_diff_split_figures)
    return


@app.cell
def _(
    mo,
    plot_combined_saline_dcz_by_group,
    speed_per_animal_condition_hm3,
    speed_per_animal_condition_hm4,
):
    combined_speed_condition_df, combined_speed_condition_figures = (
        plot_combined_saline_dcz_by_group(
            speed_per_animal_condition_hm3,
            speed_per_animal_condition_hm4,
            metric_col="mean_speed",
            ylabel="Mean speed",
            title_prefix="Mean speed",
        )
    )

    mo.hstack(combined_speed_condition_figures)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## stationary time ratio
    """)
    return


@app.cell
def _(behavior_utils, np, pd):
    def _empty_stationary_time_ratio_output(difference=False):
        condition_keys = (
            ["vis", "aud"]
            if difference
            else ["vis_saline", "vis_dcz", "aud_saline", "aud_dcz"]
        )
        return {condition_key: {} for condition_key in condition_keys}

    def _mouse_stationary_split_output(
        condition_dict,
        subject,
        split_labels,
    ):
        return condition_dict.setdefault(
            subject,
            {split_label: [] for split_label in split_labels},
        )

    def _stationary_frame_intervals(
        behav_df,
        bodypart="Center",
        speed_col="mean_speed",
    ):
        timestamp = pd.to_numeric(
            behavior_utils.get_behavior_column(
                behav_df,
                ("timestamp", ""),
            ),
            errors="coerce",
        )
        speed = pd.to_numeric(
            behavior_utils.get_behavior_column(
                behav_df,
                (bodypart, speed_col),
            ),
            errors="coerce",
        )
        frame_df = (
            pd.DataFrame(
                {
                    "timestamp": timestamp.to_numpy(),
                    "speed": speed.to_numpy(),
                }
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["timestamp"])
            .sort_values("timestamp")
        )
        if frame_df.empty:
            return None, None, None

        t = frame_df["timestamp"].to_numpy(dtype=float)
        speed_values = frame_df["speed"].to_numpy(dtype=float)
        dt = np.diff(t)
        positive_dt = dt[np.isfinite(dt) & (dt > 0)]
        median_dt = (
            float(np.median(positive_dt))
            if len(positive_dt)
            else 0.0
        )
        frame_end = np.empty_like(t)
        if len(t) > 1:
            frame_end[:-1] = t[1:]
        frame_end[-1] = t[-1] + median_dt
        return t, frame_end, speed_values

    def get_stationary_time_ratio_for_trials(
        behav_df,
        trial_df,
        bodypart="Center",
        speed_col="mean_speed",
        stationary_speed_threshold=10,
    ):
        if behav_df.empty or trial_df.empty:
            return np.nan

        t, frame_end, speed_values = _stationary_frame_intervals(
            behav_df,
            bodypart=bodypart,
            speed_col=speed_col,
        )
        if t is None:
            return np.nan

        stationary = np.isfinite(speed_values) & (
            speed_values <= stationary_speed_threshold
        )
        stationary_time = 0.0
        trial_time = 0.0

        for _, trial_row in trial_df.iterrows():
            trial_start = pd.to_numeric(
                trial_row["TRIAL_START"],
                errors="coerce",
            )
            trial_end = pd.to_numeric(
                trial_row["TRIAL_END"],
                errors="coerce",
            )
            if (
                not np.isfinite(trial_start)
                or not np.isfinite(trial_end)
                or trial_end <= trial_start
            ):
                continue

            trial_time += float(trial_end - trial_start)
            overlap_start = np.maximum(t, float(trial_start))
            overlap_end = np.minimum(frame_end, float(trial_end))
            overlap = np.clip(overlap_end - overlap_start, 0, None)
            stationary_time += float(overlap[stationary].sum())

        if trial_time <= 0:
            return np.nan
        return float(stationary_time / trial_time)

    def get_trial_stationary_time_ratio_by_split_col(
        behav_df,
        trial_df,
        split_column="engaged",
        split_labels=("engaged", "disengaged"),
        bodypart="Center",
        speed_col="mean_speed",
        stationary_speed_threshold=10,
    ):
        true_label, false_label = split_labels
        ratio_by_split = {true_label: [], false_label: []}
        stationary_time_by_split = {true_label: 0.0, false_label: 0.0}
        trial_time_by_split = {true_label: 0.0, false_label: 0.0}

        if behav_df.empty or trial_df.empty:
            return ratio_by_split

        t, frame_end, speed_values = _stationary_frame_intervals(
            behav_df,
            bodypart=bodypart,
            speed_col=speed_col,
        )
        if t is None:
            return ratio_by_split

        stationary = np.isfinite(speed_values) & (
            speed_values <= stationary_speed_threshold
        )

        for _, trial_row in trial_df.iterrows():
            split_label = (
                true_label
                if bool(trial_row[split_column])
                else false_label
            )
            trial_start = pd.to_numeric(
                trial_row["TRIAL_START"],
                errors="coerce",
            )
            trial_end = pd.to_numeric(
                trial_row["TRIAL_END"],
                errors="coerce",
            )
            if (
                not np.isfinite(trial_start)
                or not np.isfinite(trial_end)
                or trial_end <= trial_start
            ):
                continue

            trial_time_by_split[split_label] += float(
                trial_end - trial_start
            )
            overlap_start = np.maximum(t, float(trial_start))
            overlap_end = np.minimum(frame_end, float(trial_end))
            overlap = np.clip(overlap_end - overlap_start, 0, None)
            stationary_time_by_split[split_label] += float(
                overlap[stationary].sum()
            )

        for split_label in split_labels:
            if trial_time_by_split[split_label] > 0:
                ratio_by_split[split_label].append(
                    stationary_time_by_split[split_label]
                    / trial_time_by_split[split_label]
                )

        return ratio_by_split

    def get_trial_stationary_time_ratio_per_mouse_sessions(
        behav_df_dic_saline,
        behav_df_dic_dcz,
        df_dic_saline,
        df_dic_dcz,
        behav_pair_map,
        subjects=None,
        bodypart="Center",
        speed_col="mean_speed",
        stationary_speed_threshold=10,
        split_column="engaged",
        split_labels=("engaged", "disengaged"),
        difference=False,
    ):
        output = _empty_stationary_time_ratio_output(
            difference=difference
        )

        for _, pair_row in behav_pair_map.sort_values("pair_id").iterrows():
            pair_id = pair_row["pair_id"]
            subject = str(pair_row["subject"])
            if subjects is not None and subject not in subjects:
                continue
            if (
                pair_id not in behav_df_dic_saline
                or pair_id not in behav_df_dic_dcz
                or pair_id not in df_dic_saline
                or pair_id not in df_dic_dcz
            ):
                continue

            modality = str(pair_row.get("stimulus_modality", "")).lower()
            has_visual = "visual" in modality
            has_auditory = "auditory" in modality
            if has_visual == has_auditory:
                continue
            modality_prefix = "vis" if has_visual else "aud"

            saline_ratios = get_trial_stationary_time_ratio_by_split_col(
                behav_df_dic_saline[pair_id],
                df_dic_saline[pair_id],
                split_column=split_column,
                split_labels=split_labels,
                bodypart=bodypart,
                speed_col=speed_col,
                stationary_speed_threshold=stationary_speed_threshold,
            )
            dcz_ratios = get_trial_stationary_time_ratio_by_split_col(
                behav_df_dic_dcz[pair_id],
                df_dic_dcz[pair_id],
                split_column=split_column,
                split_labels=split_labels,
                bodypart=bodypart,
                speed_col=speed_col,
                stationary_speed_threshold=stationary_speed_threshold,
            )

            if difference:
                mouse_output = _mouse_stationary_split_output(
                    output[modality_prefix],
                    subject,
                    split_labels,
                )
                for split_label in split_labels:
                    saline_values = np.asarray(
                        saline_ratios[split_label],
                        dtype=float,
                    )
                    dcz_values = np.asarray(
                        dcz_ratios[split_label],
                        dtype=float,
                    )
                    if len(saline_values) == 0 or len(dcz_values) == 0:
                        continue
                    mouse_output[split_label].append(
                        {
                            "pair_id": pair_id,
                            "value": float(
                                np.nanmean(saline_values)
                                - np.nanmean(dcz_values)
                            ),
                        }
                    )
                continue

            saline_output = _mouse_stationary_split_output(
                output[f"{modality_prefix}_saline"],
                subject,
                split_labels,
            )
            dcz_output = _mouse_stationary_split_output(
                output[f"{modality_prefix}_dcz"],
                subject,
                split_labels,
            )
            for split_label in split_labels:
                saline_output[split_label].extend(
                    saline_ratios[split_label]
                )
                dcz_output[split_label].extend(dcz_ratios[split_label])

        return output

    def get_stationary_time_ratio_per_mouse_sessions(
        behav_df_dic_saline,
        behav_df_dic_dcz,
        df_dic_saline,
        df_dic_dcz,
        behav_pair_map,
        subjects=None,
        bodypart="Center",
        speed_col="mean_speed",
        stationary_speed_threshold=10,
        split_column=None,
        split_labels=("engaged", "disengaged"),
        difference=False,
    ):
        if split_column is not None:
            return get_trial_stationary_time_ratio_per_mouse_sessions(
                behav_df_dic_saline=behav_df_dic_saline,
                behav_df_dic_dcz=behav_df_dic_dcz,
                df_dic_saline=df_dic_saline,
                df_dic_dcz=df_dic_dcz,
                behav_pair_map=behav_pair_map,
                subjects=subjects,
                bodypart=bodypart,
                speed_col=speed_col,
                stationary_speed_threshold=stationary_speed_threshold,
                split_column=split_column,
                split_labels=split_labels,
                difference=difference,
            )

        output = _empty_stationary_time_ratio_output(
            difference=difference
        )
        for _, pair_row in behav_pair_map.sort_values("pair_id").iterrows():
            pair_id = pair_row["pair_id"]
            subject = str(pair_row["subject"])
            if subjects is not None and subject not in subjects:
                continue
            if (
                pair_id not in behav_df_dic_saline
                or pair_id not in behav_df_dic_dcz
                or pair_id not in df_dic_saline
                or pair_id not in df_dic_dcz
            ):
                continue

            modality = str(pair_row.get("stimulus_modality", "")).lower()
            has_visual = "visual" in modality
            has_auditory = "auditory" in modality
            if has_visual == has_auditory:
                continue
            modality_prefix = "vis" if has_visual else "aud"

            saline_ratio = get_stationary_time_ratio_for_trials(
                behav_df_dic_saline[pair_id],
                df_dic_saline[pair_id],
                bodypart=bodypart,
                speed_col=speed_col,
                stationary_speed_threshold=stationary_speed_threshold,
            )
            dcz_ratio = get_stationary_time_ratio_for_trials(
                behav_df_dic_dcz[pair_id],
                df_dic_dcz[pair_id],
                bodypart=bodypart,
                speed_col=speed_col,
                stationary_speed_threshold=stationary_speed_threshold,
            )

            if difference:
                if pd.notna(saline_ratio) and pd.notna(dcz_ratio):
                    output[modality_prefix].setdefault(
                        subject,
                        [],
                    ).append(
                        {
                            "pair_id": pair_id,
                            "value": float(saline_ratio - dcz_ratio),
                        }
                    )
                continue

            if pd.notna(saline_ratio):
                output[f"{modality_prefix}_saline"].setdefault(
                    subject,
                    [],
                ).append({"pair_id": pair_id, "value": float(saline_ratio)})
            if pd.notna(dcz_ratio):
                output[f"{modality_prefix}_dcz"].setdefault(
                    subject,
                    [],
                ).append({"pair_id": pair_id, "value": float(dcz_ratio)})

        return output

    return (get_stationary_time_ratio_per_mouse_sessions,)


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    df_dic_dcz,
    df_dic_saline,
    get_stationary_time_ratio_per_mouse_sessions,
    hM3Dq_mice,
    hM4Di_mice,
    split_col,
    split_label,
):
    stationary_time_ratio_per_animal_condition_hm4 = (
        get_stationary_time_ratio_per_mouse_sessions(
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            df_dic_saline=df_dic_saline,
            df_dic_dcz=df_dic_dcz,
            behav_pair_map=behav_pair_map,
            subjects=hM4Di_mice,
        )
    )
    stationary_time_ratio_per_animal_condition_hm4_diff = (
        get_stationary_time_ratio_per_mouse_sessions(
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            df_dic_saline=df_dic_saline,
            df_dic_dcz=df_dic_dcz,
            behav_pair_map=behav_pair_map,
            subjects=hM4Di_mice,
            difference=True,
        )
    )
    stationary_time_ratio_per_animal_condition_hm3 = (
        get_stationary_time_ratio_per_mouse_sessions(
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            df_dic_saline=df_dic_saline,
            df_dic_dcz=df_dic_dcz,
            behav_pair_map=behav_pair_map,
            subjects=hM3Dq_mice,
        )
    )
    stationary_time_ratio_per_animal_condition_hm3_diff = (
        get_stationary_time_ratio_per_mouse_sessions(
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            df_dic_saline=df_dic_saline,
            df_dic_dcz=df_dic_dcz,
            behav_pair_map=behav_pair_map,
            subjects=hM3Dq_mice,
            difference=True,
        )
    )
    stationary_time_ratio_per_animal_condition_hm4_split_col = (
        get_stationary_time_ratio_per_mouse_sessions(
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            df_dic_saline=df_dic_saline,
            df_dic_dcz=df_dic_dcz,
            behav_pair_map=behav_pair_map,
            subjects=hM4Di_mice,
            split_column=split_col,
            split_labels=split_label,
        )
    )
    stationary_time_ratio_per_animal_condition_hm4_diff_split_col = (
        get_stationary_time_ratio_per_mouse_sessions(
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            df_dic_saline=df_dic_saline,
            df_dic_dcz=df_dic_dcz,
            behav_pair_map=behav_pair_map,
            subjects=hM4Di_mice,
            split_column=split_col,
            split_labels=split_label,
            difference=True,
        )
    )
    stationary_time_ratio_per_animal_condition_hm3_split_col = (
        get_stationary_time_ratio_per_mouse_sessions(
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            df_dic_saline=df_dic_saline,
            df_dic_dcz=df_dic_dcz,
            behav_pair_map=behav_pair_map,
            subjects=hM3Dq_mice,
            split_column=split_col,
            split_labels=split_label,
        )
    )
    stationary_time_ratio_per_animal_condition_hm3_diff_split_col = (
        get_stationary_time_ratio_per_mouse_sessions(
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            df_dic_saline=df_dic_saline,
            df_dic_dcz=df_dic_dcz,
            behav_pair_map=behav_pair_map,
            subjects=hM3Dq_mice,
            split_column=split_col,
            split_labels=split_label,
            difference=True,
        )
    )
    return (
        stationary_time_ratio_per_animal_condition_hm3,
        stationary_time_ratio_per_animal_condition_hm3_diff,
        stationary_time_ratio_per_animal_condition_hm3_diff_split_col,
        stationary_time_ratio_per_animal_condition_hm3_split_col,
        stationary_time_ratio_per_animal_condition_hm4,
        stationary_time_ratio_per_animal_condition_hm4_diff,
        stationary_time_ratio_per_animal_condition_hm4_diff_split_col,
        stationary_time_ratio_per_animal_condition_hm4_split_col,
    )


@app.cell
def _(
    mo,
    plot_test,
    stationary_time_ratio_per_animal_condition_hm3,
    stationary_time_ratio_per_animal_condition_hm3_diff,
    stationary_time_ratio_per_animal_condition_hm4,
    stationary_time_ratio_per_animal_condition_hm4_diff,
):
    hm4_stationary_time_ratio_condition_session_fig = (
        plot_test.plot_condition_session_values_by_mouse(
            stationary_time_ratio_per_animal_condition_hm4,
            "hM4Di",
            ylabel="Session stationary time ratio",
            title="hM4Di: stationary time ratio by condition",
        )
    )
    hm3_stationary_time_ratio_condition_session_fig = (
        plot_test.plot_condition_session_values_by_mouse(
            stationary_time_ratio_per_animal_condition_hm3,
            "hM3Dq",
            ylabel="Session stationary time ratio",
            title="hM3Dq: stationary time ratio by condition",
        )
    )
    stationary_time_ratio_condition_diff_session_fig = (
        plot_test.plot_group_condition_values_by_mouse(
            stationary_time_ratio_per_animal_condition_hm3_diff,
            stationary_time_ratio_per_animal_condition_hm4_diff,
            ylabel="saline - DCZ (stationary time ratio)",
            title="stationary time ratio saline - DCZ",
        )
    )

    mo.vstack(
        [
            hm4_stationary_time_ratio_condition_session_fig,
            hm3_stationary_time_ratio_condition_session_fig,
            stationary_time_ratio_condition_diff_session_fig,
        ]
    )
    return


@app.cell
def _(
    compare_value_name,
    mo,
    plot_test,
    split_col,
    split_label,
    stationary_time_ratio_per_animal_condition_hm3_diff_split_col,
    stationary_time_ratio_per_animal_condition_hm3_split_col,
    stationary_time_ratio_per_animal_condition_hm4_diff_split_col,
    stationary_time_ratio_per_animal_condition_hm4_split_col,
):
    hm4_stationary_time_ratio_split_condition_session_fig = (
        plot_test.plot_condition_session_values_by_mouse(
            stationary_time_ratio_per_animal_condition_hm4_split_col,
            "hM4Di",
            ylabel="Session stationary time ratio",
            title=(
                f"hM4Di: stationary time ratio by condition and "
                f"{compare_value_name}"
            ),
            split_column=split_col,
            split_labels=split_label,
        )
    )
    hm3_stationary_time_ratio_split_condition_session_fig = (
        plot_test.plot_condition_session_values_by_mouse(
            stationary_time_ratio_per_animal_condition_hm3_split_col,
            "hM3Dq",
            ylabel="Session stationary time ratio",
            title=(
                f"hM3Dq: stationary time ratio by condition and "
                f"{compare_value_name}"
            ),
            split_column=split_col,
            split_labels=split_label,
        )
    )
    stationary_time_ratio_split_condition_diff_session_fig = (
        plot_test.plot_group_condition_values_by_mouse(
            stationary_time_ratio_per_animal_condition_hm3_diff_split_col,
            stationary_time_ratio_per_animal_condition_hm4_diff_split_col,
            ylabel="saline - DCZ (stationary time ratio)",
            title=(
                f"stationary time ratio saline - DCZ by "
                f"{compare_value_name}"
            ),
            split_column=split_col,
            split_labels=split_label,
        )
    )
    mo.vstack(
        [
            hm4_stationary_time_ratio_split_condition_session_fig,
            hm3_stationary_time_ratio_split_condition_session_fig,
            stationary_time_ratio_split_condition_diff_session_fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### combine the subjects
    """)
    return


@app.cell
def _(
    compare_value_name,
    mo,
    np,
    pd,
    plt,
    split_label,
    stationary_time_ratio_per_animal_condition_hm3_diff_split_col,
    stationary_time_ratio_per_animal_condition_hm4_diff_split_col,
):
    from scipy import stats as _stats

    _split_labels = list(split_label)

    def _p_value_to_label(_p_value):
        if pd.isna(_p_value):
            return "n/a"
        if _p_value < 0.001:
            return "***"
        if _p_value < 0.01:
            return "**"
        if _p_value < 0.05:
            return "*"
        return "ns"

    def _add_significance_bar(_ax, _x1, _x2, _y, _h, _label):
        _ax.plot(
            [_x1, _x1, _x2, _x2],
            [_y, _y + _h, _y + _h, _y],
            color="black",
            linewidth=1,
        )
        _ax.text(
            (_x1 + _x2) / 2,
            _y + _h,
            _label,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    def _combined_stationary_diff_rows(_output_dic, _group_name):
        _rows = []

        def _entry_to_pair_value(_entry, _fallback_idx):
            if isinstance(_entry, dict):
                return (
                    _entry.get("pair_id", f"row_{_fallback_idx}"),
                    _entry.get("value", np.nan),
                )
            return f"row_{_fallback_idx}", _entry

        for _modality in ["vis", "aud"]:
            for _subject, _split_values in (
                _output_dic.get(_modality, {}).items()
            ):
                if not isinstance(_split_values, dict):
                    continue
                if any(
                    _label not in _split_values
                    for _label in _split_labels
                ):
                    continue

                for _label in _split_labels:
                    for _entry_idx, _entry in enumerate(
                        _split_values[_label]
                    ):
                        _pair_id, _value = _entry_to_pair_value(
                            _entry,
                            _entry_idx,
                        )
                        _value = pd.to_numeric(
                            pd.Series([_value]),
                            errors="coerce",
                        ).iloc[0]
                        if pd.isna(_value):
                            continue
                        _rows.append(
                            {
                                "group": _group_name,
                                "subject": _subject,
                                "modality": _modality,
                                "pair_id": _pair_id,
                                "split_label": _label,
                                "stationary_time_ratio_diff": float(
                                    _value
                                ),
                            }
                        )
        return _rows

    combined_stationary_time_ratio_diff_split_df = pd.DataFrame(
        _combined_stationary_diff_rows(
            stationary_time_ratio_per_animal_condition_hm3_diff_split_col,
            "hM3Dq",
        )
        + _combined_stationary_diff_rows(
            stationary_time_ratio_per_animal_condition_hm4_diff_split_col,
            "hM4Di",
        )
    )

    def _plot_combined_stationary_group(_df, _group_name):
        _fig, _ax = plt.subplots(figsize=(5, 5), dpi=150)
        _df_group = _df[_df["group"] == _group_name].copy()

        if _df_group.empty:
            _ax.text(
                0.5,
                0.5,
                "No paired values to plot.",
                ha="center",
                va="center",
                transform=_ax.transAxes,
            )
            _ax.set_axis_off()
            _fig.tight_layout()
            return _fig

        _data = [
            _df_group.loc[
                _df_group["split_label"] == _label,
                "stationary_time_ratio_diff",
            ].dropna()
            for _label in _split_labels
        ]
        _colors = ["#4C78A8", "#F58518"]
        _x_positions = np.arange(1, len(_split_labels) + 1)

        _boxplot = _ax.boxplot(
            _data,
            labels=_split_labels,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
        )
        for _patch, _color in zip(_boxplot["boxes"], _colors):
            _patch.set_facecolor(_color)
            _patch.set_alpha(0.22)
            _patch.set_edgecolor(_color)

        _paired_df = _df_group.pivot_table(
            index=["subject", "modality", "pair_id"],
            columns="split_label",
            values="stationary_time_ratio_diff",
            aggfunc="mean",
        )
        _paired_df = _paired_df.dropna(subset=_split_labels)

        for _, _row in _paired_df.iterrows():
            _ax.plot(
                _x_positions,
                [_row[_label] for _label in _split_labels],
                color="gray",
                alpha=0.3,
                linewidth=1,
                zorder=2,
            )

        for _idx, (_label, _color) in enumerate(
            zip(_split_labels, _colors),
            start=1,
        ):
            _values = _df_group.loc[
                _df_group["split_label"] == _label,
                "stationary_time_ratio_diff",
            ].dropna().to_numpy(dtype=float)
            if len(_values) == 0:
                continue
            _jitter = np.linspace(-0.06, 0.06, len(_values))
            _ax.scatter(
                _idx + _jitter,
                _values,
                color=_color,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.75,
                s=35,
                zorder=3,
            )

        if len(_split_labels) == 2 and len(_paired_df) >= 2:
            try:
                _p_value = _stats.ttest_rel(
                    _paired_df[_split_labels[0]],
                    _paired_df[_split_labels[1]],
                    nan_policy="omit",
                ).pvalue
            except ValueError:
                _p_value = np.nan

            _y_values = pd.to_numeric(
                _df_group["stationary_time_ratio_diff"],
                errors="coerce",
            ).dropna()
            if not _y_values.empty:
                _y_min = _y_values.min()
                _y_max = _y_values.max()
                _y_range = _y_max - _y_min
                if _y_range == 0:
                    _y_range = abs(_y_max) * 0.1 if _y_max != 0 else 1
                _bar_y = _y_max + _y_range * 0.12
                _bar_h = _y_range * 0.06
                _add_significance_bar(
                    _ax,
                    _x_positions[0],
                    _x_positions[1],
                    _bar_y,
                    _bar_h,
                    _p_value_to_label(_p_value),
                )
                _ax.set_ylim(top=_bar_y + _bar_h + _y_range * 0.15)

        _ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        _ax.set_xlabel(compare_value_name)
        _ax.set_ylabel("saline - DCZ (stationary time ratio)")
        _ax.set_title(f"{_group_name}: combined aud/vis subjects")
        _ax.grid(True, axis="y", alpha=0.3)
        _fig.tight_layout()
        return _fig

    combined_stationary_time_ratio_diff_split_figures = [
        _plot_combined_stationary_group(
            combined_stationary_time_ratio_diff_split_df,
            "hM3Dq",
        ),
        _plot_combined_stationary_group(
            combined_stationary_time_ratio_diff_split_df,
            "hM4Di",
        ),
    ]
    _plot_combined_stationary_group(combined_stationary_time_ratio_diff_split_df, "hM3Dq").savefig('/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/hm3_bypreviouschoice_stationarytimerario.svg')
    _plot_combined_stationary_group(combined_stationary_time_ratio_diff_split_df, "hM4Di").savefig('/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/hm4_bypreviouschoice_stationarytimerario.svg')
    mo.hstack(combined_stationary_time_ratio_diff_split_figures)
    return


@app.cell
def _(
    mo,
    plot_combined_saline_dcz_by_group,
    stationary_time_ratio_per_animal_condition_hm3,
    stationary_time_ratio_per_animal_condition_hm4,
):
    (
        combined_stationary_time_ratio_condition_df,
        combined_stationary_time_ratio_condition_figures,
    ) = plot_combined_saline_dcz_by_group(
        stationary_time_ratio_per_animal_condition_hm3,
        stationary_time_ratio_per_animal_condition_hm4,
        metric_col="stationary_time_ratio",
        ylabel="Stationary time ratio",
        title_prefix="Stationary time ratio",
    )

    mo.hstack(combined_stationary_time_ratio_condition_figures)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # compare performance according to the conditions
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # compare the trial with time
    """)
    return


@app.cell
def _(
    df_dic_dcz,
    df_dic_saline,
    hM3Dq_mice,
    hM4Di_mice,
    mo,
    pd,
    plots,
    plt,
    utils,
):
    _point_color_column = "previous_correct"

    _point_colors = {
        ("saline", True): "steelblue",
        ("saline", False): "limegreen",
        ("dcz", True): "firebrick",
        ("dcz", False): "gold",
    }

    _paired_ids = set(df_dic_saline) & set(df_dic_dcz)
    _paired_frames = [
        condition_dic[pair_id]
        for pair_id in sorted(_paired_ids)
        for condition_dic in (df_dic_saline, df_dic_dcz)
    ]

    _df_test_paired = (
        pd.concat(_paired_frames, ignore_index=True)
        if _paired_frames
        else pd.DataFrame(columns=["subject"])
    )

    _trial_time_figures = []

    for _subject, _df_mouse in _df_test_paired.groupby("subject"):
        _group_label = (
            "hM3Dq"
            if _subject in hM3Dq_mice
            else "hM4Di"
            if _subject in hM4Di_mice
            else "unknown"
        )
        _df_mouse = utils.add_time_from_session_start(
            _df_mouse.copy()
        )
        _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

        for _ax, _modality in zip(
            _axes,
            ("visual", "auditory"),
        ):
            _df_modality = _df_mouse[
                _df_mouse["stimulus_modality"] == _modality
            ]

            for _observation in ("saline", "dcz"):
                _df_plot = _df_modality[
                    _df_modality["observations"].str.contains(
                        _observation,
                        case=False,
                        na=False,
                    )
                ].dropna(
                    subset=[
                        _point_color_column,
                        "time_from_start",
                        "trial",
                    ]
                )

                if _df_plot.empty:
                    continue

                plots.plot_trial_time_of_start(
                    _df_plot,
                    ax=_ax,
                )

                _colors = [
                    _point_colors[
                        (_observation, bool(value))
                    ]
                    for value in _df_plot[_point_color_column]
                ]

                # plot_trial_time_of_start 新生成的 scatter
                _scatter = _ax.collections[-1]
                _scatter.set_facecolors(_colors)
                _scatter.set_edgecolors(_colors)

            # 自动生成四个 legend 项
            for (_observation, _value), _color in _point_colors.items():
                _ax.scatter(
                    [],
                    [],
                    color=_color,
                    s=16,
                    label=(
                        f"{_observation} "
                        f"{_point_color_column}={_value}"
                    ),
                )

            _ax.set_title(_modality.capitalize())
            _ax.legend(frameon=False)

        _fig.suptitle(f"{_subject} ({_group_label})")
        _fig.tight_layout()
        _trial_time_figures.append(_fig)

    mo.vstack(_trial_time_figures) if _trial_time_figures else mo.md(
        "No paired saline/DCZ sessions to plot."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## generate roi out trials
    """)
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behavior_utils,
    df_dic_dcz,
    df_dic_saline,
    np,
    pd,
    roi_bottom,
    roi_left,
    roi_right,
    roi_top,
):
    for _trial_dic, _behav_dic in [
        (df_dic_saline, behav_df_dic_saline),
        (df_dic_dcz, behav_df_dic_dcz),
    ]:
        for _pair_id in set(_trial_dic) & set(_behav_dic):
            _trial_df = _trial_dic[_pair_id].copy()
            _behav_df = _behav_dic[_pair_id]

            _timestamp = pd.to_numeric(
                behavior_utils.get_behavior_column(
                    _behav_df,
                    ("timestamp", ""),
                ),
                errors="coerce",
            ).to_numpy()

            _x = pd.to_numeric(
                behavior_utils.get_behavior_column(
                    _behav_df,
                    ("Center", "x"),
                ),
                errors="coerce",
            ).to_numpy()

            _y = pd.to_numeric(
                behavior_utils.get_behavior_column(
                    _behav_df,
                    ("Center", "y"),
                ),
                errors="coerce",
            ).to_numpy()

            _valid_xy = np.isfinite(_x) & np.isfinite(_y)

            _outside_roi = _valid_xy & (
                (_x < roi_left)
                | (_x > roi_right)
                | (_y < roi_bottom)
                | (_y > roi_top)
            )

            _trial_starts = pd.to_numeric(
                _trial_df["TRIAL_START"],
                errors="coerce",
            ).to_numpy()

            _trial_ends = pd.to_numeric(
                _trial_df["TRIAL_END"],
                errors="coerce",
            ).to_numpy()

            _trial_df["roi_out"] = [
                bool(
                    _outside_roi[
                        (_timestamp >= _start)
                        & (_timestamp <= _end)
                    ].any()
                )
                for _start, _end in zip(
                    _trial_starts,
                    _trial_ends,
                )
            ]

            # 明确写回原来的字典
            _trial_dic[_pair_id] = _trial_df
    return


@app.cell
def _(
    df_dic_dcz,
    df_dic_saline,
    hM3Dq_mice,
    hM4Di_mice,
    mo,
    pd,
    plots,
    plt,
    utils,
):
    _point_color_column = "roi_out"

    _point_colors = {
        ("saline", False): "steelblue",
        ("saline", True): "limegreen",
        ("dcz", False): "firebrick",
        ("dcz", True): "gold",
    }

    _paired_ids = set(df_dic_saline) & set(df_dic_dcz)
    _paired_frames = [
        condition_dic[pair_id]
        for pair_id in sorted(_paired_ids)
        for condition_dic in (df_dic_saline, df_dic_dcz)
    ]

    _df_test_paired = (
        pd.concat(_paired_frames, ignore_index=True)
        if _paired_frames
        else pd.DataFrame(columns=["subject"])
    )

    _trial_time_figures = []

    for _subject, _df_mouse in _df_test_paired.groupby("subject"):
        _group_label = (
            "hM3Dq"
            if _subject in hM3Dq_mice
            else "hM4Di"
            if _subject in hM4Di_mice
            else "unknown"
        )
        _df_mouse = utils.add_time_from_session_start(
            _df_mouse.copy()
        )
        _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

        for _ax, _modality in zip(
            _axes,
            ("visual", "auditory"),
        ):
            _df_modality = _df_mouse[
                _df_mouse["stimulus_modality"] == _modality
            ]

            for _observation in ("saline", "dcz"):
                _df_plot = _df_modality[
                    _df_modality["observations"].str.contains(
                        _observation,
                        case=False,
                        na=False,
                    )
                ].dropna(
                    subset=[
                        _point_color_column,
                        "time_from_start",
                        "trial",
                    ]
                )

                if _df_plot.empty:
                    continue

                plots.plot_trial_time_of_start(
                    _df_plot,
                    ax=_ax,
                )

                _colors = [
                    _point_colors[
                        (_observation, bool(value))
                    ]
                    for value in _df_plot[_point_color_column]
                ]

                # plot_trial_time_of_start 新生成的 scatter
                _scatter = _ax.collections[-1]
                _scatter.set_facecolors(_colors)
                _scatter.set_edgecolors(_colors)

            # 自动生成四个 legend 项
            for (_observation, _value), _color in _point_colors.items():
                _ax.scatter(
                    [],
                    [],
                    color=_color,
                    s=16,
                    label=(
                        f"{_observation} "
                        f"{_point_color_column}={_value}"
                    ),
                )

            _ax.set_title(_modality.capitalize())
            _ax.legend(frameon=False)

        _fig.suptitle(f"{_subject} ({_group_label})")
        _fig.tight_layout()
        _trial_time_figures.append(_fig)

    mo.vstack(_trial_time_figures) if _trial_time_figures else mo.md(
        "No paired saline/DCZ sessions to plot."
    )
    return


@app.cell
def _(
    df_dic_dcz,
    df_dic_saline,
    hM3Dq_mice,
    hM4Di_mice,
    mo,
    pd,
    plots,
    plt,
    utils,
):
    _point_color_column = "engaged"

    _point_colors = {
        ("saline", True): "steelblue",
        ("saline", False): "limegreen",
        ("dcz", True): "firebrick",
        ("dcz", False): "gold",
    }

    _paired_ids = set(df_dic_saline) & set(df_dic_dcz)
    _paired_frames = [
        condition_dic[pair_id]
        for pair_id in sorted(_paired_ids)
        for condition_dic in (df_dic_saline, df_dic_dcz)
    ]

    _df_test_paired = (
        pd.concat(_paired_frames, ignore_index=True)
        if _paired_frames
        else pd.DataFrame(columns=["subject"])
    )

    _trial_time_figures = []

    for _subject, _df_mouse in _df_test_paired.groupby("subject"):
        _group_label = (
            "hM3Dq"
            if _subject in hM3Dq_mice
            else "hM4Di"
            if _subject in hM4Di_mice
            else "unknown"
        )
        _df_mouse = utils.add_time_from_session_start(
            _df_mouse.copy()
        )
        _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

        for _ax, _modality in zip(
            _axes,
            ("visual", "auditory"),
        ):
            _df_modality = _df_mouse[
                _df_mouse["stimulus_modality"] == _modality
            ]

            for _observation in ("saline", "dcz"):
                _df_plot = _df_modality[
                    _df_modality["observations"].str.contains(
                        _observation,
                        case=False,
                        na=False,
                    )
                ].dropna(
                    subset=[
                        _point_color_column,
                        "time_from_start",
                        "trial",
                    ]
                )

                if _df_plot.empty:
                    continue

                plots.plot_trial_time_of_start(
                    _df_plot,
                    ax=_ax,
                )

                _colors = [
                    _point_colors[
                        (_observation, bool(value))
                    ]
                    for value in _df_plot[_point_color_column]
                ]

                # plot_trial_time_of_start 新生成的 scatter
                _scatter = _ax.collections[-1]
                _scatter.set_facecolors(_colors)
                _scatter.set_edgecolors(_colors)

            # 自动生成四个 legend 项
            for (_observation, _value), _color in _point_colors.items():
                _ax.scatter(
                    [],
                    [],
                    color=_color,
                    s=16,
                    label=(
                        f"{_observation} "
                        f"{_point_color_column}={_value}"
                    ),
                )

            _ax.set_title(_modality.capitalize())
            _ax.legend(frameon=False)

        _fig.suptitle(f"{_subject} ({_group_label})")
        _fig.tight_layout()
        _trial_time_figures.append(_fig)

    mo.vstack(_trial_time_figures) if _trial_time_figures else mo.md(
        "No paired saline/DCZ sessions to plot."
    )
    return


@app.cell
def _(df_dic_dcz, df_dic_saline, pd, plots, plt, utils):
    _point_color_column = "engaged"

    _point_colors = {
        ("saline", True): "steelblue",
        ("saline", False): "limegreen",
        ("dcz", True): "firebrick",
        ("dcz", False): "gold",
    }

    _selected_subjects = ["NUO010", "NUO012"]

    _paired_ids = set(df_dic_saline) & set(df_dic_dcz)
    _paired_frames = [
        condition_dic[pair_id]
        for pair_id in sorted(_paired_ids)
        for condition_dic in (df_dic_saline, df_dic_dcz)
    ]

    _df_test_paired = (
        pd.concat(_paired_frames, ignore_index=True)
        if _paired_frames
        else pd.DataFrame(columns=["subject"])
    )

    _df_mouse = _df_test_paired[
        _df_test_paired["subject"].isin(_selected_subjects)
    ].copy()

    _df_mouse = utils.add_time_from_session_start(_df_mouse)

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    for _ax, _modality in zip(
        _axes,
        ("visual", "auditory"),
    ):
        _df_modality = _df_mouse[
            _df_mouse["stimulus_modality"] == _modality
        ]

        for _observation in ("saline", "dcz"):
            _df_plot = _df_modality[
                _df_modality["observations"].str.contains(
                    _observation,
                    case=False,
                    na=False,
                )
            ].dropna(
                subset=[
                    _point_color_column,
                    "time_from_start",
                    "trial",
                ]
            )

            if _df_plot.empty:
                continue

            plots.plot_trial_time_of_start(
                _df_plot,
                ax=_ax,
            )

            _colors = [
                _point_colors[
                    (_observation, bool(value))
                ]
                for value in _df_plot[_point_color_column]
            ]

            _scatter = _ax.collections[-1]
            _scatter.set_facecolors(_colors)
            _scatter.set_edgecolors(_colors)

        for (_observation, _value), _color in _point_colors.items():
            _ax.scatter(
                [],
                [],
                color=_color,
                s=16,
                label=(
                    f"{_observation} "
                    f"{_point_color_column}={_value}"
                ),
            )

        _ax.set_title(_modality.capitalize())
        _ax.legend(frameon=False)

    _fig.suptitle("NUO010 + NUO012")
    _fig.tight_layout()
    # _fig.savefig("/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/trials_start_across_time_engaged_nuo010nuo012.png")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## venn plot of disengaged trials
    """)
    return


@app.cell
def _(df_dic_dcz, df_dic_saline, hM3Dq_mice, hM4Di_mice, pd):
    def _get_trial_sets_for_venn(df):
        if df.empty:
            return {
                "previous_correct = False": set(),
                "engaged = False": set(),
                "roi_out = True": set(),
                "react_slow = True": set(),
                "break_more = True": set(),
            }

        if "trial" in df.columns:
            trial_ids = df["trial"]
        else:
            trial_ids = pd.Series(df.index, index=df.index)

        condition_masks = {
            "previous_correct = False": (
                df["previous_correct"].eq(False)
                if "previous_correct" in df.columns
                else pd.Series(False, index=df.index)
            ),
            "engaged = False": (
                df["engaged"].eq(False)
                if "engaged" in df.columns
                else pd.Series(False, index=df.index)
            ),
            "roi_out = True": (
                df["roi_out"].eq(True)
                if "roi_out" in df.columns
                else pd.Series(False, index=df.index)
            ),
            "react_slow = True": (
                df["react_slow"].eq(True)
                if "react_slow" in df.columns
                else pd.Series(False, index=df.index)
            ),
            "break_more = True": (
                df["break_more"].eq(True)
                if "break_more" in df.columns
                else pd.Series(False, index=df.index)
            ),
        }

        return {
            label: set(trial_ids[mask].dropna().tolist())
            for label, mask in condition_masks.items()
        }

    _hm3_mice = {str(_mouse) for _mouse in hM3Dq_mice}
    _hm4_mice = {str(_mouse) for _mouse in hM4Di_mice}
    venn_disengaged_trial_sets = {}
    venn_disengaged_trial_metadata = {}
    _venn_disengaged_trial_rows = []

    for _pair_id in sorted(set(df_dic_saline) & set(df_dic_dcz)):
        venn_disengaged_trial_sets[_pair_id] = {}
        _pair_df = pd.concat(
            [
                df_dic_saline[_pair_id],
                df_dic_dcz[_pair_id],
            ],
            ignore_index=True,
        )

        _subjects = (
            _pair_df["subject"].dropna().astype(str).unique()
            if "subject" in _pair_df.columns
            else []
        )
        _subject = _subjects[0] if len(_subjects) else "unknown"
        _mouse_group = (
            "hM3Dq"
            if _subject in _hm3_mice
            else "hM4Di"
            if _subject in _hm4_mice
            else "unknown"
        )
        _modalities = (
            _pair_df["stimulus_modality"].dropna().astype(str)
            if "stimulus_modality" in _pair_df.columns
            else []
        )
        _modality_label = (
            _modalities.iloc[0]
            if len(_modalities)
            else "unknown modality"
        )
        venn_disengaged_trial_metadata[_pair_id] = {
            "subject": _subject,
            "mouse_group": _mouse_group,
            "modality": _modality_label,
        }

        for _condition_name, _condition_dic in {
            "saline": df_dic_saline,
            "dcz": df_dic_dcz,
        }.items():
            _trial_sets = _get_trial_sets_for_venn(
                _condition_dic[_pair_id]
            )
            venn_disengaged_trial_sets[_pair_id][
                _condition_name
            ] = _trial_sets

            _previous_incorrect = _trial_sets[
                "previous_correct = False"
            ]
            _disengaged = _trial_sets["engaged = False"]
            _roi_out = _trial_sets["roi_out = True"]
            _react_slow = _trial_sets["react_slow = True"]
            _break_more = _trial_sets["break_more = True"]

            _venn_disengaged_trial_rows.append(
                {
                    "pair_id": _pair_id,
                    "subject": _subject,
                    "mouse_group": _mouse_group,
                    "modality": _modality_label,
                    "condition": _condition_name,
                    "previous_correct_false": len(
                        _previous_incorrect
                    ),
                    "engaged_false": len(_disengaged),
                    "roi_out_true": len(_roi_out),
                    "react_slow_true": len(_react_slow),
                    "break_more_true": len(_break_more),
                    "previous_false_and_disengaged": len(
                        _previous_incorrect & _disengaged
                    ),
                    "previous_false_and_roi_out": len(
                        _previous_incorrect & _roi_out
                    ),
                    "disengaged_and_roi_out": len(
                        _disengaged & _roi_out
                    ),
                    "all_three": len(
                        _previous_incorrect & _disengaged & _roi_out
                    ),
                    "all_five": len(
                        _previous_incorrect
                        & _disengaged
                        & _roi_out
                        & _react_slow
                        & _break_more
                    ),
                }
            )

    venn_disengaged_trial_summary = pd.DataFrame(
        _venn_disengaged_trial_rows
    )
    venn_disengaged_trial_summary
    return (
        venn_disengaged_trial_metadata,
        venn_disengaged_trial_sets,
        venn_disengaged_trial_summary,
    )


@app.cell
def _(
    hM3Dq_mice,
    mo,
    plot_five_set_venn,
    plot_upset_from_trial_sets,
    venn_disengaged_trial_sets,
    venn_disengaged_trial_summary,
):
    venn_disengaged_trial_summary_selectforplot = venn_disengaged_trial_summary[
        (venn_disengaged_trial_summary['subject'].isin(hM3Dq_mice)) 
        # (venn_disengaged_trial_summary['subject'].isin(hM4Di_mice)) 
        # & (venn_disengaged_trial_summary['modality'] == 'auditory')
        ]
    _selected_pair_ids = (
        venn_disengaged_trial_summary_selectforplot["pair_id"]
        .dropna()
        .unique()
    )

    _selected_summary_sets = {}

    for _condition_name in ["saline", "dcz"]:
        _first_pair_id = _selected_pair_ids[0]
        _labels = venn_disengaged_trial_sets[
            _first_pair_id
        ][_condition_name].keys()

        _selected_summary_sets[_condition_name] = {
            _label: set().union(
                *[
                    venn_disengaged_trial_sets[_pair_id][
                        _condition_name
                    ][_label]
                    for _pair_id in _selected_pair_ids
                ]
            )
            for _label in _labels
        }

    _selected_summary_view = []
    for _condition_name in ["saline", "dcz"]:
        _condition_title = f"{_condition_name.upper()}"
        _selected_summary_view.append(mo.md(f"### {_condition_title}"))
        _selected_summary_view.append(
            mo.hstack(
                [
                    plot_five_set_venn(
                        _selected_summary_sets[_condition_name],
                        title=_condition_title,
                    ),
                    plot_upset_from_trial_sets(
                        _selected_summary_sets[_condition_name],
                        title=f"{_condition_title} UpSet",
                    ),
                ]
            )
        )
        plot_five_set_venn(
            _selected_summary_sets[_condition_name],title=_condition_title
        ).savefig(
            f"/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/hm3_{_condition_title}_venn.svg"
        )
        plot_upset_from_trial_sets(
            _selected_summary_sets[_condition_name],title=_condition_title
        ).savefig(
            f"/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/hm3_{_condition_title}_upset.svg"
        )
    mo.vstack(_selected_summary_view)
    return


@app.cell
def _(mo, plt, venn_disengaged_trial_metadata, venn_disengaged_trial_sets):
    import sys as _sys
    import importlib as _importlib
    from upsetplot import UpSet as _UpSet
    from upsetplot import from_memberships as _from_memberships

    _pyvenn_path = "/home/kudongdong/code/pkg/pyvenn"
    if _pyvenn_path not in _sys.path:
        _sys.path.insert(0, _pyvenn_path)

    import venn as _pyvenn
    _pyvenn = _importlib.reload(_pyvenn)

    def plot_five_set_venn(trial_sets, title):
        _labels = list(trial_sets)
        _sets = [trial_sets[_label] for _label in _labels]

        if not any(_sets):
            _fig, _ax = plt.subplots(figsize=(5, 5), dpi=150)
            _ax.text(
                0.5,
                0.5,
                "No trials",
                ha="center",
                va="center",
                transform=_ax.transAxes,
            )
            _ax.set_title(title)
            _ax.axis("off")
            return _fig

        _venn_labels = _pyvenn.get_labels(
            _sets,
            fill=["number"],
        )
        _fig, _ax = _pyvenn.venn5(
            _venn_labels,
            names=_labels,
            figsize=(7, 7),
            dpi=150,
            fontsize=9,
        )
        _fig.suptitle(title)
        _fig.tight_layout()
        return _fig

    def plot_upset_from_trial_sets(trial_sets, title):
        _labels = list(trial_sets)
        _all_trials = set().union(*trial_sets.values())
        _memberships = [
            [
                _label
                for _label in _labels
                if _trial_id in trial_sets[_label]
            ]
            for _trial_id in _all_trials
        ]
        _memberships = [
            _membership
            for _membership in _memberships
            if _membership
        ]

        if not _memberships:
            _fig, _ax = plt.subplots(figsize=(6, 4), dpi=150)
            _ax.text(
                0.5,
                0.5,
                "No trials",
                ha="center",
                va="center",
                transform=_ax.transAxes,
            )
            _ax.set_title(title)
            _ax.axis("off")
            return _fig

        _upset_data = _from_memberships(_memberships)
        _fig = plt.figure(figsize=(7, 4.5), dpi=150)
        _upset = _UpSet(
            _upset_data,
            subset_size="count",
            show_counts=True,
            show_percentages=True,
            sort_by="cardinality",
        )
        _upset.plot(fig=_fig)
        _fig.suptitle(title)
        _fig.tight_layout()
        return _fig

    _venn_disengaged_trial_view = []

    for _pair_id, _condition_sets in venn_disengaged_trial_sets.items():
        _metadata = venn_disengaged_trial_metadata.get(
            _pair_id,
            {},
        )
        _venn_disengaged_trial_view.append(
            mo.md(
            (
                    f"### {_pair_id} "
                f"({_metadata.get('mouse_group', 'unknown')}, "
                f"{_metadata.get('modality', 'unknown modality')})"
            )
            )
        )

        for _condition_name in ["saline", "dcz"]:
            _condition_title = _condition_name.upper()
            _venn_disengaged_trial_view.append(
                mo.hstack(
                    [
                        plot_five_set_venn(
                            _condition_sets[_condition_name],
                            title=_condition_title,
                        ),
                        plot_upset_from_trial_sets(
                            _condition_sets[_condition_name],
                            title=f"{_condition_title} UpSet",
                        ),
                    ]
                )
            )

    mo.vstack(_venn_disengaged_trial_view) 
    return plot_five_set_venn, plot_upset_from_trial_sets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # condition compare by sessions
    """)
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    df_dic_dcz,
    df_dic_saline,
    hM3Dq_mice,
    hM4Di_mice,
    mo,
    plot_test,
    roi_bottom,
    roi_left,
    roi_right,
    roi_top,
    save_behavior_svg,
    stimulus_modality_select,
):
    trial_split_analysis = plot_test.analyze_paired_behavior_by_trial_column(
        stimulus=stimulus_modality_select.value,
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
        mo=mo,
    )
    # trial_split_analysis["view"]
    return (trial_split_analysis,)


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    df_dic_dcz,
    df_dic_saline,
    hM3Dq_mice,
    hM4Di_mice,
    mo,
    plot_test,
    roi_bottom,
    roi_left,
    roi_right,
    roi_top,
    save_behavior_svg,
    stimulus_modality_select,
):
    analysis_by_column = {}
    view_list = []

    for column_name in ["engaged", "correct", "previous_correct"]:
        analysis = plot_test.analyze_paired_behavior_by_trial_column(
            stimulus=stimulus_modality_select.value,
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
            mo=mo,
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
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    df_dic_dcz,
    df_dic_saline,
    hM3Dq_mice,
    hM4Di_mice,
    mo,
    plot_test,
    roi_bottom,
    roi_left,
    roi_right,
    roi_top,
    save_behavior_svg,
    stimulus_modality_select,
):

    trial_split_analysis_engage_roi = plot_test.analyze_paired_behavior_by_trial_column(
        stimulus=stimulus_modality_select.value,
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
        include_trajectory_speed=False,
        include_roi_time=False,
        include_stationary_speed=False,
        include_stationary_time=False,
        stationary_speed_threshold=10,
        speed_col="mean_speed",
        mo=mo,
    )
    trial_split_analysis_engage_roi["view"]
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    df_dic_dcz,
    df_dic_saline,
    hM3Dq_mice,
    hM4Di_mice,
    mo,
    plot_test,
    roi_bottom,
    roi_left,
    roi_right,
    roi_top,
    save_behavior_svg,
    stimulus_modality_select,
):
    analysis_by_column_roi = {}
    view_list_roi = []

    for column_name_roi in ["engaged", "correct", "previous_correct"]:
        analysis_roi = plot_test.analyze_paired_behavior_by_trial_column(
            stimulus=stimulus_modality_select.value,
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
            mo=mo,
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

    _fig, _ax = plt.subplots(figsize=(13, 4))

    for trial_idx, (_, row) in enumerate(plot_df.iterrows()):
        y = len(plot_df) - trial_idx

        trial_start_times = as_event_times(row.get("TRIAL_START", np.nan))
        t0 = trial_start_times[0]

        for col in event_cols:
            event_times = as_event_times(row.get(col, np.nan))
            for event_time in event_times:
                _ax.vlines(
                    event_time,
                    y - 0.38,
                    y + 0.38,
                    color=event_color_map[col],
                    linewidth=2,
                    alpha=0.95,
                )

    _ax.set_yticks(range(1, len(plot_df) + 1))
    _ax.set_yticklabels([f"trial {i}" for i in range(len(plot_df), 0, -1)])
    _ax.set_xlabel("time from TRIAL_START (s)")
    _ax.set_ylabel("trial")
    _ax.set_title("First 5 trials state event raster")
    _ax.grid(axis="x", alpha=0.25)

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

    _ax.legend(
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
