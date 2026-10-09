import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["ipynb"])


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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    _default_project = 'frequency_vs_locationPureTone_data'
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
    animals = [_a for _a in animals if "test" not in _a]
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
        value="Auto",
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
    return (available_dates,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # date select
    """)
    return


@app.cell
def _(available_dates, mo):
    date_select_start, date_select_end = (
        mo.ui.dropdown(
            options=available_dates,
            value=default,
            label=f"Date_{label}",
        )
        for label, default in [
            ("start", available_dates[0]),
            ("end", available_dates[-1]),
        ]
    )
    mo.vstack([date_select_start, date_select_end])
    return date_select_end, date_select_start


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # stimulus modality select
    """)
    return


@app.cell
def _(mo):
    stimulus_modality_select = mo.ui.dropdown(
        options=["visual", "auditory", "auditory_locaPT", "all"],
        value="all",
        label="Stimulus modality",
    )
    stimulus_modality_select
    return (stimulus_modality_select,)


@app.cell
def _(animals, mo):
    fit_mouse_select = mo.ui.multiselect(
        options=animals,
        value=animals,
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
def _(
    Path,
    dft,
    fit_mouse_select,
    mouse,
    pd,
    project,
    safe_rsync_cluster_data,
    stimulus_modality_select,
    update_processData_button,
    utils,
):
    parameters_output_dir = (
        Path(utils.get_outpath()) / project / "processed" / "parameters_for_fit" / "auto"
    )
    stim_columns = {
        "df_test_vis": "visual",
        "df_test_aud": "auditory",
        "df_test_aud_loca": "auditory_locaPT"
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
        df_manual = df_all_mouse[df_all_mouse["run_mode"] == "Auto"]
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

    df_test_vis_raw = (
        pd.concat(combined_results["df_test_vis"], ignore_index=True)
        if combined_results["df_test_vis"]
        else pd.DataFrame()
    )
    df_test_aud_raw = (
        pd.concat(combined_results["df_test_aud"], ignore_index=True)
        if combined_results["df_test_aud"]
        else pd.DataFrame()
    )
    df_test_aud_loca_raw = (
        pd.concat(combined_results["df_test_aud_loca"], ignore_index=True)
        if combined_results["df_test_aud_loca"]
        else pd.DataFrame()
    )
    if not df_test_vis_raw.empty:
        df_test_vis_raw = df_test_vis_raw.copy()
        df_test_vis_raw["stimulus_modality"] = "visual"
    if not df_test_aud_raw.empty:
        df_test_aud_raw = df_test_aud_raw.copy()
        df_test_aud_raw["stimulus_modality"] = "auditory"
    if not df_test_aud_loca_raw.empty:
        df_test_aud_loca_raw = df_test_aud_loca_raw.copy()
        df_test_aud_loca_raw["stimulus_modality"] = "auditory_locaPT"
    if stimulus_modality_select.value == "visual":
        df_test_selected_raw = df_test_vis_raw
    elif stimulus_modality_select.value == "auditory":
        df_test_selected_raw = df_test_aud_raw
    elif stimulus_modality_select.value == "auditory_locaPT":
        df_test_selected_raw = df_test_aud_loca_raw
    else:
        selected_raw_dfs = [
            df_raw
            for df_raw in [df_test_vis_raw, df_test_aud_raw, df_test_aud_loca_raw]
            if not df_raw.empty
        ]
        df_test_selected_raw = (
            pd.concat(selected_raw_dfs, ignore_index=True)
            if selected_raw_dfs
            else pd.DataFrame()
        )

    # auditory_stimulus is stored as a string representation of a dictionary.
    # Keep both the unsigned sound level and a left-negative/right-positive version.
    if not df_test_selected_raw.empty and "auditory_stimulus" in df_test_selected_raw:
        import ast as _ast

        def _extract_sound_db_spl(_stimulus):
            if isinstance(_stimulus, dict):
                return _stimulus.get("sound_db_spl")
            if pd.isna(_stimulus):
                return pd.NA
            try:
                return _ast.literal_eval(_stimulus).get("sound_db_spl")
            except (ValueError, SyntaxError, AttributeError, TypeError):
                return pd.NA

        df_test_selected_raw = df_test_selected_raw.copy()
        df_test_selected_raw["sound_db_spl"] = pd.to_numeric(
            df_test_selected_raw["auditory_stimulus"].map(_extract_sound_db_spl),
            errors="coerce",
        )
        _side_sign = df_test_selected_raw["correct_side"].map(
            {"left": 1, "right": -1}
        )
        df_test_selected_raw["signed_sound_db_spl"] = (
            df_test_selected_raw["sound_db_spl"] * _side_sign
        )
        df_test_selected_raw = dft.add_day_column_to_df(df_test_selected_raw.copy())

    return (df_test_selected_raw,)


@app.cell
def _(pd, utils):
    def add_number_of_pokes(df_raw: pd.DataFrame, port_number: int):
        df = df_raw.copy()
        port_hold_column = f"port{port_number}_holds"
        df[port_hold_column] = df.apply(lambda row: utils.get_trial_port_hold(row, port_number), axis=1)
        df[f"port{port_number}_pokes_num"] = df[port_hold_column].apply(lambda x: len(x))
        return df

    return


@app.cell
def _():
    # df_test_selected_upd = dft.add_trial_duration_column_to_df(
    #     dft.calculate_time_between_trials_and_reaction_time(
    #         dft.add_day_column_to_df(
    #             add_number_of_pokes(df_test_selected_raw, port_number=2)
    #         )
    #     )
    # )
    # df_test_selected_upd = utils_test.add_number_of_centralpokes_after_choice(
    #     utils_test.add_fixation_break_columns(
    #         df_test_selected_upd
    #         )
    #     )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Psychometric curve
    """)
    return


@app.cell
def _(df_test_selected_raw):
    df_test_selected_raw
    return


@app.cell
def _(
    date_select_end,
    date_select_start,
    df_test_selected_raw,
    plt,
    utils_test,
):
    _psychometric_config = {
        "visual": ("visual_stimulus_ratio", "discrete"),
        "auditory": ("total_evidence_strength", "continue"),
        "auditory_locaPT": ("signed_sound_db_spl", "continue"),
    }
    _selected_date_start = date_select_start.value
    _selected_date_end = date_select_end.value

    _df_plot = df_test_selected_raw[
        df_test_selected_raw["year_month_day"].astype(str).between(
            _selected_date_start,
            _selected_date_end,
            inclusive="both",
        )
    ].copy()

    for _stim_mod, _df_mod in _df_plot.groupby("stimulus_modality"):
        if _stim_mod not in _psychometric_config:
            continue

        _stimulus_col, _value_type = _psychometric_config[_stim_mod]
        _fig, _ax = plt.subplots(figsize=(5, 5), dpi=150)
        _mouse_groups = list(_df_mod.groupby("subject"))
        _colors = plt.colormaps["tab10"].resampled(len(_mouse_groups))

        for _index, (_animal, _df_animal) in enumerate(_mouse_groups):
            utils_test.psychometric_plot_easy_logistic(
                _df_animal,
                x=_stimulus_col,
                y="first_choice_numeric",
                ax=_ax,
                valueType=_value_type,
                bins=6,
                point_kwargs={
                    "color": _colors(_index),
                    "label": "",
                    "alpha": 0.3,
                },
                line_kwargs={
                    "color": _colors(_index),
                    "label": _animal,
                    "alpha": 0.3,
                },
            )

        utils_test.psychometric_plot_easy_logistic(
            _df_mod,
            x=_stimulus_col,
            y="first_choice_numeric",
            ax=_ax,
            valueType=_value_type,
            bins=6,
            point_kwargs={"color": "black", "label": ""},
            line_kwargs={"color": "black", "label": "All mice"},
        )

        _ax.set_title(f"{_stim_mod} - {_selected_date_start} to {_selected_date_end}", fontsize=10)
        _ax.legend(frameon=False)
        _fig.tight_layout()
        plt.show()
    return


if __name__ == "__main__":
    app.run()
