import marimo

__generated_with = "0.25.1"
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
    import seaborn as sns
    import scipy.stats as stats

    from lecilab_behavior_analysis import utils as utils
    utils.IDIBAPS_TV_PROJECTS = "/archive/training_village/"
    from lecilab_behavior_analysis import df_transforms as dft
    from lecilab_behavior_analysis import plots
    from lecilab_behavior_analysis.figure_maker import (
        session_summary_figure,
        subject_progress_figure,
    )
    from scipy.optimize import minimize

    warnings.filterwarnings("ignore")
    return Path, dft, pd, utils


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
def _(Path, utils):
    tv_projects = utils.get_server_projects()
    print(tv_projects)
    local_root_path = Path(utils.get_outpath())
    return (local_root_path,)


@app.cell
def _(utils):
    project_idx = {
        "mk801": {"project": "COT_cannula_GAD2_data", "animals": [_a for _a in utils.get_animals_in_project("COT_cannula_GAD2_data") if "test" not in _a]},
        "dcz": {"project": "COT_cannula_data", "animals": [_a for _a in utils.get_animals_in_project("COT_cannula_data") if "test" not in _a]}
        }
    return (project_idx,)


@app.cell
def _(mo):
    download_button = mo.ui.run_button(label="Download / update this mouse")
    download_button
    return (download_button,)


@app.cell
def _(
    Path,
    download_button,
    local_root_path,
    pd,
    project_idx,
    safe_rsync_cluster_data,
    utils,
):
    for _drug in project_idx.keys():
        _project = project_idx[_drug]["project"]
        _animals = project_idx[_drug]["animals"]

        _df_all = pd.DataFrame()
        for _mouse in _animals:
            _local_path = local_root_path / _project / "sessions" / _mouse
            _local_path.mkdir(parents=True, exist_ok=True)
            if download_button.value:
                safe_rsync_cluster_data(
                    project_name=_project,
                    file_path="sessions/{}/{}.csv".format(_mouse, _mouse),
                    local_path=str(_local_path),
                    credentials=utils.get_idibaps_cluster_credentials(),
                )
            _df_all = pd.concat([_df_all, pd.read_csv(_local_path / Path(f"{_mouse}.csv"), sep=";")], ignore_index=True)
        project_idx[_drug]["df"] = _df_all
    return


@app.cell
def _(mo):
    update_processData_button = mo.ui.run_button(label="update selected mice processed data")
    update_processData_button
    return (update_processData_button,)


@app.cell
def _(dft, local_root_path, pd, project_idx, update_processData_button):
    for _drug in project_idx.keys():
        _project = project_idx[_drug]["project"]
        _df_project = project_idx[_drug]["df"]
        _animals = project_idx[_drug]["animals"]

        stim_columns = {
            "df_test_vis": "visual",
            "df_test_aud": "auditory",
        }

        for _run_mode in ["Manual", "Auto"]:
            parameters_output_dir = (local_root_path/ _project /"processed" / "parameters_for_fit" / _run_mode.lower())

            def load_or_compute_mouse_parameters(_mouse):
                parameters_output_path = (parameters_output_dir / f"{_mouse}_parameters_for_fit.pkl")

                if (parameters_output_path.exists() and not update_processData_button.value):
                    print(f"Loaded saved parameters_for_fit data: "
                        f"{parameters_output_path}")
                    return pd.read_pickle(parameters_output_path)

                _df_all_mouse = _df_project.loc[_df_project["subject"].eq(_mouse)].copy()

                _df_mode = _df_all_mouse.loc[_df_all_mouse["run_mode"].eq(_run_mode)].copy()

                _computed_results = {}
                for _output_name, _stim_name in stim_columns.items():
                    _df_stim = _df_mode.loc[_df_mode["stimulus_modality"].eq(_stim_name)].copy()

                    if _df_stim.empty:
                        _computed_results[_output_name] = _df_stim.copy()
                        print(
                            f"No {_run_mode} {_stim_name} trials found "
                            f"for {_mouse}; skipped parameters_for_fit."
                        )
                    else:
                        _computed_results[_output_name] = (
                            dft.parameters_for_fit(_df_stim)
                        )

                parameters_output_dir.mkdir(parents=True, exist_ok=True)
                pd.to_pickle(_computed_results, parameters_output_path)
                print(f"Saved parameters_for_fit data: {parameters_output_path}")
                return _computed_results

            _combined_results = {_output_name: [] for _output_name in stim_columns}

            for _mouse in _animals:
                mouse_results = load_or_compute_mouse_parameters(_mouse)

                for _output_name in stim_columns:
                    df_mouse_result = mouse_results.get(_output_name, pd.DataFrame())

                    if not df_mouse_result.empty:
                        if "subject" not in df_mouse_result.columns:
                            df_mouse_result = df_mouse_result.copy()
                            df_mouse_result["subject"] = _mouse

                        _combined_results[_output_name].append(df_mouse_result)

            _df_test_vis_raw = (
                pd.concat(_combined_results["df_test_vis"], ignore_index=True)
                if _combined_results["df_test_vis"]
                else pd.DataFrame()
            )

            _df_test_aud_raw = (
                pd.concat(_combined_results["df_test_aud"], ignore_index=True)
                if _combined_results["df_test_aud"]
                else pd.DataFrame()
            )

            _df_test_selected_raw = pd.concat(
                [_df_test_vis_raw, _df_test_aud_raw],
                ignore_index=True,
            )

            project_idx[_drug][f"df_{_run_mode.lower()}"] = _df_test_selected_raw

        del project_idx[_drug]["df"]
    return


@app.cell
def _(mo, pd, project_idx):
    _controls = {}

    for _drug, _info in project_idx.items():
        _df = _info["df_manual"]

        if _df.empty:
            continue

        _dates = sorted(pd.to_datetime(_df["year_month_day"], errors="coerce").dt.strftime("%Y-%m-%d").dropna().unique().tolist())

        if not _dates:
            continue

        _controls[_drug] = mo.ui.dictionary({
            "start": mo.ui.dropdown(
                options=_dates,
                value=_dates[0],
                label="start date",
            ),
            "end": mo.ui.dropdown(
                options=_dates,
                value=_dates[-1],
                label="end date",
            ),
        })

    manual_date_select = mo.ui.dictionary(_controls)

    mo.vstack([
        mo.hstack(
            [
                mo.md(f"**{_drug} · Manual**"),
                _selectors["start"],
                _selectors["end"],
            ],
            justify="start",
        )
        for _drug, _selectors in manual_date_select.items()
    ])
    return (manual_date_select,)


@app.cell
def _(manual_date_select, pd, project_idx):
    for _drug, _range in manual_date_select.value.items():
        _df = project_idx[_drug]["df_manual"]
        _dates = pd.to_datetime(_df["date"]).dt.date

        project_idx[_drug]["df_manual"] = _df.loc[
            _dates.between(*_range)
        ].copy()
    return


@app.cell
def _(manual_date_select, pd, project_idx):
    manual_selected_dfs = {}

    for _drug, _range in manual_date_select.value.items():
        _df = project_idx[_drug]["df_manual"]
        _dates = pd.to_datetime(_df["date"]).dt.date

        manual_selected_dfs[_drug] = _df.loc[
            _dates.between(*_range)
        ].copy()
    return


@app.cell
def _(df_day, mo):
    available_dates = sorted(
        df_day["year_month_day"].dropna().astype(str).unique().tolist()
    )
    # mix1 using time: 2026-07-18 to 2026-07-31
    # mix2 using time: 2026-08-18 to 2026-09-02
    # mix3 using time: 2026-09-03 to now
    date_select_start, date_select_end = (
        mo.ui.dropdown(
            options=available_dates,
            value=default,
            label=f"Date_{label}",
        )
        for label, default in [
            ("start", "2026-07-18"),
            ("end", available_dates[-1]),
        ]
    )
    mo.vstack([date_select_start, date_select_end])
    return


app._unparsable_cell(
    r"""
    _df_test_selected_raw = dft.add_day_column_to_df(_df_test_selected_raw.copy())
    _df_test_selected_raw = dft.add_trial_of_day_column_to_df(_df_test_selected_raw)
    _dates = pd.to_datetime(_df_test_selected_raw["year_month_day"], errors="coerce").dt.normalize()

    _start_date = pd.to_datetime(date_select_start.value).normalize()
    _end_date = pd.to_datetime(date_select_end.value).normalize()

    _df_test_selected_raw = _df_test_selected_raw[_dates.between(_start_date, _end_date, inclusive="both")].copy()project_idx
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
