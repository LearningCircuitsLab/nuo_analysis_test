import marimo

__generated_with = "0.23.15"
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
    _default_project = 'COT_unilateral_sound_data'
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
    return


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
    pd,
    project,
    safe_rsync_cluster_data,
    utils,
):
    local_path = Path(utils.get_outpath()) / project / "sessions" 
    local_path.mkdir(parents=True, exist_ok=True)


    df_all = pd.DataFrame()
    for _mouse in animals:
        _local_path = Path(utils.get_outpath()) / project / "sessions" / _mouse
        _local_path.mkdir(parents=True, exist_ok=True)
        if download_button.value:
            safe_rsync_cluster_data(
                project_name=project,
                file_path="sessions/{}/{}.csv".format(_mouse, _mouse),
                local_path=str(_local_path),
                credentials=utils.get_idibaps_cluster_credentials(),
            )
        df_all = pd.concat([df_all, pd.read_csv(_local_path / Path(f"{_mouse}.csv"), sep=";")], ignore_index=True)
    return (df_all,)


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
def _(df, dft, np):
    df_day = dft.add_day_column_to_df(df.copy())
    df_day = dft.add_trial_of_day_column_to_df(df_day)
    df_day["total_trial"] = np.arange(1, df_day.shape[0] + 1)

    available_dates = sorted(
        df_day["year_month_day"].dropna().astype(str).unique().tolist()
    )
    return (df_day,)


@app.cell
def _(pd):
    injection_info_file_path = '/home/nuo/Documents/data/LeciLab/behavioral_data/manual_record/manual_test_record_nuo_muscimol_localInjection.xlsx'
    injection_info_df = pd.read_excel(injection_info_file_path)
    injection_info_df
    return (injection_info_df,)


@app.cell
def _(injection_info_df, pd):
    injection_long = injection_info_df.melt(
        id_vars=[injection_info_df.columns[0]],
        var_name="subject",
        value_name="injection",
    ).rename(columns={injection_info_df.columns[0]: "injection_date"})

    injection_long["injection_date"] = pd.to_datetime(
        injection_long["injection_date"], errors="coerce"
    ).dt.normalize()

    injection_long["subject"] = (
        injection_long["subject"].astype("string").str.strip()
    )
    injection_long["injection"] = (
        injection_long["injection"].astype("string").str.strip()
    )

    injection_long
    return (injection_long,)


@app.cell
def _(injection_long, mo):
    _injection_values = sorted(injection_long['injection'].unique())

    injection_select = mo.ui.dropdown(
        options=_injection_values,
        value=_injection_values[0],
        label="Injection condition",
        full_width=True,
    )

    injection_select
    return (injection_select,)


@app.cell
def _(injection_long, injection_select):
    selected_mouse_dates = (
        injection_long.loc[
            injection_long["injection"].eq(injection_select.value),
            ["subject", "injection_date"],
        ]
        .dropna()
        .drop_duplicates()
    )
    return (selected_mouse_dates,)


@app.cell
def _(df_day, pd, selected_mouse_dates):
    _day_keys = pd.MultiIndex.from_arrays([
        df_day["subject"].astype("string").str.strip(),
        pd.to_datetime(df_day["year_month_day"], errors="coerce").dt.normalize(),
    ])

    _selected_keys = pd.MultiIndex.from_frame(selected_mouse_dates)

    df_day_injection = df_day.loc[
        _day_keys.isin(_selected_keys)
    ].copy()

    df_day_injection
    return (df_day_injection,)


@app.cell
def _(df_day_injection, dft, injection_select, plots, plt):
    _mice = df_day_injection["subject"].dropna().unique()
    _window = 50

    # 每只鼠一行，每行三个图
    _fig, _all_axes = plt.subplots(
        nrows=len(_mice),
        ncols=3,
        figsize=(18, 5 * len(_mice)),
        squeeze=False,  # 只有一只鼠时也保持二维 axes
    )

    for _row, _mouse in enumerate(_mice):
        _df_mouse = df_day_injection.loc[
            df_day_injection["subject"].eq(_mouse)
        ].copy()

        _axes = _all_axes[_row]

        # 图 1：正确反应数量
        plots.number_of_correct_responses_plot(
            _df_mouse,
            ax=_axes[0],
        )

        # 图 2：repeat / alternate 正确率
        _roap = dft.get_performance_through_trials(
            _df_mouse.copy(),
            window=_window,
        ).reset_index(drop=True)

        _roap["total_trial"] = range(1, len(_roap) + 1)

        _session_changes = _roap.index[
            _roap["session"].ne(_roap["session"].shift())
        ]

        _roap["repeat_or_alternate"] = (
            dft.get_repeat_or_alternate_series(_roap["correct_side"])
        )
        _roap = dft.get_repeat_or_alternate_performance(
            _roap,
            window=_window,
        )

        plots.repeat_or_alternate_performance_plot(
            _roap,
            ax=_axes[1],
            session_changes=_session_changes,
        )

        # 图 3：psychometric 曲线
        _psych_df = dft.get_performance_by_difficulty_ratio(_df_mouse)

        plots.psychometric_plot(
            _psych_df,
            x="total_evidence_strength",
            y="first_choice_numeric",
            valueType="continus",
            ax=_axes[2],
        )

        # 每一行的鼠名，放在中间子图上方；留出图例空间
        _axes[1].set_title(
            f"Mouse: {_mouse}",
            fontsize=14,
            pad=40,
        )

    # 整张图的标题：注射条件
    _fig.suptitle(str(injection_select.value), fontsize=16, fontweight="bold",)
    _fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=4)
                                                                                                                                                                                                                                                         
    _fig
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
    return


@app.cell
def _(mo):
    update_processData_button = mo.ui.run_button(label="update selected mice processed data")
    update_processData_button
    return


if __name__ == "__main__":
    app.run()
