import marimo

__generated_with = "0.25.2"
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
    return Path, dft, np, pd, plt, sns, utils


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
def _():
    basic_cols = [
        "subject",
        "year_month_day",
        "session",
        "trial",
        "stimulus_modality",
        "TRIAL_START",
        "first_choice_numeric"
    ]

    continuous_cols = [
        "reaction_time",
        "trial_duration",
        "inter_trial_interval",
    ]

    binary_cols = [
        "correct_numeric",
        "previous_same_choice_numeric",
        "previous_correct_numeric",
        "roa_choice_numeric",
    ]

    count_cols = [
        "fixation_breaks",
        "eager_pokes",
    ]

    features = continuous_cols + binary_cols + count_cols
    return basic_cols, binary_cols, features


@app.cell
def _(
    basic_cols,
    dft,
    features,
    local_root_path,
    pd,
    project_idx,
    update_processData_button,
):
    for _drug in project_idx.keys():
        _project = project_idx[_drug]["project"]
        _df_project = project_idx[_drug]["df"]
        _animals = project_idx[_drug]["animals"]

        stim_columns = {
            "df_test_vis": "visual",
            "df_test_aud": "auditory",
        }
        for _run_mode in ["Manual", "Auto"]:
            project_idx[_drug][f"df_{_run_mode.lower()}"] = pd.DataFrame()
            for _mouse in _animals:
                parameters_output_dir = (local_root_path/ _project /"processed" / "parameters_for_fit" / _run_mode.lower())
                parameters_output_dir.mkdir(parents=True, exist_ok=True)
                parameters_output_path = (parameters_output_dir / f"{_mouse}_parameters_for_fit.pkl")
                _df_mouse_mode = _df_project.loc[(_df_project["subject"] == _mouse) & (_df_project["run_mode"] == _run_mode)].copy()
                if (parameters_output_path.exists() and not update_processData_button.value):
                    print(f"Loaded saved parameters_for_fit data: "
                        f"{parameters_output_path}")
                    _fit_df = pd.read_pickle(parameters_output_path)
                else:
                    _df_mouse_mode_vis = _df_mouse_mode[_df_mouse_mode["stimulus_modality"] == "visual"].copy()
                    _df_mouse_mode_aud = _df_mouse_mode[_df_mouse_mode["stimulus_modality"] == "auditory"].copy()
                    _fit_parts = []
                    for _df_stim, _stim_col in [
                        (_df_mouse_mode_vis, "visual_stimulus_ratio"),
                        (_df_mouse_mode_aud, "total_evidence_strength"),
                    ]:
                        if not _df_stim.empty:
                            _fit_parts.append(
                                dft.parameters_for_fit(_df_stim).loc[
                                    :, basic_cols + features + [_stim_col]
                                ].copy()
                            )

                    if not _fit_parts:
                        continue

                    _fit_df = pd.concat(_fit_parts, ignore_index=True)
                    pd.to_pickle(_fit_df, parameters_output_path)
                    print(f"Saved parameters_for_fit data: {parameters_output_path}")

                if isinstance(_fit_df, dict):
                    _fit_parts = [
                        _fit_df[_key]
                        for _key in stim_columns
                        if _key in _fit_df and not _fit_df[_key].empty
                    ]
                    if not _fit_parts:
                        continue
                    _fit_df = pd.concat(_fit_parts, ignore_index=True)

                project_idx[_drug][f"df_{_run_mode.lower()}"] = pd.concat(
                    [project_idx[_drug][f"df_{_run_mode.lower()}"], _fit_df[features + basic_cols]],
                    ignore_index=True,
                )

        del project_idx[_drug]["df"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # stimulus select
    """)
    return


@app.cell
def _(mo):
    stimulus_select = mo.ui.dropdown(
        options=["visual", "auditory"],
        value="auditory",
        label="Stimulus modality",
    )
    stimulus_select
    return (stimulus_select,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # for auto baseline
    """)
    return


@app.cell
def _(features, project_idx, stimulus_select):

    df_mk801_auto_aud_mean = project_idx['mk801']['df_auto'][project_idx['mk801']['df_auto']['stimulus_modality'] == stimulus_select.value].groupby(
        ['subject', 'session'], observed=True)[features].mean().reset_index()
    df_dcz_auto_aud_mean = project_idx['dcz']['df_auto'][project_idx['dcz']['df_auto']['stimulus_modality'] == stimulus_select.value].groupby(
        ['subject', 'session'], observed=True)[features].mean().reset_index()
    return df_dcz_auto_aud_mean, df_mk801_auto_aud_mean


@app.cell
def _(pd):
    _injection_info_file_path = "/mnt/e/data/LeciLab/behavioral_data/data_test/manual_test_record_nuo.xlsx"
    dcz_injection_info_df = pd.read_excel(_injection_info_file_path)
    _injection_info_file_path = "/mnt/e/data/LeciLab/behavioral_data/data_test/manual_test_record_nuo_mk801.xlsx"
    mk801_injection_info_df = pd.read_excel(_injection_info_file_path)
    return dcz_injection_info_df, mk801_injection_info_df


@app.cell
def _(dcz_injection_info_df, mk801_injection_info_df, pd, project_idx):
    for _drug in ["dcz", "mk801"]:
        if _drug == "dcz":
            _injection_info_df = dcz_injection_info_df
        elif _drug == "mk801":
            _injection_info_df = mk801_injection_info_df
        _date_col = _injection_info_df.columns[0]
        _injection_info = _injection_info_df.copy()

        _injection_info["_injection_date"] = pd.to_datetime(
            _injection_info[_date_col],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

        _injection_long = _injection_info.melt(
            id_vars="_injection_date",
            value_vars=list(_injection_info_df.columns[1:]),
            var_name="_mouse",
            value_name="_observations",
        )
        _injection_long["_mouse"] = _injection_long["_mouse"].astype(str)

        injection_lookup = (
            _injection_long
            .set_index(["_injection_date", "_mouse"])["_observations"]
            .to_dict()
        )

        _df = project_idx[_drug]["df_manual"].copy()

        _trial_dates = pd.to_datetime(
            _df["year_month_day"],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

        _trial_mice = _df["subject"].astype(str)

        _df["observations"] = [
            injection_lookup.get((_date, _mouse), pd.NA)
            for _date, _mouse in zip(_trial_dates, _trial_mice)
        ]

        project_idx[_drug]["df_manual"] = _df
    return


@app.cell
def _(project_idx):
    def observation_group(observation):
        observation = str(observation).lower()
        if "dcz" in observation:
            return "dcz"
        if "mk801" in observation:
            return "mk801"
        if "saline" in observation:
            return "saline"
        return None

    for _drug in project_idx:
        project_idx[_drug]["df_manual"]["observation_group"] = project_idx[_drug]["df_manual"]['observations'].apply(observation_group)
        project_idx[_drug]["df_manual"] = project_idx[_drug]["df_manual"].dropna(subset=["observation_group"])
    return


@app.cell
def _(project_idx):
    hM3Dq_mice = ['NUO062', 'NUO063', 'NUO064', 'NUO065', 'NUO007', 'NUO008', 'NUO009', 'NUO010', 'NUO011', 'NUO012']
    hM4Di_mice = ['NUO057', 'NUO058', 'NUO059', 'NUO060', 'NUO061', 'NUO001', 'NUO002', 'NUO003', 'NUO005', 'NUO006']

    _mouse_groups = {
        **dict.fromkeys(hM3Dq_mice, "hM3Dq"),
        **dict.fromkeys(hM4Di_mice, "hM4Di"),
    }

    project_idx['dcz']["df_manual"]["mice_group"] = project_idx['dcz']["df_manual"]["subject"].map(_mouse_groups)
    return


@app.cell
def _(features, project_idx, stimulus_select):
    df_mk801_manual_aud_mean = project_idx['mk801']['df_manual'][(project_idx['mk801']['df_manual']['stimulus_modality'] == stimulus_select.value) & (project_idx['mk801']['df_manual']['observation_group'] == 'mk801')].groupby(
        ['subject', 'session'], observed=True)[features].mean().reset_index()
    df_dcz_manual_aud_mean = project_idx['dcz']['df_manual'][(project_idx['dcz']['df_manual']['stimulus_modality'] == stimulus_select.value) & (project_idx['dcz']['df_manual']['observation_group'] == 'dcz')].groupby(
        ['subject', 'session', 'mice_group'], observed=True)[features].mean().reset_index()
    return df_dcz_manual_aud_mean, df_mk801_manual_aud_mean


@app.cell
def _(np):
    def get_drug_effect(drug_session_mean, control_session_mean, features):
        drug_mean = (
            drug_session_mean.groupby("subject", observed=True)[features]
            .mean()
        )
        control_mean = (
            control_session_mean.groupby("subject", observed=True)[features]
            .mean()
        )

        # 只比较两边都有记录的鼠
        common_mice = drug_mean.index.intersection(control_mean.index)
        delta = (
            drug_mean.loc[common_mice]
            - control_mean.loc[common_mice]
        )

        # 与原函数一致：用 control 的 session-level SD 标准化
        control_sd = (
            control_session_mean[features]
            .std()
            .replace(0, np.nan)
        )
        delta_z = delta.div(control_sd, axis="columns")

        return delta, delta_z

    return (get_drug_effect,)


@app.cell
def _(
    df_dcz_auto_aud_mean,
    df_dcz_manual_aud_mean,
    df_mk801_auto_aud_mean,
    df_mk801_manual_aud_mean,
    features,
    get_drug_effect,
):
    delta_mk801, delta_z_mk801 = get_drug_effect(
        df_mk801_manual_aud_mean,
        df_mk801_auto_aud_mean,
        features,
    )

    delta_dcz_hm3, delta_z_dcz_hm3 = get_drug_effect(
        df_dcz_manual_aud_mean[df_dcz_manual_aud_mean['mice_group']=='hM3Dq'],
        df_dcz_auto_aud_mean,
        features,
    )
    delta_dcz_hm4, delta_z_dcz_hm4 = get_drug_effect(
        df_dcz_manual_aud_mean[df_dcz_manual_aud_mean['mice_group']=='hM4Di'],
        df_dcz_auto_aud_mean,
        features,
    )
    return delta_z_dcz_hm3, delta_z_dcz_hm4, delta_z_mk801


@app.cell
def _(delta_z_dcz_hm3, delta_z_dcz_hm4, delta_z_mk801, pd, plt, sns):
    _fig, _axes = plt.subplots(
        1, 3,
        figsize=(17, 6),
        sharey=False
    )

    for _ax, _data, _title in zip(
        _axes,
        [delta_z_dcz_hm3, delta_z_dcz_hm4, delta_z_mk801],
        ["hM3Dq + DCZ", "hM4Di + DCZ", "MK-801"]
    ):
        _data = _data.apply(pd.to_numeric, errors="raise").astype(float)
        sns.heatmap(
            _data,
            ax=_ax,
            cmap="coolwarm",
            center=0,
            vmin=-2,
            vmax=2,
            annot=True,
            fmt=".2f",
        )

        _ax.set_title(_title)
        _ax.set_xlabel("Behavioral variables")
        _ax.set_ylabel("Mouse")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(delta_z_dcz_hm3, delta_z_dcz_hm4, delta_z_mk801, pd, plt, sns):
    effect_matrix = pd.DataFrame({
        "hM3Dq + DCZ": delta_z_dcz_hm3.mean(axis=0),
        "hM4Di + DCZ": delta_z_dcz_hm4.mean(axis=0),
        "MK-801": delta_z_mk801.mean(axis=0),
    }).T

    effect_matrix = effect_matrix.apply(
        pd.to_numeric, errors="raise"
    ).astype(float)

    _fig, _ax = plt.subplots(figsize=(12, 4))

    sns.heatmap(
        effect_matrix,
        ax=_ax,
        cmap="coolwarm",
        center=0,
        vmin=-2,
        vmax=2,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Mean standardized difference"},
    )

    _ax.set_title("Standardized Drug Effects")
    _ax.set_xlabel("Behavioral variables")
    _ax.set_ylabel("")
    _ax.tick_params(axis="y", rotation=0)
    _ax.tick_params(axis="x", rotation=45)

    _fig.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GLM
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # factor analysis
    """)
    return


@app.cell
def _():
    from factor_analyzer import FactorAnalyzer
    from latentcor import latentcor

    return FactorAnalyzer, latentcor


@app.cell
def _(
    binary_cols,
    features,
    latentcor,
    np,
    pd,
    plt,
    project_idx,
    sns,
    stimulus_select,
):

    X = pd.concat(
        [
            _info["df_auto"].loc[
                _info["df_auto"]["stimulus_modality"] == stimulus_select.value,
                features,
            ]
            for _info in project_idx.values()
        ],
        ignore_index=True,
    ).dropna().copy()

    # 与 DataFrame 列顺序一一对应
    types = ["bin" if _col in binary_cols else "con" for _col in features]

    # 检查 binary 数据
    for _col in binary_cols:
        assert set(X[_col].unique()).issubset({0, 1})

    # 计算 latent correlation matrix
    result = latentcor(
        X.to_numpy(dtype=float),
        tps=types,
        method='original',
    )

    corr = pd.DataFrame(
        np.asarray(result["R"]),
        index=features,
        columns=features,
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
    )

    plt.title("Latent Correlation Matrix of Features")
    plt.show()
    return (corr,)


@app.cell
def _(corr, np, plt):
    eigenvalues = np.linalg.eigvalsh(corr.values)[::-1]

    plt.figure(figsize=(6, 4))
    plt.plot(
        range(1, len(eigenvalues) + 1),
        eigenvalues,
        "o-",
    )

    plt.axhline(1, color="gray", linestyle="--")

    plt.xlabel("Factor / Component Number")
    plt.ylabel("Eigenvalue")
    plt.title("Scree Plot")
    plt.xticks(range(1, len(eigenvalues) + 1))
    plt.show()
    return


@app.cell
def _(FactorAnalyzer, corr, features, pd):
    fa = FactorAnalyzer(
        n_factors=4,
        rotation="oblimin",
        method="minres",
        is_corr_matrix=True,
    )

    fa.fit(corr)

    loadings = pd.DataFrame(
        fa.loadings_,
        index=features,
        columns=["Factor1", "Factor2", "Factor3", "Factor4"],
    )

    print(loadings)
    return (loadings,)


@app.cell
def _(loadings, plt, sns):
    sns.heatmap(
        loadings,
        annot=True,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
    )

    plt.title("EFA Factor Loadings")
    plt.show()
    return


@app.cell
def _():
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


if __name__ == "__main__":
    app.run()
