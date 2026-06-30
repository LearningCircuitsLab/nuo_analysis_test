import numpy as np
import pandas as pd
import ssm
import matplotlib.pyplot as plt
import lecilab_behavior_analysis.df_transforms as dft
import lecilab_behavior_analysis.utils as utils
import matplotlib.pyplot as plt
import lecilab_behavior_analysis.utils as behavior_utils


def rsync_specific_file(file_path: str, credentials: dict, local_path: str) -> bool:
    """
    Sync one remote file or glob pattern to a local folder.

    This is a safer version of lecilab_behavior_analysis.utils.rsync_specific_file:
    it captures stderr/stdout and returns False on rsync failure instead of raising
    an AttributeError while printing the error.
    """
    import os
    import subprocess

    os.makedirs(local_path, exist_ok=True)
    remote_path = f"{credentials['username']}@{credentials['host']}:{file_path}"
    result = subprocess.run(
        ["rsync", "-avz", remote_path, str(local_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error syncing data for {file_path}")
        if result.stderr:
            print(result.stderr.strip())
        if result.stdout:
            print(result.stdout.strip())
        return False

    return True


def add_lag_features(df, source_col, n_lags=10, group_col="session"):
    df = df.copy()
    for lag in range(1, n_lags + 1):
        df[f"{source_col}_lag{lag}"] = (
            df.groupby(group_col)[source_col].shift(lag)
        )
    return df


def recursive_kernel_prior(history, alpha, pi0=0.5):
    """
    history: 1D array-like of 0/1 values for updating the prior

    alpha: float in (0, 1)
    pi0: initial prior, usually set to 0.5

    returns
    -------
    pi : np.ndarray, shape (T,)
         pi[t] represents the prior before the t-th trial
    """
    history = np.asarray(history, dtype=float)
    T = len(history)
    pi = np.empty(T, dtype=float)

    # Before the first trial, there is no history, so use the initial value
    pi[0] = pi0

    # From the second trial onwards, update using the previous trial's history
    for t in range(1, T):
        pi[t] = (1 - alpha) * pi[t - 1] + alpha * history[t - 1]

    return pi

def logit(p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _infer_easy_stage(df, easy=None, stage=None):
    if easy is not None:
        return bool(easy)

    stage_values = []
    if stage is not None:
        stage_values.append(stage)

    for stage_col in [
        "selected_df_option",
        "current_training_stage",
        "training_stage",
    ]:
        if stage_col in df.columns:
            stage_values.extend(df[stage_col].dropna().astype(str).unique())

    stage_text = " ".join(str(value).lower() for value in stage_values)
    if "easy" in stage_text:
        return True
    if "hard" in stage_text:
        return False
    return False


def _fit_ordinary_logistic_curve(x, y, xs):
    from scipy.optimize import minimize

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    xs = np.asarray(xs, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return np.full_like(xs, np.nan, dtype=float)
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.full_like(xs, np.nanmean(y), dtype=float)

    x_mean = np.nanmean(x)
    x_scale = np.nanstd(x)
    if not np.isfinite(x_scale) or x_scale == 0:
        x_scale = 1.0

    x_z = (x - x_mean) / x_scale
    xs_z = (xs - x_mean) / x_scale
    y_mean = np.clip(np.nanmean(y), 1e-6, 1 - 1e-6)
    initial_params = [np.log(y_mean / (1 - y_mean)), 1.0]

    def nll(params):
        p = sigmoid(params[0] + params[1] * x_z)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    result = minimize(nll, initial_params, method="BFGS")
    params = result.x
    return sigmoid(params[0] + params[1] * xs_z)


def psychometric_plot_easy_logistic(
    df: pd.DataFrame,
    x,
    y,
    ax=None,
    point_kwargs=None,
    line_kwargs=None,
    valueType="discrete",
    bins=6,
    log=True,
    easy=None,
    stage=None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        ax = plt.gca()

    use_easy_logistic = _infer_easy_stage(df, easy=easy, stage=stage)

    default_point_kwargs = {
        "color": "cornflowerblue",
        "markers": "o",
        "errorbar": ("ci", 95),
        "label": "Observed Choices",
        "native_scale": True,
        "linestyles": "",
        "markersize": 5,
        "capsize": 0.01,
        "alpha": 1.0,
    }
    if point_kwargs is None:
        point_kwargs = {}
    point_kwargs = {**default_point_kwargs, **point_kwargs}

    default_line_kwargs = {
        "color": "lightcoral",
        "label": (
            "Logistic Fit"
            if use_easy_logistic
            else "Lapse Logistic Fit (Independent)"
        ),
        "linestyle": "-",
        "alpha": 1.0,
    }
    if line_kwargs is None:
        line_kwargs = {}
    line_kwargs = {**default_line_kwargs, **line_kwargs}

    behavior_utils.column_checker(df, required_columns={x, y})
    df_copy = df.copy(deep=True)
    df_copy = df_copy.dropna(subset=[x, y])
    if df_copy.empty:
        ax.text(0.5, 0.5, f"No valid {x}/{y} values", ha="center")
        ax.set_axis_off()
        return ax

    if valueType == "discrete":
        df_copy[x + "_fit"] = df_copy[x].copy()
        if 0 in df_copy[x].unique():
            df_copy[x + "_fit"].replace(0, 0.0001, inplace=True)
        if log:
            df_copy[x + "_fit"] = np.sign(df_copy[x + "_fit"]) * (
                np.log(abs(df_copy[x + "_fit"])).round(4)
            )
            ax.set_xlabel("log_" + x)
        else:
            ax.set_xlabel(x)
    else:
        nbins = min(bins, df_copy[x].nunique())
        if nbins < 1:
            ax.text(0.5, 0.5, f"No valid {x} values", ha="center")
            ax.set_axis_off()
            return ax
        bin_groups = pd.cut(df_copy[x], bins=nbins)
        bin_means = df_copy.groupby(bin_groups, observed=True)[x].mean()
        df_copy[x + "_fit"] = bin_groups.map(bin_means).astype(float)
        ax.set_xlabel(x + " (binned)")

    sns.pointplot(
        x=x + "_fit",
        y=y,
        data=df_copy,
        estimator=lambda values: np.mean(values),
        ax=ax,
        **point_kwargs,
    )

    xs = np.linspace(
        df_copy[x + "_fit"].min(),
        df_copy[x + "_fit"].max(),
        100,
    ).reshape(-1, 1)
    if use_easy_logistic:
        p_left = _fit_ordinary_logistic_curve(
            df_copy[x + "_fit"],
            df_copy[y],
            xs,
        )
    else:
        p_left, _ = behavior_utils.fit_lapse_logistic_independent(
            df_copy[x + "_fit"],
            df_copy[y],
        )
    ax.plot(xs, p_left, **line_kwargs)

    if y == "first_choice_numeric":
        ax.set_ylabel("P(Leftward Choices)")
    elif y == "correct_choice_numeric":
        ax.set_ylabel("P(Correct Choices)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.get_legend().get_frame().set_linewidth(0.0)
    ax.get_legend().set_frame_on(False)
    return ax


def filter_variables_for_model(df_fit: pd.DataFrame, X: list, y: str):
    """
    Filter variables for a logistic regression model from one combined dataframe.

    The output matches lecilab_behavior_analysis.utils.filter_variables_for_model:
    one correlation matrix per subject and one normalized-contribution dataframe
    with subjects as columns.
    """
    corr_mat_list = []
    norm_contribution_df = pd.DataFrame([])

    missing_cols = [col for col in list(X) + [y] if col not in df_fit.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in df_fit: {missing_cols}")

    if "analysis_mouse" in df_fit.columns:
        subject_col = "analysis_mouse"
    elif "subject" in df_fit.columns:
        subject_col = "subject"
    else:
        subject_col = None

    if subject_col is None:
        grouped_dfs = [("all_data", df_fit)]
    else:
        grouped_dfs = df_fit.groupby(subject_col, sort=True)

    for df_name, df_for_fit in grouped_dfs:

        try:
            corr_fit_X_df = df_for_fit[X].corr()
            norm_contribution = behavior_utils.drop_one_var_contribution(
                df_for_fit,
                x_cols=X,
                y_col=y,
                method="bfgs",
            )
        except Exception as exc:
            print(f"[skip] {df_name}: {exc}")
            continue

        corr_mat_list.append(corr_fit_X_df)
        norm_contribution_df[df_name] = norm_contribution

    return corr_mat_list, norm_contribution_df


def choice_prob_from_prior(stim, pi, beta, w, bias, lapse):
    dv = bias + beta * stim + w * logit(pi)
    p = sigmoid(dv)
    p = (1 - lapse) * p + lapse * 0.5
    return np.clip(p, 1e-9, 1 - 1e-9)

def session_loglik_kernel(theta, sess_df, prior_type="stim"):
    alpha = theta["alpha"]
    beta  = theta["beta"]
    w     = theta["w"]
    bias  = theta["bias"]
    lapse = theta["lapse"]

    if sess_df['stimulus_modality'].iloc[0] == "visual":
        x = sess_df["visual_stimulus_ratio"].to_numpy(dtype=float)
    elif sess_df['stimulus_modality'].iloc[0] == "auditory":
        x = sess_df["total_evidence_strength"].to_numpy(dtype=float)
    else:
        raise ValueError("Unknown stimulus modality: {}".format(sess_df['stimulus_modality'].iloc[0]))
   
    if sess_df["first_choice_numeric"].isna().any():
        raise ValueError("NaN in first_choice_numeric")
    

    if prior_type == "stim":
        history = sess_df["correct_side_numeric"]
        if history.isnull().any():
            raise ValueError("NaN in correct_side_numeric")
    elif prior_type == "act":
        history = sess_df["first_choice_numeric"]
    y = sess_df["first_choice_numeric"].to_numpy(dtype=int)

    pi = recursive_kernel_prior(history, alpha=alpha, pi0=0.5)
    p = choice_prob_from_prior(x, pi, beta=beta, w=w, bias=bias, lapse=lapse)

    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return ll

def dataset_loglik(theta, df, prior_type="stim"):
    total_ll = 0.0
    for _, sess_df in df.groupby("session", sort=False):
        total_ll += session_loglik_kernel(theta, sess_df, prior_type=prior_type)
    return total_ll

def unpack_params(v):
    alpha = 1 / (1 + np.exp(-v[0]))
    beta  = v[1]
    w     = v[2]
    bias  = v[3]
    lapse = 1 / (1 + np.exp(-v[4]))
    return {"alpha": alpha, "beta": beta, "w": w, "bias": bias, "lapse": lapse}

def neg_loglik_reg(v, df, prior_type):
    theta = unpack_params(v)
    ll = dataset_loglik(theta, df, prior_type=prior_type)

    # add regularization to make sure the parameters don't grow too large, which can lead to overfitting and numerical instability
    reg = 1 * (theta["beta"]**2 + theta["w"]**2 + theta["bias"]**2)

    return -(ll - reg)

def build_glmhmm_inputs_by_session(
    df: pd.DataFrame,
    y_col="first_choice_numeric",
    stim_col=["visual_stimulus_ratio"],
):
    """
    Returns:
        datas: list of (T_i, 1)
        inpts: list of (T_i, M)
    """
    datas, inpts = [], []

    add_bias = "bias" in stim_col
    real_cols = [c for c in stim_col if c != "bias"]

    for _, df_sess in df.groupby("session", sort=False):
        df_sess = df_sess[real_cols + [y_col]].dropna()

        y = df_sess[y_col].to_numpy(dtype=int).reshape(-1, 1)
        X = df_sess[real_cols].to_numpy(dtype=float)

        if add_bias:
            X = np.column_stack([X, np.ones(len(X))])

        datas.append(y)
        inpts.append(X)

    return datas, inpts

def fit_glmhmm(obs, 
                inputs,
                num_states = 3,        # number of discrete states
                obs_dim = 1,           # number of observed dimensions
                num_categories = 2,    # number of categories for output
                input_dim = 2,         # input dimensions 
                N_iters = 100,         # number of EM iterations
                prior_sigma=2,
                prior_alpha=2,
                kappa=1
                ):
    
    map_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
            observation_kwargs=dict(C=num_categories,prior_sigma=prior_sigma),
            transitions="sticky", 
            transition_kwargs=dict(alpha=prior_alpha,kappa=kappa),
            )
    hmm_lls = map_glmhmm.fit(obs, inputs=inputs, method="em", num_iters=N_iters, tolerance=10**-4)
    ll = map_glmhmm.log_probability(obs, inputs=inputs)
    return map_glmhmm, ll, hmm_lls

def fit_best_glmhmm(obs, 
                    inputs,
                    num_states = 3,        # number of discrete states
                    obs_dim = 1,           # number of observed dimensions
                    num_categories = 2,    # number of categories for output
                    input_dim = 2,         # input dimensions 
                    n_restarts = 10,        # number of EM iterations
                    N_iters = 100,         # number of EM iterations
                    prior_sigma=2,
                    prior_alpha=2,
                    kappa=1
                    ):
    
    best_hmm, best_ll = None, -np.inf

    for r in range(n_restarts):
        map_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
             observation_kwargs=dict(C=num_categories,prior_sigma=prior_sigma),
             transitions="sticky", 
             transition_kwargs=dict(alpha=prior_alpha,kappa=kappa),
             )
        hmm_lls = map_glmhmm.fit(obs, inputs=inputs, method="em", num_iters=N_iters, tolerance=10**-4)
        ll = map_glmhmm.log_probability(obs, inputs=inputs)
        if ll > best_ll:
            best_ll = ll
            best_hmm = map_glmhmm
            best_hmm_lls = hmm_lls
    return best_hmm, best_ll, best_hmm_lls


def _state_colors(n_states, colors=None):
    if colors is not None and len(colors) >= n_states:
        return list(colors)

    import matplotlib.pyplot as plt

    default_colors = [plt.get_cmap("tab10")(i % 10) for i in range(n_states)]
    if colors is None:
        return default_colors

    colors = list(colors)
    return colors + default_colors[len(colors):]


def plot_transition_matrix(map_glmhmm, title="", ax=None, cmap="gray", fontsize=8):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4))

    A = map_glmhmm.transitions.transition_matrix
    im = ax.imshow(A, vmin=0, vmax=1, aspect="auto", cmap=cmap)

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("State t")
    ax.set_ylabel("State t-1")

    K = A.shape[0]
    for i in range(K):
        for j in range(K):
            val = A[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="black" if val > 0.5 else "white",
                fontsize=fontsize,
            )

    return ax, im


def plot_transition_graph(map_glmhmm, ax=None, colors=None, fontsize=8):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4))

    A = np.asarray(map_glmhmm.transitions.transition_matrix)
    K = A.shape[0]
    colors = _state_colors(K, colors)
    node_radius = 0.105 if K <= 4 else 0.08

    if K == 3:
        positions = np.array(
            [
                [0.50, 0.82],
                [0.18, 0.25],
                [0.82, 0.25],
            ]
        )
    else:
        theta = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, K, endpoint=False)
        positions = np.column_stack(
            [0.5 + 0.35 * np.cos(theta), 0.5 + 0.35 * np.sin(theta)]
        )

    center = positions.mean(axis=0)

    for i in range(K):
        for j in range(K):
            prob = A[i, j]
            lw = 0.6 + 3.0 * prob
            alpha = 0.25 + 0.65 * prob
            edge_color = colors[i]

            if i == j:
                pos = positions[i]
                outward = pos - center
                norm = np.linalg.norm(outward)
                if norm == 0:
                    outward = np.array([0.0, 1.0])
                else:
                    outward = outward / norm

                tangent = np.array([-outward[1], outward[0]])
                start = pos + outward * node_radius * 1.15 - tangent * node_radius * 0.65
                end = pos + outward * node_radius * 1.15 + tangent * node_radius * 0.65
                label_pos = pos + outward * node_radius * 2.7

                arrow = FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    mutation_scale=8,
                    connectionstyle="arc3,rad=1.6",
                    linewidth=lw,
                    color=edge_color,
                    alpha=alpha,
                    zorder=1,
                )
                ax.add_patch(arrow)
                ax.text(
                    label_pos[0],
                    label_pos[1],
                    f"{prob:.2f}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1),
                    zorder=4,
                )
                continue

            start = positions[i]
            end = positions[j]
            vector = end - start
            dist = np.linalg.norm(vector)
            if dist == 0:
                continue

            unit = vector / dist
            perp = np.array([-unit[1], unit[0]])
            lane_offset = node_radius * 0.35
            arrow_start = start + unit * node_radius * 1.25 + perp * lane_offset
            arrow_end = end - unit * node_radius * 1.25 + perp * lane_offset
            rad = 0.10

            arrow = FancyArrowPatch(
                arrow_start,
                arrow_end,
                arrowstyle="-|>",
                mutation_scale=8,
                connectionstyle=f"arc3,rad={rad}",
                linewidth=lw,
                color=edge_color,
                alpha=alpha,
                zorder=1,
            )
            ax.add_patch(arrow)

            mid = (arrow_start + arrow_end) / 2
            label_pos = mid + perp * node_radius * 0.45
            ax.text(
                label_pos[0],
                label_pos[1],
                f"{prob:.2f}",
                ha="center",
                va="center",
                fontsize=fontsize,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1),
                zorder=4,
            )

    for i, pos in enumerate(positions):
        node = Circle(
            pos,
            node_radius,
            facecolor=colors[i],
            edgecolor="black",
            linewidth=1,
            alpha=0.9,
            zorder=3,
        )
        ax.add_patch(node)
        ax.text(
            pos[0],
            pos[1],
            f"State {i}",
            ha="center",
            va="center",
            fontsize=fontsize,
            color="white",
            fontweight="bold",
            zorder=5,
        )

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.12, 1.15)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


def get_posterior_probs_concat(map_glmhmm, datas, inpts):
    """Return one posterior probability array per session."""
    return [
        map_glmhmm.expected_states(data=data, input=inpt)[0]
        for data, inpt in zip(datas, inpts)
    ]


def add_df_glmhmm_state(df, post_prob_list):
    """Add MAP GLM-HMM state labels to a dataframe."""
    posterior_probs_con = np.concatenate(post_prob_list, axis=0)
    if len(df) != len(posterior_probs_con):
        raise ValueError(
            "df and posterior probabilities have different lengths: "
            f"{len(df)} vs {len(posterior_probs_con)}"
        )

    df = df.copy()
    df["glmhmm_state"] = np.argmax(posterior_probs_con, axis=1)
    return df


def add_performance_by_session(df, session_col="session"):
    

    df = df.copy()
    if df.empty or session_col not in df.columns:
        return df

    if "performance_w" not in df.columns:
        df["performance_w"] = np.nan

    for _, df_session in df.groupby(session_col, sort=False):
        try:
            df_session = dft.get_performance_through_trials(df_session.copy())
        except Exception:
            continue
        if "performance_w" in df_session.columns:
            df.loc[df_session.index, "performance_w"] = (
                df_session["performance_w"].values
            )

    return df


def plot_posteriors_with_performance(
    df,
    post_prob_list,
    colors=None,
    ax=None,
    lw=2,
    perf_scale=0.01,
):

    posterior_probs_con = np.concatenate(post_prob_list, axis=0)
    K = posterior_probs_con.shape[1]
    colors = _state_colors(K, colors)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(20, 2.5), dpi=80)

    for k in range(K):
        ax.plot(
            posterior_probs_con[:, k],
            label=f"State {k + 1}",
            lw=lw,
            color=colors[k],
        )

    df = add_performance_by_session(df)
    performance = pd.to_numeric(df["performance_w"], errors="coerce").to_numpy()
    if np.isfinite(performance).any():
        if np.nanmax(performance) > 1:
            performance = performance * perf_scale
        ax.plot(
            performance,
            "-k",
            label="Performance",
            alpha=0.7,
        )

    ax.set_ylim(-0.01, 1.01)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("trial #")
    ax.set_ylabel("p(state)")
    ax.legend(frameon=False)
    return ax


def plot_param_by_state(
    hmm,
    df,
    X,
    y,
    colors=None,
    state_col="glmhmm_state",
    marker="o",
    linestyle="-",
    ax=None,
):


    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = _state_colors(hmm.K, colors)
    weights = np.asarray(hmm.observations.params).squeeze()
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)

    if weights.ndim != 2 or weights.shape[0] != hmm.K:
        ax.text(
            0.5,
            0.5,
            f"Unexpected GLM-HMM weight shape: {weights.shape}",
            ha="center",
        )
        ax.set_axis_off()
        return ax

    param_cols = list(X)
    if "bias" in param_cols:
        param_cols = [col for col in param_cols if col != "bias"] + ["bias"]
    if weights.shape[1] == len(param_cols) + 1 and "bias" not in param_cols:
        param_cols = param_cols + ["bias"]
    elif weights.shape[1] != len(param_cols):
        param_cols = [f"input_{idx}" for idx in range(weights.shape[1])]

    df_param_states = pd.DataFrame(
        weights.T,
        index=param_cols,
        columns=range(hmm.K),
    )
    if df_param_states.empty:
        ax.text(0.5, 0.5, "No GLM-HMM observation weights", ha="center")
        ax.set_axis_off()
        return ax

    for state in df_param_states.columns:
        ax.plot(
            df_param_states.index,
            df_param_states[state],
            marker=marker,
            linestyle=linestyle,
            color=colors[state],
            label=f"State {state}",
        )

    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Weights")
    ax.legend(title="State", frameon=False)
    return ax


def plot_psychometric_by_state(
    df,
    map_glmhmm,
    colors=None,
    x="visual_stimulus_ratio",
    y="first_choice_numeric",
    state_col="glmhmm_state",
    ax=None,
    valueType="discrete",
    point_kwargs=None,
    line_kwargs=None,
    log=True,
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))

    required_cols = [x, y, state_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        ax.text(0.5, 0.5, f"Missing columns: {missing_cols}", ha="center")
        ax.set_axis_off()
        return ax

    colors = _state_colors(map_glmhmm.K, colors)
    point_kwargs = point_kwargs or {}
    line_kwargs = line_kwargs or {}

    for state in range(map_glmhmm.K):
        df_s = df[df[state_col] == state].copy()
        if df_s.empty:
            continue

        psychometric_plot_easy_logistic(
            df_s,
            x=x,
            y=y,
            ax=ax,
            valueType=valueType,
            log=log,
            point_kwargs={
                "color": colors[state],
                "label": "",
                **point_kwargs,
            },
            line_kwargs={
                "color": colors[state],
                "label": f"state {state}",
                **line_kwargs,
            },
        )

    ax.legend(frameon=False)
    return ax


def plot_glmhmm_pipeline_figure(
    map_glmhmm,
    df,
    datas,
    inpts,
    input_cols,
    y_col="first_choice_numeric",
    psychometric_x="model_stimulus",
    colors=None,
    title=None,
    psychometric_value_type="discrete",
):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    colors = _state_colors(map_glmhmm.K, colors)

    fig = plt.figure(figsize=(10, 16), dpi=300)
    gs = GridSpec(
        nrows=3,
        ncols=2,
        figure=fig,
        height_ratios=[1.0, 0.7, 1.2],
        width_ratios=[1.0, 1.0],
    )

    ax_trans = fig.add_subplot(gs[0, 0])
    ax_param = fig.add_subplot(gs[0, 1])
    ax_post = fig.add_subplot(gs[1, :])
    ax_psy = fig.add_subplot(gs[2, 0])
    ax_trans_graph = fig.add_subplot(gs[2, 1])

    plot_transition_matrix(
        map_glmhmm,
        title="",
        ax=ax_trans,
        cmap="gray",
        fontsize=8,
    )

    post_prob_list = get_posterior_probs_concat(map_glmhmm, datas, inpts)
    df_with_state = add_df_glmhmm_state(df, post_prob_list)
    df_with_state = add_performance_by_session(df_with_state)

    plot_posteriors_with_performance(
        df_with_state,
        post_prob_list,
        colors,
        ax=ax_post,
    )
    ax_post.set_title("Posterior p(state) + performance", fontsize=10)

    param_cols = [col for col in input_cols]
    plot_param_by_state(
        map_glmhmm,
        df_with_state,
        X=param_cols,
        y=y_col,
        colors=colors,
        ax=ax_param,
    )
    ax_param.set_title("Logistic parameters by state", fontsize=10)

    plot_psychometric_by_state(
        df_with_state,
        map_glmhmm,
        colors,
        x=psychometric_x,
        y=y_col,
        ax=ax_psy,
        valueType=psychometric_value_type,
        log=False,
    )
    ax_psy.set_title("Psychometric by state", fontsize=10)

    plot_transition_graph(
        map_glmhmm,
        ax=ax_trans_graph,
        colors=colors,
        fontsize=8,
    )
    ax_trans_graph.set_title("Transition probabilities", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=12, y=0.995)

    fig.tight_layout()
    return fig, df_with_state, post_prob_list



def detect_incorrect_hold_and_next_event(events):
    incorrect_hold_sequence_one = ['Port2In', 'Tup', 'Port2Out']
    incorrect_hold_sequence_two = ['Port2In', 'Port2Out']
    correct_hold_sequence = ['Port2In', 'Tup', 'Tup', 'Port2Out']
    next_events = []
    events = eval(events)
    # remove all events after a correct hold
    for i in range(len(events) - 3):
        if events[i:i+4] == correct_hold_sequence:
            events = events[:i+4]
            break
    for i in range(len(events) - 2):
        if events[i:i+3] == incorrect_hold_sequence_one:
            if i + 3 < len(events):
                next_events.append(events[i + 3])
        elif events[i:i+2] == incorrect_hold_sequence_two:
            if i + 2 < len(events):
                next_events.append(events[i + 2])
    return next_events


def add_fixation_break_columns(
    df,
    event_col="ordered_list_of_events",
    next_event_col="next_event_after_fixation_break",
    count_col="fixation_breaks",
):
    df = df.copy()
    df[next_event_col] = df[event_col].apply(detect_incorrect_hold_and_next_event)
    df[count_col] = df[next_event_col].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    return df

def add_number_of_pokes(df_raw: pd.DataFrame, port_number: int):
    df = df_raw.copy()
    port_hold_column = f"port{port_number}_holds"
    df[port_hold_column] = df.apply(lambda row: utils.get_trial_port_hold(row, port_number), axis=1)
    df[f"port{port_number}_pokes_num"] = df[port_hold_column].apply(lambda x: len(x))
    return df


def add_number_of_centralpokes_after_choice(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df["eager_pokes"] = (
        pd.to_numeric(df["port2_pokes_num"], errors="coerce")
        - pd.to_numeric(df["fixation_breaks"], errors="coerce") - 1
    )
    return df
