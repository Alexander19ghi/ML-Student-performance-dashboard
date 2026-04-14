import os
import io
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Student ML Pipeline Studio", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(59,130,246,0.15), transparent 30%),
            radial-gradient(circle at 90% 10%, rgba(16,185,129,0.12), transparent 35%),
            #020617;
    }
    .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px;}
    .hero-box {
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 12px;
        background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.95));
        box-shadow: 0 10px 30px rgba(2,6,23,0.35);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        color: #e2e8f0;
    }
    .hero-subtitle {
        color: #93c5fd;
        margin: 0;
        font-size: 1rem;
    }
    .guide-box {
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 14px;
        background: rgba(15, 23, 42, 0.9);
        margin: 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero-box">
        <p class="hero-title">Student ML Pipeline Studio</p>
        <p class="hero-subtitle">Interactive end-to-end ML workflow with guided concepts, task-aware models, and live evaluation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def _separator_candidates(separator_label: str) -> list[tuple[Optional[str], dict]]:
    if separator_label == "Comma (,)":
        return [(",", {})]
    if separator_label == "Semicolon (;)":
        return [(";", {})]
    if separator_label == "Tab":
        return [("\t", {})]
    return [
        (",", {}),
        (";", {}),
        ("\t", {}),
        (None, {"engine": "python"}),
    ]


def _pick_best_parsed_df(candidates: list[pd.DataFrame]) -> pd.DataFrame:
    if not candidates:
        raise ValueError("Unable to parse CSV file.")
    # Prefer parse with the most columns and most non-null cells.
    return max(candidates, key=lambda d: (d.shape[1], int(d.notna().sum().sum())))


def _read_csv_with_separator_fallback(csv_path: Path, separator_label: str) -> pd.DataFrame:
    candidates = []
    for sep, kwargs in _separator_candidates(separator_label):
        try:
            parsed = pd.read_csv(csv_path, sep=sep, **kwargs)
            candidates.append(parsed)
        except Exception:
            continue
    return _pick_best_parsed_df(candidates)


def _read_uploaded_csv_with_fallback(uploaded_file, separator_label: str) -> pd.DataFrame:
    raw_bytes = uploaded_file.getvalue()
    candidates = []
    for sep, kwargs in _separator_candidates(separator_label):
        try:
            parsed = pd.read_csv(io.BytesIO(raw_bytes), sep=sep, **kwargs)
            candidates.append(parsed)
        except Exception:
            continue
    return _pick_best_parsed_df(candidates)


def discover_student_dataset_files() -> dict[str, Path]:
    cwd = Path.cwd()
    home = Path.home()
    temp_dir = Path(os.environ.get("TEMP", ""))
    dataset_names = [
        "student-mat.csv",
        "student-por.csv",
        "Student_Performance.csv",
        "StudentsPerformance.csv",
        "student_performance.csv",
    ]
    found: dict[str, Path] = {}

    # Prefer project-local datasets first.
    base_dirs = [
        cwd,
        cwd / "data",
        cwd.parent,
        home / "Documents",
        home / "Downloads",
    ]
    for dataset_name in dataset_names:
        for base_dir in base_dirs:
            path = base_dir / dataset_name
            if path.exists():
                found[dataset_name] = path
                break

    # Fallback for extracted archives in temp folder.
    if temp_dir.exists():
        for dataset_name in dataset_names:
            if dataset_name in found:
                continue
            for path in temp_dir.glob(f"Rar$*/*{dataset_name}"):
                if path.exists():
                    found[dataset_name] = path
                    break

    return found


@st.cache_data(show_spinner=False)
def load_public_default_student_dataset() -> tuple[pd.DataFrame, str]:
    """
    Fallback dataset loader for cloud deployments where local files are unavailable.
    """
    public_sources = [
        (
            "UCI Student Math (student-mat.csv)",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv",
        ),
        (
            "UCI Student Portuguese (student-por.csv)",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-por.csv",
        ),
    ]
    for source_name, source_url in public_sources:
        try:
            fallback_df = pd.read_csv(source_url, sep=";")
            if fallback_df.shape[1] >= 2:
                return fallback_df, source_name
        except Exception:
            continue
    raise ValueError("Could not load public fallback student dataset.")


@st.cache_data(show_spinner=False)
def load_embedded_default_student_dataset() -> tuple[pd.DataFrame, str]:
    """
    Final fallback that always works without internet access.
    """
    rng = np.random.default_rng(42)
    n_rows = 240
    study_hours = rng.integers(1, 8, n_rows)
    attendance = rng.integers(60, 101, n_rows)
    sleep_hours = rng.integers(4, 9, n_rows)
    previous_score = rng.integers(40, 96, n_rows)
    internet_access = rng.choice(["yes", "no"], n_rows, p=[0.82, 0.18])
    extracurricular = rng.choice(["yes", "no"], n_rows, p=[0.55, 0.45])
    parental_support = rng.choice(["low", "medium", "high"], n_rows, p=[0.24, 0.5, 0.26])
    final_score = (
        0.34 * previous_score
        + 2.6 * study_hours
        + 0.18 * attendance
        + 0.9 * sleep_hours
        + np.where(internet_access == "yes", 2.4, -1.6)
        + np.where(extracurricular == "yes", 1.2, 0.0)
        + np.where(parental_support == "high", 2.2, np.where(parental_support == "medium", 0.8, -1.0))
        + rng.normal(0, 3.2, n_rows)
    )
    final_score = np.clip(final_score, 0, 100).round(1)
    final_grade = pd.cut(
        final_score,
        bins=[-1, 39, 59, 74, 89, 100],
        labels=["F", "D", "C", "B", "A"],
    ).astype(str)
    embedded_df = pd.DataFrame(
        {
            "study_hours_per_day": study_hours,
            "attendance_percent": attendance,
            "sleep_hours": sleep_hours,
            "previous_exam_score": previous_score,
            "internet_access": internet_access,
            "extracurricular": extracurricular,
            "parental_support": parental_support,
            "final_score": final_score,
            "final_grade": final_grade,
        }
    )
    return embedded_df, "Embedded synthetic student dataset"


def inject_data_issues(
    source_df: pd.DataFrame,
    missing_pct: float,
    outlier_pct: float,
    random_state: int,
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    issue_df = source_df.copy()
    rng = np.random.default_rng(random_state)
    stats = {"missing_cells_added": 0, "outliers_added": 0, "categorical_typos_added": 0}

    # Inject missing values into feature columns only.
    feature_cols = [c for c in issue_df.columns if c != target_column]
    if missing_pct > 0 and feature_cols:
        total_cells = issue_df[feature_cols].size
        n_missing = int(total_cells * (missing_pct / 100.0))
        for _ in range(n_missing):
            r = rng.integers(0, len(issue_df))
            c = feature_cols[rng.integers(0, len(feature_cols))]
            issue_df.at[issue_df.index[r], c] = np.nan
        stats["missing_cells_added"] = n_missing

    # Inject numeric outliers.
    num_cols = [c for c in issue_df.select_dtypes(include=["number"]).columns if c != target_column]
    if outlier_pct > 0 and num_cols:
        n_rows = max(1, int(len(issue_df) * (outlier_pct / 100.0)))
        for _ in range(n_rows):
            r = rng.integers(0, len(issue_df))
            c = num_cols[rng.integers(0, len(num_cols))]
            original = issue_df.at[issue_df.index[r], c]
            if pd.notna(original):
                issue_df.at[issue_df.index[r], c] = float(original) * 4.0
                stats["outliers_added"] += 1

    # Inject simple categorical inconsistency (extra spaces / case change).
    cat_cols = [c for c in issue_df.select_dtypes(exclude=["number"]).columns if c != target_column]
    if cat_cols:
        n_typo_rows = max(1, int(len(issue_df) * 0.02))
        for _ in range(n_typo_rows):
            r = rng.integers(0, len(issue_df))
            c = cat_cols[rng.integers(0, len(cat_cols))]
            val = issue_df.at[issue_df.index[r], c]
            if pd.notna(val):
                issue_df.at[issue_df.index[r], c] = f" {str(val).upper()} "
                stats["categorical_typos_added"] += 1

    return issue_df, stats


def get_preprocessor(x_df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = x_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = x_df.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def get_model(model_name: str, random_state: int, task_type: str):
    if task_type == "regression":
        if model_name == "Linear Regression":
            return LinearRegression()
        if model_name == "Random Forest Regressor":
            return RandomForestRegressor(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            )
        return MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=800,
            random_state=random_state,
        )

    if model_name == "Logistic-like (MLP Classifier)":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=800,
            random_state=random_state,
        )
    if model_name == "Random Forest Classifier":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        )
    return KNeighborsClassifier(n_neighbors=7)


def show_corr_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        st.warning("No numeric columns available for correlation heatmap.")
        return

    corr = numeric_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    st.pyplot(fig)


def explain_box(what: str, why: str):
    st.markdown(
        f"""
        <div class="guide-box">
            <b>What is happening:</b> {what}<br>
            <b>Why this step matters:</b> {why}
        </div>
        """,
        unsafe_allow_html=True,
    )


def prepare_target_series(y_series: pd.Series) -> tuple[pd.Series, str]:
    """
    Convert target to a numeric series suitable for modeling.
    Returns numeric target and a short mode label for UI messaging.
    """
    if pd.api.types.is_numeric_dtype(y_series):
        return y_series, "numeric"

    coerced = pd.to_numeric(y_series, errors="coerce")
    if coerced.notna().sum() > 0 and coerced.notna().sum() >= int(0.8 * len(y_series)):
        # Keep values that can be interpreted as numbers (e.g. "85", "90", ...).
        return coerced, "numeric_from_text"

    encoded = pd.Series(pd.Categorical(y_series).codes, index=y_series.index).astype(float)
    return encoded, "categorical_encoded"


def detect_task_type(original_target: pd.Series, prepared_mode: str) -> str:
    # If target is categorical text, prefer classification workflow.
    if prepared_mode == "categorical_encoded":
        return "classification"
    # Numeric target with few distinct integer-ish levels is usually classification.
    unique_count = original_target.nunique(dropna=True)
    if unique_count <= 12 and not pd.api.types.is_float_dtype(original_target):
        return "classification"
    return "regression"


def show_concept_explainer(task_type: str, model_name: str):
    with st.expander("Concept explainer: why this pipeline choice?", expanded=True):
        st.markdown(f"**Detected task:** `{task_type.title()}`")
        if task_type == "regression":
            st.markdown(
                "- **Why regression:** target behaves like a numeric value, so the model predicts a continuous number.\n"
                "- **Feature score method (`f_regression`):** ranks features by linear relation with numeric target.\n"
                "- **Main metrics:** MSE/MAE/RMSE/R2.\n"
                "  - MSE/RMSE penalize large errors more.\n"
                "  - MAE gives average absolute error.\n"
                "  - R2 shows explained variance."
            )
            if model_name == "Linear Regression":
                st.info(
                    "Model concept: Linear Regression is a baseline that learns straight-line relationships and is easy to interpret."
                )
            elif model_name == "Random Forest Regressor":
                st.info(
                    "Model concept: Random Forest Regressor combines many trees, capturing non-linear patterns and feature interactions."
                )
            else:
                st.info(
                    "Model concept: MLP Regressor is a neural network that can learn complex non-linear mappings."
                )
        else:
            st.markdown(
                "- **Why classification:** target behaves like labels/classes, so the model predicts class membership.\n"
                "- **Feature score method (`f_classif`):** ranks features by how well they separate classes.\n"
                "- **Main metrics:** Accuracy, Precision, Recall, F1, Confusion Matrix.\n"
                "  - Accuracy: overall correctness.\n"
                "  - Precision/Recall/F1: class-quality trade-offs.\n"
                "  - Confusion matrix: class-wise error analysis."
            )
            if model_name == "Logistic-like (MLP Classifier)":
                st.info(
                    "Model concept: MLP Classifier learns non-linear decision boundaries between classes."
                )
            elif model_name == "Random Forest Classifier":
                st.info(
                    "Model concept: Random Forest Classifier is robust on tabular data and handles mixed patterns well."
                )
            else:
                st.info(
                    "Model concept: KNN Classifier predicts by nearest neighbors in feature space."
                )
        st.caption(
            "Cross-validation concept: K-Fold repeats train/test across multiple splits to reduce luck from one random split."
        )


st.sidebar.header("Data Source")
discovered_datasets = discover_student_dataset_files()
source_mode = st.sidebar.radio("Input Mode", ["Auto detect", "Manual upload"], horizontal=True)
separator_choice = st.sidebar.selectbox(
    "CSV Delimiter",
    ["Auto detect", "Comma (,)", "Semicolon (;)", "Tab"],
    index=0,
)

if source_mode == "Manual upload":
    uploaded = st.sidebar.file_uploader("Upload student CSV", type=["csv"])
    if uploaded is None:
        st.warning("Upload a CSV file to continue.")
        st.stop()
    df = _read_uploaded_csv_with_fallback(uploaded, separator_choice)
    st.sidebar.success(f"Using uploaded file: {uploaded.name}")
else:
    if not discovered_datasets:
        try:
            df, source_name = load_public_default_student_dataset()
            st.sidebar.success(f"Auto-loaded public default dataset: {source_name}")
            st.sidebar.caption("Source: UCI Machine Learning Repository")
        except Exception:
            df, source_name = load_embedded_default_student_dataset()
            st.sidebar.warning("Public dataset unavailable. Loaded built-in backup dataset.")
            st.sidebar.caption(f"Source: {source_name}")
    else:
        dataset_options = list(discovered_datasets.keys())
        default_idx = dataset_options.index("Student_Performance.csv") if "Student_Performance.csv" in dataset_options else 0
        selected_dataset = st.sidebar.selectbox("Auto-detected Dataset", dataset_options, index=default_idx)
        selected_path = discovered_datasets[selected_dataset]
        df = _read_csv_with_separator_fallback(selected_path, separator_choice)
        st.sidebar.success(f"Auto-loaded: {selected_dataset}")
        st.sidebar.caption(f"Source: {selected_path}")

df.columns = [str(c).strip() for c in df.columns]
if df.shape[1] < 2:
    st.error(
        "CSV appears to have only one column after parsing. "
        "Change 'CSV Delimiter' in sidebar (try Comma or Semicolon)."
    )
    st.stop()

target_default = "G3" if "G3" in df.columns else df.columns[-1]
target_col = st.sidebar.selectbox("Target Column", df.columns, index=df.columns.get_loc(target_default))
random_state = st.sidebar.number_input("Random State", min_value=0, max_value=9999, value=42, step=1)
test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
drop_duplicates = st.sidebar.checkbox("Drop Duplicates", value=True)
k_features = st.sidebar.slider("Top K Features", min_value=3, max_value=25, value=10, step=1)
folds = st.sidebar.slider("K-Folds", min_value=3, max_value=10, value=5, step=1)
st.sidebar.markdown("### Real-world data issues simulator")
inject_missing_pct = st.sidebar.slider("Add missing values (%)", min_value=0, max_value=20, value=0, step=1)
inject_outlier_pct = st.sidebar.slider("Add outliers (%)", min_value=0, max_value=15, value=0, step=1)

work_df = df.copy()
work_df, issue_stats = inject_data_issues(
    work_df,
    missing_pct=inject_missing_pct,
    outlier_pct=inject_outlier_pct,
    random_state=random_state,
    target_column=target_col,
)
if drop_duplicates:
    work_df = work_df.drop_duplicates()

X = work_df.drop(columns=[target_col])
y_original = work_df[target_col]
y, y_mode = prepare_target_series(y_original)
task_type = detect_task_type(y_original, y_mode)

if X.shape[1] == 0:
    st.error("No feature columns left after selecting target column.")
    st.stop()
if y.isna().all():
    st.error("Target column contains only missing values.")
    st.stop()
if y_mode == "numeric_from_text":
    st.sidebar.info("Target values were text numbers and were auto-converted to numeric.")
elif y_mode == "categorical_encoded":
    st.sidebar.warning(
        "Selected target is categorical (text). It has been auto-encoded to numeric labels."
    )
st.sidebar.caption(f"Detected task: {task_type.title()}")
split_kwargs = {"test_size": test_size, "random_state": random_state}
if task_type == "classification" and y.nunique(dropna=True) > 1:
    split_kwargs["stratify"] = y

overview_c1, overview_c2, overview_c3, overview_c4 = st.columns(4)
overview_c1.metric("Rows", f"{len(df)}")
overview_c2.metric("Columns", f"{df.shape[1]}")
overview_c3.metric("Target", str(target_col))
overview_c4.metric("Task", task_type.title())

if task_type == "regression":
    model_name = st.sidebar.selectbox(
        "Model",
        ["Linear Regression", "Random Forest Regressor", "ANN (MLP Regressor)"],
    )
else:
    model_name = st.sidebar.selectbox(
        "Model",
        ["Logistic-like (MLP Classifier)", "Random Forest Classifier", "KNN Classifier"],
    )

show_concept_explainer(task_type, model_name)

preprocessor, numeric_cols, categorical_cols = get_preprocessor(X)
selected_model = get_model(model_name, random_state, task_type)

x_preview = preprocessor.fit_transform(X)
if hasattr(x_preview, "toarray"):
    x_preview = x_preview.toarray()
safe_k = min(k_features, x_preview.shape[1])
score_func = f_regression if task_type == "regression" else f_classif
selector_preview = SelectKBest(score_func=score_func, k=safe_k)
selector_preview.fit(x_preview, y)
feature_scores = selector_preview.scores_
feature_score_df = pd.DataFrame(
    {"FeatureIndex": np.arange(len(feature_scores)), "Score": feature_scores}
).sort_values("Score", ascending=False)

selector = SelectKBest(score_func=score_func, k=safe_k)
final_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("feature_select", selector),
        ("model", selected_model),
    ]
)

if "step_idx" not in st.session_state:
    st.session_state.step_idx = 0

steps = [
    "1) Input Data",
    "2) EDA",
    "3) Cleaning & Engineering",
    "4) Feature Selection",
    "5) Data Split",
    "6) Model Selection",
    "7) Model Training",
    "8) K-Fold Validation",
    "9) Final Performance",
]

selected_step = st.selectbox("Choose Pipeline Step", steps, index=st.session_state.step_idx)
st.session_state.step_idx = steps.index(selected_step)
st.progress((st.session_state.step_idx + 1) / len(steps), text=f"Current: {selected_step}")

nav1, nav2, nav3 = st.columns([1, 1, 6])
with nav1:
    if st.button("Previous", disabled=st.session_state.step_idx == 0):
        st.session_state.step_idx -= 1
        st.rerun()
with nav2:
    if st.button("Next", disabled=st.session_state.step_idx == len(steps) - 1):
        st.session_state.step_idx += 1
        st.rerun()

current_step = steps[st.session_state.step_idx]

if current_step == "1) Input Data":
    st.subheader("Step 1: Input Data")
    explain_box(
        "We load the student dataset and show a quick sample.",
        "Before building a model, we should verify what data we have.",
    )
    st.dataframe(df.head(15), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum())}")
    st.dataframe(
        df.dtypes.astype(str).reset_index().rename(columns={"index": "Column", 0: "Type"}),
        use_container_width=True,
    )

elif current_step == "2) EDA":
    st.subheader("Step 2: Exploratory Data Analysis (EDA)")
    explain_box(
        "We inspect patterns, missing values, and target distribution.",
        "EDA helps us understand behavior before training a model.",
    )
    missing_df = df.isna().sum().reset_index().rename(columns={"index": "Column", 0: "MissingCount"})
    fig_missing = px.bar(missing_df, x="Column", y="MissingCount", title="Missing Values by Column")
    fig_missing.update_layout(xaxis_tickangle=-45, height=350)
    st.plotly_chart(fig_missing, use_container_width=True)
    fig_target = px.histogram(df, x=target_col, nbins=20, title=f"Target Distribution: {target_col}")
    st.plotly_chart(fig_target, use_container_width=True)
    show_corr_heatmap(df)

elif current_step == "3) Cleaning & Engineering":
    st.subheader("Step 3: Data Engineering & Cleaning")
    explain_box(
        "We prepare clean, model-ready data using missing-value filling, encoding, and scaling.",
        "Good data quality usually gives better model performance.",
    )
    st.write(f"Original shape: {df.shape}")
    st.write(f"After duplicate option: {work_df.shape}")
    st.write(f"Numeric columns: {len(numeric_cols)}")
    st.write(f"Categorical columns: {len(categorical_cols)}")
    st.info("Numeric: median fill + scaling | Categorical: mode fill + one-hot encoding")
    c1, c2, c3 = st.columns(3)
    c1.metric("Missing added", issue_stats["missing_cells_added"])
    c2.metric("Outliers added", issue_stats["outliers_added"])
    c3.metric("Category typos added", issue_stats["categorical_typos_added"])
    st.caption("These simulated issues help demonstrate real-time data cleaning use case.")

elif current_step == "4) Feature Selection":
    st.subheader("Step 4: Feature Selection")
    explain_box(
        f"We select top {safe_k} features based on statistical relevance.",
        "Removing weak features can make models faster and cleaner.",
    )
    top_feature_df = feature_score_df.head(safe_k).copy()
    st.dataframe(top_feature_df, use_container_width=True)
    fig_fs = px.bar(
        top_feature_df.sort_values("Score", ascending=True),
        x="Score",
        y="FeatureIndex",
        orientation="h",
        title=f"Top {safe_k} Features",
    )
    st.plotly_chart(fig_fs, use_container_width=True)

elif current_step == "5) Data Split":
    st.subheader("Step 5: Data Split")
    explain_box(
        "We split data into training and testing sets.",
        "This checks if the model works on unseen data, not just memorized data.",
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
    st.write(f"Train: {X_train.shape} | Test: {X_test.shape}")

elif current_step == "6) Model Selection":
    st.subheader("Step 6: Model Selection")
    explain_box(
        f"Current selected model: {model_name}.",
        "Different models learn patterns in different ways.",
    )
    if task_type == "regression":
        st.markdown(
            "- Linear Regression: simple baseline\n"
            "- Random Forest Regressor: strong non-linear model\n"
            "- ANN (MLP Regressor): neural-network approach"
        )
    else:
        st.markdown(
            "- MLP Classifier: neural-network style classifier\n"
            "- Random Forest Classifier: robust tree-based classifier\n"
            "- KNN Classifier: neighborhood-based classification"
        )

elif current_step == "7) Model Training":
    st.subheader("Step 7: Model Training")
    explain_box(
        "Now we train the selected model on training data.",
        "Training is where the model learns relationships from examples.",
    )
    if st.button("Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
        final_pipeline.fit(X_train, y_train)
        st.success("Training completed.")

elif current_step == "8) K-Fold Validation":
    st.subheader("Step 8: K-Fold Validation")
    explain_box(
        "We evaluate model stability by repeating train/test in multiple folds.",
        "This gives a more reliable estimate than one single split.",
    )
    if st.button("Run K-Fold"):
        kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        cv_model = clone(final_pipeline)
        if task_type == "regression":
            neg_mse_scores = cross_val_score(
                cv_model, X, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1
            )
            rmse_scores = np.sqrt(-neg_mse_scores)
            cv_df = pd.DataFrame({"Fold": np.arange(1, folds + 1), "RMSE": rmse_scores})
            st.dataframe(cv_df, use_container_width=True)
            st.metric("Average RMSE", f"{rmse_scores.mean():.4f}")
        else:
            acc_scores = cross_val_score(cv_model, X, y, cv=kf, scoring="accuracy", n_jobs=-1)
            f1_scores = cross_val_score(
                cv_model, X, y, cv=kf, scoring="f1_weighted", n_jobs=-1
            )
            cv_df = pd.DataFrame(
                {
                    "Fold": np.arange(1, folds + 1),
                    "Accuracy": acc_scores,
                    "F1_weighted": f1_scores,
                }
            )
            st.dataframe(cv_df, use_container_width=True)
            c1, c2 = st.columns(2)
            c1.metric("Average Accuracy", f"{acc_scores.mean():.4f}")
            c2.metric("Average F1 (weighted)", f"{f1_scores.mean():.4f}")

elif current_step == "9) Final Performance":
    st.subheader("Step 9: Performance Metrics")
    explain_box(
        "We test on unseen data and show final error metrics.",
        "These numbers tell us how good the model is in practical use.",
    )
    if st.button("Evaluate Final Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
        eval_model = clone(final_pipeline)
        eval_model.fit(X_train, y_train)
        preds = eval_model.predict(X_test)

        if task_type == "regression":
            mse_val = mean_squared_error(y_test, preds)
            mae_val = mean_absolute_error(y_test, preds)
            rmse_val = np.sqrt(mse_val)
            r2_val = r2_score(y_test, preds)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MSE", f"{mse_val:.3f}")
            c2.metric("MAE", f"{mae_val:.3f}")
            c3.metric("RMSE", f"{rmse_val:.3f}")
            c4.metric("R2", f"{r2_val:.3f}")

            results_df = pd.DataFrame({"Actual": y_test.values, "Predicted": preds})
            fig_pred = px.scatter(results_df, x="Actual", y="Predicted", title="Actual vs Predicted")
            st.plotly_chart(fig_pred, use_container_width=True)
        else:
            acc_val = accuracy_score(y_test, preds)
            precision_val = precision_score(y_test, preds, average="weighted", zero_division=0)
            recall_val = recall_score(y_test, preds, average="weighted", zero_division=0)
            f1_val = f1_score(y_test, preds, average="weighted", zero_division=0)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc_val:.3f}")
            c2.metric("Precision (weighted)", f"{precision_val:.3f}")
            c3.metric("Recall (weighted)", f"{recall_val:.3f}")
            c4.metric("F1 (weighted)", f"{f1_val:.3f}")

            confusion_df = pd.crosstab(
                pd.Series(y_test, name="Actual"),
                pd.Series(preds, name="Predicted"),
            )
            st.write("Confusion Matrix")
            st.dataframe(confusion_df, use_container_width=True)
