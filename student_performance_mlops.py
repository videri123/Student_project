import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns we focus on for the correlation story. Hours_Studied and Attendance remain
# numeric features that drive the exam score so we keep them in every transformation.
NUMERIC_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
]

CATEGORICAL_FEATURES = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Motivation_Level",
    "Internet_Access",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender",
]

TARGET_COLUMN = "Exam_Score"
FOCUS_COLUMNS = ["Hours_Studied", "Attendance", TARGET_COLUMN]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlation-first student performance regression project"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "StudentPerformanceFactors.csv",
        help="Path to the source CSV that contains the student data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory that will hold plots, reports, and saved models",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state to keep experimentation deterministic",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def run_correlation_analysis(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    correlation_matrix = df[FOCUS_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Focus column correlation")
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = output_dir / "correlation_focus_heatmap.png"
    fig.tight_layout()
    fig.savefig(heatmap_path)
    plt.close(fig)

    pairplot_path = output_dir / "focus_pairplot.png"
    sns.pairplot(df[FOCUS_COLUMNS])
    plt.suptitle("Hours studied & attendance vs. exam score", y=1.02)
    plt.savefig(pairplot_path)
    plt.close()

    return correlation_matrix


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        numeric_names = NUMERIC_FEATURES
        ohe = preprocessor.named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        return numeric_names + cat_names


def evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    model,
    output_dir: Path,
    random_state: int,
) -> dict:
    preprocessor = build_preprocessor()
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    scoring = "neg_root_mean_squared_error"
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring=scoring)
    pipeline.fit(X, y)

    transformed_names = get_transformed_feature_names(pipeline.named_steps["preprocess"])
    report = {
        "model": model_name,
        "cv_rmse_mean": -cv_scores.mean(),
        "cv_rmse_std": cv_scores.std(),
        "feature_names": transformed_names,
    }

    if hasattr(pipeline.named_steps["model"], "coef_"):
        coef_target = pipeline.named_steps["model"].coef_
        report["feature_importances"] = dict(zip(transformed_names, coef_target))
    elif hasattr(pipeline.named_steps["model"], "feature_importances_"):
        fi = pipeline.named_steps["model"].feature_importances_
        report["feature_importances"] = dict(zip(transformed_names, fi))

    perm = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=20,
        random_state=random_state,
        scoring="neg_root_mean_squared_error",
    )
    importance_df = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .head(15)
    )
    perm_path = output_dir / f"{model_name.lower()}_permutation_importance.csv"
    importance_df.to_csv(perm_path, index=False)
    report["permutation_importance_path"] = perm_path

    predictions = pipeline.predict(X)
    plot_path = output_dir / f"{model_name.lower()}_pred_vs_actual.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, predictions, alpha=0.6)
    lims = [y.min(), y.max()]
    ax.plot(lims, lims, linestyle="--", color="gray")
    ax.set_title(f"{model_name}: Predicted vs actual")
    ax.set_xlabel("Actual exam score")
    ax.set_ylabel("Predicted exam score")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    report["prediction_plot"] = plot_path

    return report


def save_summary(metadata: dict, path: Path) -> None:
    text = textwrap.dedent(
        """
        Correlation-focused regression report

        Focus columns: Hours_Studied and Attendance with Exam_Score as the target.
        """
    )
    text += "\n" + "\n".join(f"{k}: {v}" for k, v in metadata.items())
    path.write_text(text)


def main() -> None:
    args = parse_args()
    df = load_data(args.data)
    print("Data loaded, shape:", df.shape)
    print(df[FOCUS_COLUMNS].head())
    print("Missing values by column:\n", df.isna().sum())

    output_dir = args.output_dir
    correlation_matrix = run_correlation_analysis(df, output_dir)
    print("Correlation matrix:\n", correlation_matrix)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    lr_report = evaluate_model(
        X,
        y,
        model_name="LinearRegression",
        model=LinearRegression(),
        output_dir=output_dir,
        random_state=args.random_state,
    )
    rf_report = evaluate_model(
        X,
        y,
        model_name="RandomForest",
        model=RandomForestRegressor(
            n_estimators=150,
            random_state=args.random_state,
            n_jobs=1,
        ),
        output_dir=output_dir,
        random_state=args.random_state,
    )

    summary_stats = {
        "corr_hours_exam": correlation_matrix.at["Hours_Studied", TARGET_COLUMN],
        "corr_attendance_exam": correlation_matrix.at["Attendance", TARGET_COLUMN],
        "linear_cv_rmse": f"{lr_report['cv_rmse_mean']:.3f} ± {lr_report['cv_rmse_std']:.3f}",
        "rf_cv_rmse": f"{rf_report['cv_rmse_mean']:.3f} ± {rf_report['cv_rmse_std']:.3f}",
        "lr_perm_importance": lr_report["permutation_importance_path"].name,
        "rf_perm_importance": rf_report["permutation_importance_path"].name,
    }
    summary_path = output_dir / "analysis_summary.txt"
    save_summary(summary_stats, summary_path)
    print("Summary saved to", summary_path)


if __name__ == "__main__":
    main()
