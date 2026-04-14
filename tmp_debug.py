import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance

NUMERIC_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
]
SUPPORTED_CATEGORICAL = [
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

def build_preprocessor():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                SUPPORTED_CATEGORICAL,
            ),
        ]
    )

df = pd.read_csv("StudentPerformanceFactors.csv")
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]
pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", LinearRegression())])
pipeline.fit(X, y)
pre = pipeline.named_steps["preprocess"]
transformed_names = NUMERIC_FEATURES + list(
    pre.named_transformers_["cat"].get_feature_names_out(SUPPORTED_CATEGORICAL)
)
perm = permutation_importance(
    pipeline,
    X,
    y,
    n_repeats=1,
    random_state=42,
    scoring="neg_root_mean_squared_error",
)
print("names len", len(transformed_names))
print("perm len", len(perm.importances_mean))
print("matched?", len(transformed_names) == len(perm.importances_mean))
