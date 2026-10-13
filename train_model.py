import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


DATASET_PATH = "student_performance_report.csv"
TARGET_COLUMN = "performance_category_target"
MODEL_FOLDER = "model"


def safe_divide(a, b):
    return a / (b + 1e-6)


def minmax(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-9)


def add_feature_engineering(df):
    if "attendance_rate" in df.columns and "study_hours_per_week" in df.columns:
        df["attendance_study_score"] = df["attendance_rate"] * df["study_hours_per_week"]

    if "assignment_average_score" in df.columns and "midterm_score" in df.columns:
        df["assignment_midterm_score"] = df["assignment_average_score"] * df["midterm_score"]

    if "previous_gpa" in df.columns and "prior_subject_score" in df.columns:
        df["gpa_prior_score"] = df["previous_gpa"] * df["prior_subject_score"]

    if "assignments_submitted_percent" in df.columns and "assignment_average_score" in df.columns:
        df["submission_assignment_score"] = (
            df["assignments_submitted_percent"] * df["assignment_average_score"]
        )

    if "practice_quizzes_taken" in df.columns and "lms_logins_per_week" in df.columns:
        df["quiz_lms_score"] = df["practice_quizzes_taken"] * df["lms_logins_per_week"]

    if "teacher_feedback_score" in df.columns and "motivation_level" in df.columns:
        df["teacher_motivation_score"] = df["teacher_feedback_score"] * df["motivation_level"]

    if "absences" in df.columns and "late_submissions" in df.columns:
        df["absence_late_risk"] = df["absences"] + df["late_submissions"]

    if "stress_level" in df.columns and "sleep_hours_per_day" in df.columns:
        df["stress_sleep_risk"] = safe_divide(
            df["stress_level"],
            df["sleep_hours_per_day"]
        )

    if "study_hours_per_week" in df.columns and "absences" in df.columns:
        df["study_absence_ratio"] = safe_divide(
            df["study_hours_per_week"],
            df["absences"] + 1
        )

    if "attendance_rate" in df.columns and "absences" in df.columns:
        df["attendance_absence_ratio"] = safe_divide(
            df["attendance_rate"],
            df["absences"] + 1
        )

    if "assignment_average_score" in df.columns and "late_submissions" in df.columns:
        df["assignment_late_ratio"] = safe_divide(
            df["assignment_average_score"],
            df["late_submissions"] + 1
        )

    positive_columns = [
        "attendance_rate",
        "assignment_average_score",
        "midterm_score",
        "previous_gpa",
        "prior_subject_score",
        "study_hours_per_week",
        "assignments_submitted_percent",
        "practice_quizzes_taken",
        "lms_logins_per_week",
        "video_watch_time_hours_per_week",
        "discussion_posts_per_week",
        "library_visits_per_month",
        "teacher_feedback_score",
        "motivation_level",
        "sleep_hours_per_day"
    ]

    negative_columns = [
        "absences",
        "late_submissions",
        "stress_level",
        "commute_time_minutes",
        "part_time_job_hours_per_week"
    ]

    academic_parts = []

    for col in positive_columns:
        if col in df.columns:
            academic_parts.append(minmax(df[col]))

    for col in negative_columns:
        if col in df.columns:
            academic_parts.append(1 - minmax(df[col]))

    if len(academic_parts) > 0:
        df["hybrid_academic_index"] = np.mean(academic_parts, axis=0)

    return df


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            "Dataset not found. Please keep student_performance_report.csv in the same folder."
        )

    os.makedirs(MODEL_FOLDER, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)

    print("Dataset Loaded Successfully")
    print("Dataset Shape:", df.shape)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    if "student_id" in df.columns:
        df = df.drop(columns=["student_id"])

    df = add_feature_engineering(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=42
            ))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")

    print("\nLogistic Regression Evaluation")
    print("--------------------------------")
    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1 Score :", round(f1 * 100, 2), "%")
    print("AUC Score:", round(auc * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    default_values = {}

    for col in X.columns:
        if col in numerical_columns:
            default_values[col] = X[col].median()
        else:
            default_values[col] = X[col].mode()[0]

    joblib.dump(model, os.path.join(MODEL_FOLDER, "logistic_regression_student_model.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_FOLDER, "label_encoder.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_FOLDER, "feature_columns.pkl"))
    joblib.dump(default_values, os.path.join(MODEL_FOLDER, "default_values.pkl"))
    joblib.dump(df, os.path.join(MODEL_FOLDER, "processed_training_data.pkl"))

    print("\nModel files saved successfully inside the model folder.")


if __name__ == "__main__":
    main()