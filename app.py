import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime


MODEL_FOLDER = "model"

MODEL_PATH = os.path.join(MODEL_FOLDER, "logistic_regression_student_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_FOLDER, "label_encoder.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_FOLDER, "feature_columns.pkl")
DEFAULT_VALUES_PATH = os.path.join(MODEL_FOLDER, "default_values.pkl")
TRAINING_DATA_PATH = os.path.join(MODEL_FOLDER, "processed_training_data.pkl")

SAVE_RECORD_PATH = "student_prediction_records.csv"


st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


def safe_divide(a, b):
    return a / (b + 1e-6)


@st.cache_resource
def load_model_files():
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    default_values = joblib.load(DEFAULT_VALUES_PATH)
    training_df = joblib.load(TRAINING_DATA_PATH)

    return model, label_encoder, feature_columns, default_values, training_df


def normalized_value(training_df, col, value):
    if col not in training_df.columns:
        return 0

    min_val = training_df[col].min()
    max_val = training_df[col].max()

    return (value - min_val) / (max_val - min_val + 1e-9)


def recompute_engineered_features(user_data, training_df):
    if "attendance_study_score" in user_data:
        user_data["attendance_study_score"] = (
            user_data.get("attendance_rate", 0)
            * user_data.get("study_hours_per_week", 0)
        )

    if "assignment_midterm_score" in user_data:
        user_data["assignment_midterm_score"] = (
            user_data.get("assignment_average_score", 0)
            * user_data.get("midterm_score", 0)
        )

    if "gpa_prior_score" in user_data:
        user_data["gpa_prior_score"] = (
            user_data.get("previous_gpa", 0)
            * user_data.get("prior_subject_score", 0)
        )

    if "submission_assignment_score" in user_data:
        user_data["submission_assignment_score"] = (
            user_data.get("assignments_submitted_percent", 0)
            * user_data.get("assignment_average_score", 0)
        )

    if "quiz_lms_score" in user_data:
        user_data["quiz_lms_score"] = (
            user_data.get("practice_quizzes_taken", 0)
            * user_data.get("lms_logins_per_week", 0)
        )

    if "teacher_motivation_score" in user_data:
        user_data["teacher_motivation_score"] = (
            user_data.get("teacher_feedback_score", 0)
            * user_data.get("motivation_level", 0)
        )

    if "absence_late_risk" in user_data:
        user_data["absence_late_risk"] = (
            user_data.get("absences", 0)
            + user_data.get("late_submissions", 0)
        )

    if "stress_sleep_risk" in user_data:
        user_data["stress_sleep_risk"] = safe_divide(
            user_data.get("stress_level", 0),
            user_data.get("sleep_hours_per_day", 1)
        )

    if "study_absence_ratio" in user_data:
        user_data["study_absence_ratio"] = safe_divide(
            user_data.get("study_hours_per_week", 0),
            user_data.get("absences", 0) + 1
        )

    if "attendance_absence_ratio" in user_data:
        user_data["attendance_absence_ratio"] = safe_divide(
            user_data.get("attendance_rate", 0),
            user_data.get("absences", 0) + 1
        )

    if "assignment_late_ratio" in user_data:
        user_data["assignment_late_ratio"] = safe_divide(
            user_data.get("assignment_average_score", 0),
            user_data.get("late_submissions", 0) + 1
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

    index_parts = []

    for col in positive_columns:
        if col in user_data:
            index_parts.append(
                normalized_value(training_df, col, user_data[col])
            )

    for col in negative_columns:
        if col in user_data:
            index_parts.append(
                1 - normalized_value(training_df, col, user_data[col])
            )

    if "hybrid_academic_index" in user_data and len(index_parts) > 0:
        user_data["hybrid_academic_index"] = np.mean(index_parts)

    return user_data


def save_prediction(user_df, prediction_label, probability_df):
    saved_row = user_df.copy()

    saved_row["predicted_performance"] = prediction_label
    saved_row["prediction_confidence_percent"] = round(
        probability_df.iloc[0]["Probability (%)"],
        2
    )

    for _, row in probability_df.iterrows():
        category = row["Performance Category"]
        probability = row["Probability (%)"]
        saved_row[f"probability_{category}"] = round(probability, 2)

    saved_row["prediction_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(SAVE_RECORD_PATH):
        saved_row.to_csv(SAVE_RECORD_PATH, mode="a", header=False, index=False)
    else:
        saved_row.to_csv(SAVE_RECORD_PATH, index=False)


def show_interpretation(prediction_label):
    label = prediction_label.lower()

    if label == "low":
        st.error("The student is currently at academic risk.")
        st.write(
            "Recommended actions: improve attendance, increase study time, submit assignments regularly, and provide extra academic support."
        )

    elif label == "average":
        st.warning("The student is performing at an average level.")
        st.write(
            "Recommended actions: improve consistency, reduce late submissions, practice more quizzes, and prepare better for exams."
        )

    elif label == "good":
        st.success("The student is performing well.")
        st.write(
            "Recommended actions: maintain attendance, continue regular study habits, and stay active in coursework."
        )

    elif label == "excellent":
        st.success("The student is performing excellently.")
        st.write(
            "Recommended actions: encourage advanced learning, leadership tasks, and higher academic challenges."
        )

    else:
        st.info("Prediction completed successfully.")


def main():
    st.title("🎓 Student Evaluation Performance Prediction Using Machine Learning Technique"")
    st.markdown(
        "This app predicts a student's performance category using a trained **Logistic Regression** machine learning model."
    )

    required_files = [
        MODEL_PATH,
        LABEL_ENCODER_PATH,
        FEATURE_COLUMNS_PATH,
        DEFAULT_VALUES_PATH,
        TRAINING_DATA_PATH
    ]

    missing_files = [file for file in required_files if not os.path.exists(file)]

    if missing_files:
        st.error("Model files are missing.")
        st.write("Please run the training file first:")
        st.code("python train_model.py")
        st.write("Missing files:")
        st.write(missing_files)
        return

    model, label_encoder, feature_columns, default_values, training_df = load_model_files()

    st.divider()

    st.subheader("Enter Student Information")

    user_data = default_values.copy()

    left, right = st.columns(2)

    with left:
        if "attendance_rate" in feature_columns:
            user_data["attendance_rate"] = st.number_input(
                "Attendance Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(default_values["attendance_rate"]),
                step=1.0
            )

        if "study_hours_per_week" in feature_columns:
            user_data["study_hours_per_week"] = st.number_input(
                "Study Hours Per Week",
                min_value=0.0,
                max_value=80.0,
                value=float(default_values["study_hours_per_week"]),
                step=1.0
            )

        if "assignment_average_score" in feature_columns:
            user_data["assignment_average_score"] = st.number_input(
                "Assignment Average Score (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(default_values["assignment_average_score"]),
                step=1.0
            )

        if "assignments_submitted_percent" in feature_columns:
            user_data["assignments_submitted_percent"] = st.number_input(
                "Assignments Submitted (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(default_values["assignments_submitted_percent"]),
                step=1.0
            )

        if "midterm_score" in feature_columns:
            user_data["midterm_score"] = st.number_input(
                "Midterm Score (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(default_values["midterm_score"]),
                step=1.0
            )

        if "previous_gpa" in feature_columns:
            user_data["previous_gpa"] = st.number_input(
                "Previous GPA",
                min_value=0.0,
                max_value=4.0,
                value=float(default_values["previous_gpa"]),
                step=0.1
            )

    with right:
        if "prior_subject_score" in feature_columns:
            user_data["prior_subject_score"] = st.number_input(
                "Prior Subject Score (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(default_values["prior_subject_score"]),
                step=1.0
            )

        if "absences" in feature_columns:
            user_data["absences"] = st.number_input(
                "Number of Absences",
                min_value=0.0,
                max_value=100.0,
                value=float(default_values["absences"]),
                step=1.0
            )

        if "late_submissions" in feature_columns:
            user_data["late_submissions"] = st.number_input(
                "Late Submissions",
                min_value=0.0,
                max_value=100.0,
                value=float(default_values["late_submissions"]),
                step=1.0
            )

        if "stress_level" in feature_columns:
            user_data["stress_level"] = st.slider(
                "Stress Level",
                min_value=1,
                max_value=10,
                value=int(default_values["stress_level"])
            )

        if "motivation_level" in feature_columns:
            user_data["motivation_level"] = st.slider(
                "Motivation Level",
                min_value=1,
                max_value=10,
                value=int(default_values["motivation_level"])
            )

        if "sleep_hours_per_day" in feature_columns:
            user_data["sleep_hours_per_day"] = st.number_input(
                "Sleep Hours Per Day",
                min_value=0.0,
                max_value=24.0,
                value=float(default_values["sleep_hours_per_day"]),
                step=0.5
            )

    st.subheader("Support Information")

    support_left, support_right = st.columns(2)

    important_categorical_columns = [
        "internet_access",
        "device_availability",
        "prior_programming_experience",
        "tuition_support"
    ]

    with support_left:
        for col in important_categorical_columns[:2]:
            if col in feature_columns:
                options = sorted(training_df[col].astype(str).unique().tolist())
                default_option = str(default_values[col])

                if default_option not in options:
                    default_option = options[0]

                user_data[col] = st.selectbox(
                    col.replace("_", " ").title(),
                    options,
                    index=options.index(default_option)
                )

    with support_right:
        for col in important_categorical_columns[2:]:
            if col in feature_columns:
                options = sorted(training_df[col].astype(str).unique().tolist())
                default_option = str(default_values[col])

                if default_option not in options:
                    default_option = options[0]

                user_data[col] = st.selectbox(
                    col.replace("_", " ").title(),
                    options,
                    index=options.index(default_option)
                )

    user_data = recompute_engineered_features(user_data, training_df)

    user_df = pd.DataFrame([user_data])
    user_df = user_df[feature_columns]

    st.divider()

    if st.button("Predict Student Performance", type="primary"):
        prediction_encoded = model.predict(user_df)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        prediction_probability = model.predict_proba(user_df)[0]

        probability_df = pd.DataFrame({
            "Performance Category": label_encoder.classes_,
            "Probability (%)": prediction_probability * 100
        })

        probability_df = probability_df.sort_values(
            by="Probability (%)",
            ascending=False
        )

        probability_df["Probability (%)"] = probability_df["Probability (%)"].round(2)

        confidence = probability_df.iloc[0]["Probability (%)"]

        st.subheader("Prediction Result")

        col_a, col_b = st.columns(2)

        with col_a:
            st.metric("Predicted Performance", prediction_label)

        with col_b:
            st.metric("Confidence", f"{confidence}%")

        st.subheader("Class Probability Breakdown")
        st.dataframe(probability_df, use_container_width=True)

        st.subheader("Final Interpretation")
        show_interpretation(prediction_label)

        save_prediction(user_df, prediction_label, probability_df)

        st.info("Prediction saved successfully in student_prediction_records.csv")

    if os.path.exists(SAVE_RECORD_PATH):
        st.divider()
        st.subheader("Saved Prediction Records")

        saved_df = pd.read_csv(SAVE_RECORD_PATH)
        st.dataframe(saved_df.tail(10), use_container_width=True)


if __name__ == "__main__":
    main()
