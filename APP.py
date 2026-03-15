import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="ARTIFICIAL INTELLIGENCE STUDENTS PERFORMANCE DASHBOARD", layout="wide")

st.title("ARTIFICIAL INTELLIGENCE STUDENTS PERFORMANCE PREDICTION AND SUPPORT SYSTEM")
st.write("UPLOAD STUDENTS DATASET TO TRAIN ARTIFICIAL INTELLIGENCE MODEL AND PREDICT RESULTS.")

# -----------------------------
# Grade classification
# -----------------------------
def grade_category(score):

    if 85 <= score >= 100:
        return "1, STRONG DISTINCTION"
    elif if 80 <= score >= 84:
        return "2, WEAK DISTINCTION"
    elif if 70 <= score >= 79:
        return "3, STRONG CREDIT"
    elif if 65 <= score >= 69:
        return "4, STRONG CREDIT"
    elif if 60 <= score >= 64:
        return "5, WEAK CREDIT"
    elif if 50 <= score >= 59:
        return "6, WEAK CREDIT"
    elif if 45 <= score >= 49: 
        return "7, STRONG PASS"
    elif if 40 <= score >= 44:
        return "8, WEAK PASS"
    elif if 0 <= score >= 39:
        return"9, STATEMENT"

    else:
        return "INVALID MARKS"


# -----------------------------
# Student support recommendation
# -----------------------------
def recommend_support(row):

    recommendations = []

    if row["ATTENDANCE"] < 60:
        recommendations.append("IMPROVE CLASS ATTENDANCE MONITORING")

    if row["HOMEWORK"] < 50:
        recommendations.append("PROVIDE ADDITIONAL HOMEWORK PRACTICE")

    if row["CA_SCORE"] < 50:
        recommendations.append("OFFER REMEDIAL MATHEMATICS CLASSES")

    if row["PREVIOUS_MATHS"] < 50:
        recommendations.append("ASSIGN PEER TUTORING")

    if len(recommendations) == 0:
        return "STUDENT PERFORMING WELL, REINFORCE POSITIVELY"

    return ", ".join(recommendations)


# -----------------------------
# Risk detection
# -----------------------------
def risk_level(row):

    risk_score = 0

    if row["ATTENDANCE"] < 60:
        risk_score += 1

    if row["HOMEWORK"] < 50:
        risk_score += 1

    if row["CA_SCORE"] < 50:
        risk_score += 1

    if row["PREVIOUS MATH"] < 50:
        risk_score += 1

    if risk_score >= 3:
        return "HIGH RISK"

    elif risk_score == 2:
        return "MEDIUM RISK"

    else:
        return "LOW RISK"


# -----------------------------
# Upload dataset
# -----------------------------
uploaded_file = st.file_uploader("UPLOAD CSV FILE", type="csv")

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    st.subheader("DATASET PREVIEW")
    st.dataframe(data.head())

    required = ['ATTENDANCE','HOMEWORK','CA_SCORE','PREVIOUS_MATH','RESULT']

    missing = [col for col in required if col not in data.columns]

    if missing:

        st.error(f"MISSING COLUMN: {MISSING}")

    else:

        X = data[['ATTENDANCE','HOMEWORK','CA_SCORE','PREVIOUS_MATH']]
        y = data['Result']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier()

        model.fit(X_train, y_train)

        st.success("AI MODEL TRAINED SUCCESSFULLY")

        # Predictions
        data["PREDICTION"] = model.predict(X)

        # Grade estimation
        data["PREDICTED_GRADE"] = data["PREVIOUS_MATH"].apply(grade_category)

        # Risk level
        data["RISK_LEVEL"] = data.apply(risk_level, axis=1)

        # Support recommendations
        data["SUPPORT"] = data.apply(recommend_support, axis=1)

        # -----------------------------
        # Show predictions
        # -----------------------------
        st.subheader("PREDICTED STUDENT RESULTS")

        st.dataframe(data[
            ["ATTENDANCE","HOMEWORK","CA_SCORE","PREVIOUS_MATH",
             "PREDICTION","PREDICTED_GRADE","RISK_LEVEL"]
        ])

        # -----------------------------
        # National exam grade summary
        # -----------------------------
        st.subheader("END OF TERM EXAMINATION GRADE PREDICTION SUMMARY")

        grade_summary = data["PREDICTED_GRADE"].value_counts()

        st.dataframe(grade_summary)

        st.bar_chart(grade_summary)

        # Pie chart
        fig, ax = plt.subplots()

        ax.pie(
            grade_summary.values,
            labels=grade_summary.index,
            autopct='%1.1f%%'
        )

        ax.set_title("PREDICTED END OF TERM EXAMS PERFORMANCE")

        st.pyplot(fig)

        # -----------------------------
        # Early warning system
        # -----------------------------
        st.subheader("EARLY WARNING SYSTEM")

        high_risk = data[data["RISK_LEVEL"] == "HIGH RISK"]

        st.write("STUDENTS NEEDING URGENT ACADEMIC INTERVETIONS")

        st.dataframe(high_risk[
            ["ATTENDANCE","HOMEWORK","CA_SCORE",
             "PREVIOUS_MATH","PREDICTION","RISK_LEVEL"]
        ])

        # -----------------------------
        # Support recommendations
        # -----------------------------
        st.subheader("STUDENTS SUPPORT RECOMMENDATION")

        st.dataframe(data[
            ["ATTENDANCE","HOMEWORK","CA_SCORE",
             "PREDICTION","RISK_LEVEL","SUPPORT"]
        ])


# -----------------------------
# Single student prediction
# -----------------------------
st.subheader("SINGLE STUDENT PREDICTION")

attendance = st.number_input("ATTENDANCE (%)",0,100)
homework = st.number_input("HOMEWORK SCORE (%)",0,100)
ca_score = st.number_input("CA SCORE (%)",0,100)
previous_math = st.number_input("PREVIOUS MATH SCORE (%)",0,100)

if st.button("PREDICT STUDENT PERFORMANCE"):

    sample = pd.DataFrame([[attendance,homework,ca_score,previous_math]],
        columns=['ATTENDANCE','HOMEWORK','CA_SCORE','PREVIOUS MATH_SCORE'])

    model = RandomForestClassifier()

    # dummy training so prediction works if dataset not uploaded
    model.fit([[50,50,50,50],[80,80,80,80]],[0,1])

    prediction = model.predict(sample)[0]

    grade = grade_category(previous_math)

    st.success(f"PREDICTED RESULT: {prediction}")

    st.info(f"EXPECTED GRADE: {grade}")
