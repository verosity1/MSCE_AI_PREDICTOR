import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Student Performance Dashboard", layout="wide")

st.title("AI Student Performance Prediction and Support System")
st.write("Upload student dataset to train AI model and predict results.")

# -----------------------------
# Grade classification
# -----------------------------
def grade_category(score):

    if score >= 80:
        return "Distinction"

    elif score >= 65:
        return "Credit"

    elif score >= 50:
        return "Pass"

    else:
        return "Fail"


# -----------------------------
# Student support recommendation
# -----------------------------
def recommend_support(row):

    recommendations = []

    if row["Attendance"] < 60:
        recommendations.append("Improve class attendance monitoring")

    if row["Homework"] < 50:
        recommendations.append("Provide additional homework practice")

    if row["CA_Score"] < 50:
        recommendations.append("Offer remedial mathematics classes")

    if row["Previous_Math"] < 50:
        recommendations.append("Assign peer tutoring")

    if len(recommendations) == 0:
        return "Student performing well"

    return ", ".join(recommendations)


# -----------------------------
# Risk detection
# -----------------------------
def risk_level(row):

    risk_score = 0

    if row["Attendance"] < 60:
        risk_score += 1

    if row["Homework"] < 50:
        risk_score += 1

    if row["CA_Score"] < 50:
        risk_score += 1

    if row["Previous_Math"] < 50:
        risk_score += 1

    if risk_score >= 3:
        return "High Risk"

    elif risk_score == 2:
        return "Medium Risk"

    else:
        return "Low Risk"


# -----------------------------
# Upload dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    required = ['Attendance','Homework','CA_Score','Previous_Math','Result']

    missing = [col for col in required if col not in data.columns]

    if missing:

        st.error(f"Missing columns: {missing}")

    else:

        X = data[['Attendance','Homework','CA_Score','Previous_Math']]
        y = data['Result']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier()

        model.fit(X_train, y_train)

        st.success("AI Model trained successfully")

        # Predictions
        data["Prediction"] = model.predict(X)

        # Grade estimation
        data["Predicted_Grade"] = data["Previous_Math"].apply(grade_category)

        # Risk level
        data["Risk_Level"] = data.apply(risk_level, axis=1)

        # Support recommendations
        data["Support"] = data.apply(recommend_support, axis=1)

        # -----------------------------
        # Show predictions
        # -----------------------------
        st.subheader("Predicted Student Results")

        st.dataframe(data[
            ["Attendance","Homework","CA_Score","Previous_Math",
             "Prediction","Predicted_Grade","Risk_Level"]
        ])

        # -----------------------------
        # National exam grade summary
        # -----------------------------
        st.subheader("National Exam Grade Prediction Summary")

        grade_summary = data["Predicted_Grade"].value_counts()

        st.dataframe(grade_summary)

        st.bar_chart(grade_summary)

        # Pie chart
        fig, ax = plt.subplots()

        ax.pie(
            grade_summary.values,
            labels=grade_summary.index,
            autopct='%1.1f%%'
        )

        ax.set_title("Predicted National Exam Performance")

        st.pyplot(fig)

        # -----------------------------
        # Early warning system
        # -----------------------------
        st.subheader("Early Warning System")

        high_risk = data[data["Risk_Level"] == "High Risk"]

        st.write("Students needing urgent academic intervention")

        st.dataframe(high_risk[
            ["Attendance","Homework","CA_Score",
             "Previous_Math","Prediction","Risk_Level"]
        ])

        # -----------------------------
        # Support recommendations
        # -----------------------------
        st.subheader("Student Support Recommendations")

        st.dataframe(data[
            ["Attendance","Homework","CA_Score",
             "Prediction","Risk_Level","Support"]
        ])


# -----------------------------
# Single student prediction
# -----------------------------
st.subheader("Single Student Prediction")

attendance = st.number_input("Attendance (%)",0,100)
homework = st.number_input("Homework Score (%)",0,100)
ca_score = st.number_input("CA Score (%)",0,100)
previous_math = st.number_input("Previous Math Score (%)",0,100)

if st.button("Predict Student Performance"):

    sample = pd.DataFrame([[attendance,homework,ca_score,previous_math]],
        columns=['Attendance','Homework','CA_Score','Previous_Math'])

    model = RandomForestClassifier()

    # dummy training so prediction works if dataset not uploaded
    model.fit([[50,50,50,50],[80,80,80,80]],[0,1])

    prediction = model.predict(sample)[0]

    grade = grade_category(previous_math)

    st.success(f"Predicted Result: {prediction}")

    st.info(f"Expected Grade: {grade}")
