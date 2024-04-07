import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import pickle
from sklearn.svm import SVC

st.header("Thyroid Prediction App")

path = "artifacts/transformer.csv"
data = pd.read_csv(path)

# load model

with open("thyroid_model.pkl", "rb") as f:
    svc = pickle.load(f)


add_selectbox = st.sidebar.selectbox(
    "Thyroid App", ("Intro", "DataFrame", "Prediction")
)


if add_selectbox == "Intro":
    st.write(
        "Thyroid disorders affect many people worldwide, the challenge for healthcare professionals in accurately diagnosing. This Thyroid App using Streamlit to streamline the diagnostic process and help healthcare professionals to make informed decisions."
    )
    st.write(
        "Note: This app should not be used for medical purposes, but only for research."
    )

elif add_selectbox == "DataFrame":
    st.dataframe(data, use_container_width=True)

else:
    st.write("#")

    st.subheader("Please select relevant features related to your case")

    age = st.number_input(
        "Enter your Age: ", min_value=0, max_value=100, key="age", step=1
    )

    left_column, right_column = st.columns(2)
    sex = ["Male", "Female"]
    tf = ["Yes", "No"]

    gender = st.radio("Gender: ", sex)
    if gender == "Male":
        gender = 1
    else:
        gender = 0

    on_thyroxine = st.radio("Are you on thyroxine?", tf)
    if tf == "yes":
        on_thyroxine = 1
    else:
        on_thyroxine = 0

    query_on_thyroxine = st.radio("query on thyroxine", tf)
    if tf == "yes":
        query_on_thyroxine = 1
    else:
        query_on_thyroxine = 0

    on_antithyroid_medication = st.radio("Are you on antithyroid medication?", tf)
    if tf == "yes":
        on_antithyroid_medication = 1
    else:
        on_antithyroid_medication = 0

    sick = st.radio("Are you sick?", tf)

    if tf == "yes":
        sick = 1
    else:
        sick = 0

    pregnant = 0
    if gender == "Female":
        pregnant = st.radio("Are you pregnant?", tf)
        if tf == "yes":
            pregnant = 1

    thyroid_surgery = st.radio("Have you undergone any thyroid related surgery?", tf)
    if tf == "yes":
        thyroid_surgery = 1
    else:
        thyroid_surgery = 0

    I131_treatment = st.radio("Are you undergoing I131 treatment?", tf)
    if tf == "yes":
        I131_treatment = 1
    else:
        I131_treatment = 0

    query_hypothyroid = st.radio("Do you think you have hypothyroid?", tf)
    if tf == "yes":
        query_hypothyroid = 1
    else:
        query_hypothyroid = 0

    query_hyperthyroid = st.radio("Do you think you have hyperthyroid?", tf)
    if tf == "yes":
        query_hyperthyroid = 1
    else:
        query_hyperthyroid = 0

    lithium = st.radio("Do you think you have lithium?", tf)
    if tf == "yes":
        lithium = 1
    else:
        lithium = 0

    goitre = st.radio("Do you have goitre?", tf)
    if tf == "yes":
        goitre = 1
    else:
        goitre = 0

    tumor = st.radio("Do you have any kind of tumor(s)?", tf)
    if tf == "yes":
        tumor = 1
    else:
        tumor = 0

    hypopituitary = st.radio("Do you have hypopituitary gland?", tf)
    if tf == "yes":
        hypopituitary = 1
    else:
        hypopituitary = 0

    psych = st.radio(
        "Do you have any psych conditions/diseases(eg. Anxiety Disorders, Depression, PTSD, Eating Disorders, etc.) ?",
        tf,
    )
    if tf == "yes":
        psych = 1
    else:
        psych = 0

    st.write("#")

    st.subheader(
        "The following features must be selected after getting your blood test done"
    )

    tsh_measured = st.radio("Whether TSH was measured in the blood ?", tf)

    tsh = st.number_input(
        "Enter TSH level in blood from lab work ",
        min_value=0.005,
        max_value=535.000,
        key="TSH",
        step=0.001,
    )

    st.write("#")

    t3_measured = st.radio("Whether T3 was measured in the blood ?", tf)

    t3 = st.number_input(
        "Enter T3 level in blood from lab work ",
        min_value=0.005,
        max_value=20.000,
        key="T3",
        step=0.001,
    )

    st.write("#")

    tt4_measured = st.radio("Whether TT4 was measured in the blood ?", tf)

    tt4 = st.number_input(
        "Enter TT4 level in blood from lab work ",
        min_value=0.000,
        max_value=700.000,
        key="TT4",
        step=0.001,
    )

    st.write("#")

    t4u_measured = st.radio("Whether T4U was measured in the blood ?", tf)

    t4u = st.number_input(
        "Enter T4U level in blood from lab work ",
        min_value=0.100,
        max_value=3.000,
        key="T4U",
        step=0.001,
    )

    st.write("#")

    fti_measured = st.radio("Whether FTI was measured in the blood ?", tf)

    fti = st.number_input(
        "Enter FTI level in blood from lab work ",
        min_value=0.000,
        max_value=900.000,
        key="FTI",
        step=0.001,
    )
    st.write("#")

    ref_source = st.number_input(
        "Enter Referral Source ", min_value=1, max_value=4, key="Source", step=1
    )

    st.write("#")

    if st.button("Make Prediction"):
        inputs = (
            age,
            gender,
            on_thyroxine,
            query_on_thyroxine,
            on_antithyroid_medication,
            sick,
            pregnant,
            thyroid_surgery,
            I131_treatment,
            query_hypothyroid,
            query_hyperthyroid,
            lithium,
            goitre,
            tumor,
            hypopituitary,
            psych,
            tsh,
            t3,
            tt4,
            t4u,
            fti,
            ref_source,
        )
        # print(inputs)

        move = np.array(inputs)  # .values.reshape(1,-1)
        # print(move)
        pred_input = move.reshape(1, -1)

        prediction = svc.predict(pred_input)
        st.write("#")

        # st.write(prediction)

        if prediction == 0:
            st.write("Compensated Hypothyroid")
        elif prediction == 1:
            st.write("Negative")
        elif prediction == 2:
            st.write("Primary Hypothyroid")
        else:
            st.write("Secondary Hypothyroid")
