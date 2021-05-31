import streamlit as st
import pickle
import numpy
import xgboost

st.title("Covid-19 Prediction using Blood Test")

NRS = ['Note Red Series_Anisocitose +',
       'Note Red Series_Anisocitose ++', 'Note Red Series_Anisocitose +++',
       'Note Red Series_Eritrócitos normais em tamanho',
       'Note Red Series_Macrocitose +', 'Note Red Series_Macrocitose ++',
       'Note Red Series_Macrocitose +++', 'Note Red Series_Microcitose +',
       'Note Red Series_Microcitose ++', 'Note Red Series_Microcitose +++']

with st.form("my_form"):
    col1, col2, col3, col4 = st.beta_columns(4)
    with col1:
        hcm = st.number_input("HCM", 0, 100, 0)
        hemoglobin = st.number_input("Hemoglobin", 0, 100, 0)
        mchc = st.number_input("MCHC", 0, 100, 0)
        rdw_cv = st.number_input("RDW-CV", 0, 100, 0)
    with col2:
        rdw_sd = st.number_input("RDW-SD", 0, 100, 0)
        vcm = st.number_input("VCM", 0, 100, 0)
        basophils = st.number_input("basophils", 0, 100, 0)
        eosinophils = st.number_input("eosinophils", 0, 100, 0)
    with col3:
        erythroblasts = st.number_input("erythroblasts", 0, 100, 0)
        erythrocytes = st.number_input("erythrocytes", 0, 100, 0)
        hematocrit = st.number_input("hematocrit", 0, 100, 0)
        leukocytes = st.number_input("leukocytes", 0, 100, 0)
    with col4:
        lymphocytes = st.number_input("lymphocytes", 0, 100, 0)
        monocytes = st.number_input("monocytes", 0, 100, 0)
        neutrophils = st.number_input("neutrophils", 0, 100, 0)
        DAY_DIFFERENCE = st.number_input("DAY_DIFFERENCE", 0, 100, 0)

    NRS_values = {
        "Note Red Series_Anisocitose +": 0,
        "Note Red Series_Anisocitose ++": 0,
        "Note Red Series_Anisocitose +++": 0,
        "Note Red Series_Eritrócitos normais em tamanho": 0,
        "Note Red Series_Macrocitose +": 0,
        "Note Red Series_Macrocitose ++": 0,
        "Note Red Series_Macrocitose +++": 0,
        "Note Red Series_Microcitose +": 0,
        "Note Red Series_Microcitose ++": 0,
        "Note Red Series_Microcitose +++": 0,
    }
    NRS_option = st.selectbox("Select Note Red Series observation", NRS)
    NRS_values[NRS_option] = 1;

    submitted = st.form_submit_button("Submit")

classifier = pickle.load(open("covid-xgb.pickle.dat", "rb"))

predict_data = [hcm, hemoglobin, mchc, rdw_cv, rdw_sd, vcm, basophils,
                eosinophils, erythroblasts, erythrocytes, hematocrit,
                leukocytes, lymphocytes, monocytes, neutrophils,
                DAY_DIFFERENCE, NRS_values['Note Red Series_Anisocitose +'],
                NRS_values['Note Red Series_Anisocitose ++'],
                NRS_values['Note Red Series_Anisocitose +++'],
                NRS_values['Note Red Series_Eritrócitos normais em tamanho'],
                NRS_values['Note Red Series_Macrocitose +'],
                NRS_values['Note Red Series_Macrocitose ++'],
                NRS_values['Note Red Series_Macrocitose +++'],
                NRS_values['Note Red Series_Microcitose +'],
                NRS_values['Note Red Series_Microcitose ++'],
                NRS_values['Note Red Series_Microcitose +++']]

prediction = classifier.predict(numpy.array(predict_data).reshape((1, -1)))

st.success("Covid-19 Negative" if prediction[0] == 0 else "Covid-19 Positive")
