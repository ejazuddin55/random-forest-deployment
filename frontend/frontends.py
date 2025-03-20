# frontend/app.py
import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title='ML App Frontend', layout='wide')

st.title('ML Model Builder - Frontend')

# Sidebar for file upload
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Sidebar for hyperparameters
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Train split %', 10, 90, 80, 5)
    n_estimators = st.sidebar.slider('n_estimators', 0, 1000, 100, 100)
    max_features = st.sidebar.selectbox('max_features', ['sqrt', 'log2', 'None'])
    min_samples_split = st.sidebar.slider('min_samples_split', 1, 10, 2)
    min_samples_leaf = st.sidebar.slider('min_samples_leaf', 1, 10, 2)
    random_state = st.sidebar.slider('random_state', 0, 1000, 42)
    criterion = st.sidebar.selectbox('criterion', ['squared_error', 'absolute_error'])
    bootstrap = st.sidebar.selectbox('bootstrap', [True, False])
    oob_score = st.sidebar.selectbox('oob_score', [False, True])
    n_jobs = st.sidebar.selectbox('n_jobs', [1, -1])

# Display uploaded file and send to backend
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader('Uploaded Dataset')
    st.write(df.head())

    if st.button('Train Model'):
        with st.spinner('Sending data to backend...'):
            response = requests.post(
                url="https://back.grayhill-f4c60921.westus2.azurecontainerapps.io/train_model/",
                files={"file": uploaded_file.getvalue()},
                data={
                    "split_size": split_size,
                    "n_estimators": n_estimators,
                    "max_features": max_features,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "random_state": random_state,
                    "criterion": criterion,
                    "bootstrap": bootstrap,
                    "oob_score": oob_score,
                    "n_jobs": n_jobs
                }
            )

            if response.status_code == 200:
                result = response.json()
                st.subheader('Model Performance')
                st.write(result)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
