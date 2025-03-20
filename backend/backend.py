from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # <-- ADD THIS IMPORT
import os  # to make sure the directory exists
from fastapi.responses import FileResponse


app = FastAPI()

# Allow CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ADD DOWNLOAD ENDPOINT BELOW ⬇️
@app.get("/download_model/")
def download_model():
    file_path = "models/model.pkl"
    return FileResponse(path=file_path, filename="model.pkl", media_type='application/octet-stream')


@app.post("/train_model/")
async def train_model(
    file: UploadFile,
    split_size: float = Form(...),
    n_estimators: int = Form(...),
    max_features: str = Form(...),
    min_samples_split: int = Form(...),
    min_samples_leaf: int = Form(...),
    random_state: int = Form(...),
    criterion: str = Form(...),
    bootstrap: bool = Form(...),
    oob_score: bool = Form(...),
    n_jobs: int = Form(...),
):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))

    # Split into X and Y
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]

    # Encode categorical variables
    X = pd.get_dummies(X)

    # If target (Y) is categorical, encode it too
    if Y.dtype == 'object':
        Y = pd.factorize(Y)[0]

    # Split dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=(100 - split_size) / 100, random_state=random_state
    )

    # Random Forest Model
    rf = RandomForestRegressor(
        criterion=criterion,
        n_estimators=n_estimators,
        random_state=random_state,
        max_features=max_features if max_features != "None" else None,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        oob_score=oob_score,
        n_jobs=n_jobs
    )

    # Train model
    rf.fit(X_train, Y_train)

    # MAKE SURE 'models' FOLDER EXISTS
    os.makedirs("models", exist_ok=True)
    
    # SAVE THE MODEL HERE 🚩
    joblib.dump(rf, "models/model.pkl")

    # Return metrics
    result = {
        "train_r2": r2_score(Y_train, rf.predict(X_train)),
        "train_mse": mean_squared_error(Y_train, rf.predict(X_train)),
        "test_r2": r2_score(Y_test, rf.predict(X_test)),
        "test_mse": mean_squared_error(Y_test, rf.predict(X_test)),
        "model_params": rf.get_params(),
        "message": "Model saved to models/model.pkl ✅"
    }

    return result
