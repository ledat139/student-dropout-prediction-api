from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load("./model.pkl")
label_encoders = joblib.load("./label_encoders.pkl")

app = FastAPI()

class StudentData(BaseModel):
    STDNT_TEST_ENTRANCE_COMB: float
    UNMET_NEED: float
    HIGH_SCHL_GPA: float
    FIRST_TERM_Hr: float
    SECOND_TERM_Hr: float
    COST_OF_ATTEND: float
    DISTANCE_FROM_HOME: float
    EST_FAM_CONTRIBUTION: float
    GROSS_FIN_NEED: float

@app.post("/predict")
def predict(data: StudentData):
    # Chuyển input thành DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    # Encode nếu có cột dạng object (không bắt buộc trong trường hợp bạn đã loại hết object)
    for col in input_df.columns:
        if col in label_encoders:
            le = label_encoders[col]
            input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Dự đoán
    prediction = model.predict(input_df)[0]
    return {
        "prediction": int(prediction),
        "meaning": "Quay lại năm 2" if prediction == 1 else "Không quay lại"
    }
