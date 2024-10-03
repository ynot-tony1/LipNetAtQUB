from fastapi import FastAPI, File, UploadFile
import shutil
import os
from evaluation.predict import predict  # Adjust this import based on your structure

app = FastAPI()

# Define the path for the video samples
EVALUATION_SAMPLES_PATH = os.path.join("evaluation", "samples")

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to the correct path
        file_location = os.path.join(EVALUATION_SAMPLES_PATH, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Call the predict function with the correct paths for the weights and video
        weight_path = "evaluation/models/overlapped-weights368.h5"
        response = predict(weight_path, file_location, 32, 28)  # Adjust args if needed

        # The `response` will include both "decoded_text" and "timestamps"
        decoded_text = response.get("decoded_text", "")
        timestamps = response.get("timestamps", [])

        # Return both the decoded text and timestamps as the result
        return {"decoded_text": decoded_text, "timestamps": timestamps}

    except Exception as e:
        return {"error": str(e)}
