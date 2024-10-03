from flask import request, jsonify
from . import app
from lipnet.lipreading.videos import Video  # Assuming you need these imports
from lipnet.model2 import LipNet
from lipnet.lipreading.helpers import labels_to_text
import numpy as np

@app.route('/decode', methods=['POST'])
def decode_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file, process it with LipNet, and return decoded result
    file.save("/tmp/" + file.filename)
    video = Video(vtype='face', face_predictor_path='path_to_predictor')
    video.from_video("/tmp/" + file.filename)

    # LipNet processing steps, similar to your predict.py logic
    lipnet = LipNet(img_c=3, img_w=100, img_h=50, frames_n=75)
    X_data = np.array([video.data]).astype(np.float32) / 255
    y_pred = lipnet.predict(X_data)

    result = "decoded text here"  # Replace with actual result logic

    return jsonify({"decoded_text": result})
