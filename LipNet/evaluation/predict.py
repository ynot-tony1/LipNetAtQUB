from lipnet.lipreading.videos import Video
from lipnet.lipreading.helpers import labels_to_text
from lipnet.utils.spell import Spell
from lipnet.model2 import LipNet
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import numpy as np
import sys
import os
import cv2
import dlib
import subprocess

np.random.seed(55)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
FACE_PREDICTOR_PATH = os.path.join(
    CURRENT_PATH, '..', 'common', 'predictors', 'shape_predictor_68_face_landmarks.dat')
PREDICT_DICTIONARY = os.path.join(
    CURRENT_PATH, '..', 'common', 'dictionaries', 'grid.txt')

# Define the start_time as 0 for standalone video
START_TIME = 0.0  # seconds

def detect_speech_onset(video_path, start_time=START_TIME, movement_threshold=3000):
    """
    Detects the speech onset in the video by analyzing mouth movement.

    Parameters:
    - video_path (str): Path to the video file.
    - start_time (float): Offset to start the timestamp calculation.
    - movement_threshold (int): Threshold for mouth movement to detect onset.

    Returns:
    - speech_onset_time (float): Time in seconds when speech starts.
    """
    # Initialize dlib's face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FACE_PREDICTOR_PATH)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return 0.0  # Default to 0 if video cannot be read

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_count = 1
    mouth_movement = []
    speech_onset_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        rects = detector(gray, 0)
        if len(rects) > 0:
            # Assume the first detected face is the speaker
            rect = rects[0]
            # Get facial landmarks
            shape = predictor(gray, rect)
            # Get mouth landmarks
            mouth_points = []
            for i in range(48, 68):  # Mouth landmarks
                x = shape.part(i).x
                y = shape.part(i).y
                mouth_points.append((x, y))
            # Create a mask for the mouth region
            mask = np.zeros_like(gray)
            mouth_hull = cv2.convexHull(np.array(mouth_points))
            cv2.drawContours(mask, [mouth_hull], -1, 255, -1)
            # Extract mouth region
            mouth_region = cv2.bitwise_and(gray, gray, mask=mask)
            # Calculate difference with previous frame
            mouth_diff = cv2.absdiff(prev_gray, gray)
            mouth_diff = cv2.bitwise_and(mouth_diff, mouth_diff, mask=mask)
            movement = np.sum(mouth_diff)
            mouth_movement.append(movement)

            # Log movement per frame
            print(f"Frame {frame_count}: Movement={movement}")

            if movement > movement_threshold and speech_onset_frame is None:
                speech_onset_frame = frame_count
                print(f"Speech onset detected at frame {speech_onset_frame}")
                break  # Stop after finding the speech onset

        prev_gray = gray.copy()
        frame_count += 1

    cap.release()

    if speech_onset_frame is not None:
        # Calculate speech onset time without adding start_time
        speech_onset_time = speech_onset_frame / fps + start_time
    else:
        speech_onset_time = start_time  # Default to start_time if speech onset not detected

    print(f"Speech onset detected at frame {speech_onset_frame}, time {speech_onset_time:.2f}s")
    return speech_onset_time

def get_video_duration(video_path):
    """
    Use ffprobe to get the start_time and duration of the video.
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'format=start_time,duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stdout.strip().split('\n')
    if len(output) >= 2:
        try:
            start_time = float(output[0])
            duration = float(output[1])
            return start_time, duration
        except ValueError:
            pass
    # Fallback if ffprobe fails
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return START_TIME, frame_count / fps  # Use predefined START_TIME

def predict(weight_path, video_path, absolute_max_string_len=32, output_size=28):
    # Ensure parameters are integers
    absolute_max_string_len = int(absolute_max_string_len)
    output_size = int(output_size)

    print("\nLoading video data from disk...")

    # Load the video
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    if os.path.isfile(video_path):
        video.from_video(video_path)
    else:
        print(f"Video file not found: {video_path}")
        return None

    print("Data loaded successfully.\n")

    # Extract video metadata using ffprobe
    metadata = get_video_metadata(video_path)
    if not metadata:
        print("Failed to retrieve video metadata.")
        return None

    start_time = metadata['start_time']
    video_duration = metadata['duration']
    nb_frames = metadata['nb_frames']
    fps = metadata['fps']

    print(f"Video Start Time: {start_time}s")
    print(f"Video Duration: {video_duration}s")
    print(f"Total Frames: {nb_frames}")
    print(f"Video FPS: {fps}")

    # Detect speech onset time
    # Set movement_threshold to a lower value to detect earlier speech onset
    speech_onset_time = detect_speech_onset(video_path, start_time=start_time, movement_threshold=3000)

    if speech_onset_time > (start_time + video_duration):
        print("Error: Speech onset time exceeds video duration.")
        return None

    if K.image_data_format() == 'channels_first':
        img_c, frames_n, img_w, img_h = video.data.shape
    else:
        frames_n, img_w, img_h, img_c = video.data.shape

    print(f"Initializing LipNet model with parameters:")
    print(f"img_c={img_c}, img_w={img_w}, img_h={img_h}, frames_n={frames_n}, "
          f"absolute_max_string_len={absolute_max_string_len}, output_size={output_size}")

    # Build and load the LipNet model
    lipnet = LipNet(
        img_c=img_c,
        img_w=img_w,
        img_h=img_h,
        frames_n=frames_n,
        absolute_max_string_len=absolute_max_string_len,
        output_size=output_size
    )

    print("Building the LipNet model...")
    lipnet.build()  # Ensure that the model uses unidirectional GRUs
    print("LipNet model built successfully.")

    adam = Adam(lr=0.0001)
    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    lipnet.model.load_weights(weight_path)  # Load the weights of the unidirectional model

    # Create the prediction model
    input_data = lipnet.model.get_layer('the_input').input
    y_pred = lipnet.model.get_layer('softmax').output

    prediction_model = Model(inputs=input_data, outputs=y_pred)

    spell = Spell(path=PREDICT_DICTIONARY)

    X_data = np.array([video.data]).astype(np.float32) / 255

    print("Starting prediction...")
    y_pred = prediction_model.predict(X_data)

    # Define labels as a Python list
    labels = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        ' ', ''  # space and blank labels
    ]

    # Map labels to characters
    label_to_char = {i: c for i, c in enumerate(labels)}

    # Get the blank label index
    blank_label = len(labels) - 1  # Assuming blank label is the last label

    # Get per-frame predictions and probabilities
    y_pred_probs = y_pred[0]  # Shape: (time_steps, num_classes)
    y_pred_labels = np.argmax(y_pred_probs, axis=-1)  # Shape: (time_steps,)

    # Calculate time_per_step based on video duration and model time steps
    model_time_steps = len(y_pred_labels)
    remaining_duration = (start_time + video_duration) - speech_onset_time

    # Validate remaining_duration
    if remaining_duration <= 0:
        print("Error: Speech onset time exceeds video duration.")
        return None

    time_per_step = remaining_duration / model_time_steps
    print(f"Time per model step: {time_per_step:.4f}s")

    # Process the predictions to get characters and timestamps
    prev_label = blank_label
    label_seq = []
    for t in range(len(y_pred_labels)):
        label = y_pred_labels[t]

        if label == prev_label:
            continue  # Skip duplicates
        if label == blank_label:
            prev_label = label
            continue

        char = label_to_char.get(label, '')
        # Calculate adjusted_time based on time_per_step
        adjusted_time = (t * time_per_step) + speech_onset_time

        # Ensure adjusted_time does not exceed video timeline
        if adjusted_time > (start_time + video_duration):
            adjusted_time = start_time + video_duration

        label_seq.append({'char': char, 'time': adjusted_time})
        prev_label = label

    # Group characters into words
    words = []
    current_word = ''
    word_start_time = None
    for item in label_seq:
        char = item['char']
        time = item['time']

        if char == ' ':
            if current_word:
                words.append({'word': current_word, 'start_time': word_start_time})
                current_word = ''
                word_start_time = None
        else:
            if current_word == '':
                word_start_time = time
            current_word += char

    # Add the last word
    if current_word:
        words.append({'word': current_word, 'start_time': word_start_time})

    # Correct the decoded text
    decoded_text = ' '.join([w['word'] for w in words])
    corrected_text = spell.sentence(decoded_text)
    corrected_words = corrected_text.strip().split()

    # Map corrected words back to timestamps
    if len(corrected_words) == len(words):
        for i in range(len(words)):
            words[i]['word'] = corrected_words[i]
    else:
        print("Warning: Corrected words do not match the number of detected words.")
        if words:
            words = [{'word': corrected_text, 'start_time': words[0]['start_time']}]
        else:
            words = [{'word': corrected_text, 'start_time': speech_onset_time}]

    # Output the decoded text and timestamps
    print(f"Decoded Text: {corrected_text}")
    for w in words:
        print(f"Word: {w['word']}, Start Time: {w['start_time']:.2f}s")

    return {'decoded_text': corrected_text, 'timestamps': words}



import subprocess
import logging
import cv2

def get_video_metadata(video_path):
    """
    Extracts video metadata using ffprobe.

    Parameters:
    - video_path (str): Path to the video file.

    Returns:
    - metadata (dict): Dictionary containing video metadata.
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-count_frames',  # Correct usage
        '-show_entries', 'format=start_time,duration,nb_read_frames,avg_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        output = result.stdout.strip().split('\n')
        if len(output) >= 4:
            start_time = float(output[0])
            duration = float(output[1])
            nb_frames = int(output[2])  # 'nb_read_frames' from ffprobe
            avg_frame_rate = output[3]
            # Calculate FPS from avg_frame_rate
            if '/' in avg_frame_rate:
                num, denom = avg_frame_rate.split('/')
                fps = float(num) / float(denom) if float(denom) != 0 else 25.0  # Default to 25 if denom is 0
            else:
                fps = float(avg_frame_rate) if avg_frame_rate else 25.0
            metadata = {
                'start_time': start_time,
                'duration': duration,
                'nb_frames': nb_frames,
                'fps': fps
            }
            return metadata
        else:
            logging.warning("ffprobe output incomplete. Falling back to OpenCV.")
    except subprocess.CalledProcessError as e:
        logging.error(f"ffprobe error: {e.stderr}")
    except Exception as e:
        logging.error(f"Unexpected error while running ffprobe: {e}")
    
    # Fallback using OpenCV if ffprobe fails
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Failed to open video with OpenCV.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0
    cap.release()
    metadata = {
        'start_time': 0.0,  # Assume 0 if ffprobe fails
        'duration': duration,
        'nb_frames': frame_count,
        'fps': fps
    }
    return metadata




if __name__ == '__main__':
    if len(sys.argv) >= 3:
        weight_path = sys.argv[1]
        video_path = sys.argv[2]
        if len(sys.argv) >= 4:
            absolute_max_string_len = int(sys.argv[3])
        else:
            absolute_max_string_len = 32  # default value
        if len(sys.argv) >= 5:
            output_size = int(sys.argv[4])
        else:
            output_size = 28  # default value

        response = predict(weight_path, video_path, absolute_max_string_len, output_size)
    else:
        print("Usage: python predict.py [weights path] [video path] [optional: absolute_max_string_len] [optional: output_size]")
        sys.exit()

    if response and "decoded_text" in response:
        decoded_text = response["decoded_text"]
        timestamps = response["timestamps"]

        stripe = "-" * len(decoded_text)
        print(" __                   __  __          __      ")
        print("/\\ \\       __        /\\ \\/\\ \\        /\\ \\__   ")
        print("\\ \\ \\     /\\_\\  _____\\ \\ `\\\\ \\     __\\ \\ ,_\\  ")
        print(" \\ \\ \\  __\\/\\ \\/\\ '__`\\ \\ , ` \\  /'__`\\ \\ \\/  ")
        print("  \\ \\ \\L\\ \\\\ \\ \\ \\ \\L\\ \\ \\ \\`\\ \\/\\  __/\\ \\ \\_ ")
        print("   \\ \\____/ \\ \\_\\ \\ ,__/\\ \\_\\ \\_\\ \\____\\\\ \\__\\")
        print("    \\/___/   \\/_/\\ \\ \\/  \\/_/\\/_/\\/____/ \\/__/")
        print("                  \\ \\_\\                       ")
        print("                   \\/_/                       ")
        print(f"             --{stripe}- ")
        print(f"[ DECODED ] |> {decoded_text} |")
        print(f"             --{stripe}- ")

        # Display word start times
        for word_info in timestamps:
            print(f"Word: {word_info['word']}, Start Time: {word_info['start_time']:.2f}s")
