import os
import sys
import numpy as np
import cv2
import dlib
import subprocess
import librosa



import matplotlib.pyplot as plt
import librosa.display
from collections import Counter
import collections  # Ensure collections is imported before usage
import logging
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.backend import ctc_decode, get_value

from lipnet.lipreading.videos import Video
from lipnet.lipreading.helpers import labels_to_text
from lipnet.model2 import LipNet

import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("lipnet_predict.log"),  # Log to file
        logging.StreamHandler()  # Also log to console
    ]
)

# Set random seeds for reproducibility
np.random.seed(55)
tf.compat.v1.set_random_seed(55)  # Updated for TensorFlow 1.x

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
FACE_PREDICTOR_PATH = os.path.join(CURRENT_PATH, '..', 'common', 'predictors', 'shape_predictor_68_face_landmarks.dat')


def extract_audio(video_path, audio_path='extracted_audio.wav'):
    """
    Extracts audio from the given video file using ffmpeg.

    Args:
        video_path (str): Path to the input video file.
        audio_path (str): Path to save the extracted audio file.

    Returns:
        str or None: Path to the extracted audio file, or None if extraction failed.
    """
    command = [
        'ffmpeg', '-y', '-i', video_path,
        '-b:a', '160k',  # Updated from -ab to -b:a
        '-ac', '2', '-ar', '44100', '-vn', audio_path
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logging.debug(f"Audio extracted to {audio_path}")
    except subprocess.CalledProcessError:
        logging.debug("Failed to extract audio. Ensure ffmpeg is installed, accessible, and the video has an audio stream.")
        return None
    return audio_path


def plot_onsets(y, sr, onset_times, title='Audio Waveform with Detected Onsets', save_path='onsets_plot.png'):
    """
    Plots the audio waveform with detected onsets and saves the plot to a file.

    Args:
        y (np.array): Audio time series.
        sr (int): Sampling rate.
        onset_times (list): List of onset times in seconds.
        title (str): Title of the plot.
        save_path (str): File path to save the plot.
    """
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    for onset in onset_times:
        plt.axvline(onset, color='r', linestyle='--', alpha=0.9)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig(save_path)
    plt.close()
    logging.debug(f"Onset plot saved to '{save_path}'")


def detect_audio_onsets(audio_path, sr=44100, hop_length=512, backtrack=True, save_plot=True):
    """
    Detects speech onsets in the given audio file using librosa.

    Args:
        audio_path (str): Path to the audio file.
        sr (int): Sampling rate.
        hop_length (int): Number of samples between successive frames.
        backtrack (bool): Whether to backtrack the detected onset to the nearest preceding minimum.
        save_plot (bool): Whether to save the onset plot.

    Returns:
        list: List of onset times in seconds.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=backtrack)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        logging.debug(f"Detected {len(onset_times)} speech onsets in audio using librosa.")

        if save_plot:
            plot_onsets(y, sr, onset_times)

        return onset_times.tolist()
    except Exception as e:
        logging.debug(f"Error in detecting audio onsets with librosa: {e}")
        return []





# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='lipnet_predict.log',
                    format='%(asctime)s [%(levelname)s] %(message)s')

def detect_speech_onsets(video_path, movement_multiplier=2.0, initial_frame_count=15, 
                        frame_queue_length=5, desired_threshold=0.015, verbose=True):
    """
    Detects speech onsets in the given video file using both audio and video analysis.

    Args:
        video_path (str): Path to the video file.
        movement_multiplier (float): Multiplier applied to the mean movement for thresholding.
        initial_frame_count (int): Number of initial frames to establish a movement baseline.
        frame_queue_length (int): Number of frames to consider for temporal smoothing.
        desired_threshold (float): Fixed threshold for movement detection.
        verbose (bool): Whether to print debug statements.

    Returns:
        tuple: Tuple containing two lists:
               - audio_onset_times (list): List of audio onset times in seconds.
               - video_onset_times (list): List of video onset times in seconds.
    """
    detector = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor(FACE_PREDICTOR_PATH)
    except RuntimeError as e:
        logging.error(f"Error loading shape predictor: {e}")
        return [], []

    # Extract audio from video
    audio_path = extract_audio(video_path)
    if not audio_path:
        if verbose:
            logging.debug("Audio extraction failed. Proceeding with video-based onset detection only.")
        audio_onsets = []
    else:
        # Detect audio-based onsets
        audio_onsets = detect_audio_onsets(audio_path)
        if verbose:
            logging.debug(f"Detected {len(audio_onsets)} speech onsets in audio using librosa.")

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        if verbose:
            logging.debug("FPS value is invalid or could not be read. Check the video file.")
        cap.release()
        return audio_onsets, []

    if verbose:
        logging.debug(f"FPS of video: {fps}")

    ret, prev_frame = cap.read()
    if not ret:
        if verbose:
            logging.debug("Failed to read the first frame of the video.")
        cap.release()
        return audio_onsets, []

    # Convert to grayscale and apply Gaussian blur
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    frame_count = 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize variables for adaptive thresholding
    movement_values = []
    movement_threshold = desired_threshold  # Set directly to desired_threshold
    movement_queue = collections.deque(maxlen=frame_queue_length)

    speech_onsets_video = []

    while frame_count <= total_frames:
        ret, frame = cap.read()
        if not ret:
            if verbose:
                logging.debug("End of video reached or failed to read frame.")
            break

        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect faces in the frame
        rects = detector(gray, 0)
        if len(rects) > 0:
            # Assume the first detected face is the target
            rect = rects[0]
            shape = predictor(gray, rect)
            mouth_points = [(shape.part(i).x, shape.part(i).y) for i in range(48, 68)]

            # Create a mask for the mouth region
            mask = np.zeros_like(gray)
            mouth_hull = cv2.convexHull(np.array(mouth_points))
            cv2.drawContours(mask, [mouth_hull], -1, 255, -1)

            # Compute difference between current and previous frame in the mouth region
            mouth_diff = cv2.absdiff(prev_gray, gray)
            mouth_diff = cv2.bitwise_and(mouth_diff, mouth_diff, mask=mask)

            # Calculate movement with temporal smoothing
            movement = np.mean(mouth_diff) * movement_multiplier
            movement_queue.append(movement)
            smoothed_movement = np.mean(movement_queue)

            # Calculate timestamp for the current frame
            timestamp = frame_count / fps

            if frame_count <= initial_frame_count:
                movement_values.append(smoothed_movement)
                if verbose:
                    logging.debug(f"Initial Frame {frame_count}, Smoothed Movement: {smoothed_movement:.4f}")
            else:
                # Detect speech onset based on movement threshold
                # Adding a minimum separation of 1 second between onsets to avoid multiple detections
                if (smoothed_movement > movement_threshold and
                    (len(speech_onsets_video) == 0 or (timestamp - speech_onsets_video[-1]['timestamp'] > 1.0))):
                    speech_onsets_video.append({'frame': frame_count, 'timestamp': timestamp})
                    if verbose:
                        logging.debug(f"Speech onset detected at frame {frame_count} (Timestamp: {timestamp:.2f}s)")

                # Log movements exceeding the threshold
                if smoothed_movement > movement_threshold:
                    if verbose:
                        logging.debug(f"Frame {frame_count}, Timestamp: {timestamp:.2f}s, Smoothed Movement Detected: {smoothed_movement:.4f} > Threshold: {movement_threshold:.4f}")

            # Optional: Save the mouth area overlay for debugging (only for first few frames)
            if frame_count <= 5:
                overlay = cv2.bitwise_and(gray, gray, mask=mask)
                overlay_path = f'mouth_overlay_frame_{frame_count}.png'
                cv2.imwrite(overlay_path, overlay)
                if verbose:
                    logging.debug(f"Saved mouth overlay for frame {frame_count} as '{overlay_path}'")

        # Update previous frame
        prev_gray = gray.copy()
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Convert video-based onsets to time in seconds
    video_onset_times = [onset['timestamp'] for onset in speech_onsets_video]
    if verbose:
        logging.debug(f"Detected {len(video_onset_times)} speech onsets via video.")

    return audio_onsets, video_onset_times





def map_ctc_to_timestamps_lipnet(decoded_pred, fps, audio_onset_times, video_onset_times, min_separation=0.3, tolerance=0.2):
    """
    Maps decoded CTC predictions to speech onset timestamps by aligning entire words with onsets,
    ensuring a minimum separation between consecutive word timestamps.

    Args:
        decoded_pred (list): Decoded CTC predictions as a list of label indices.
        fps (float): Frames per second of the video.
        audio_onset_times (list): List of detected speech onset times in seconds (audio).
        video_onset_times (list): List of detected speech onset times in seconds (video).
        min_separation (float): Minimum separation in seconds between consecutive word timestamps.
        tolerance (float): Time tolerance in seconds for validating onsets.

    Returns:
        list: List of dictionaries containing words and their start times.
    """
    labels = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        ' ', ''  # space and blank labels
    ]

    label_to_char = {i: c for i, c in enumerate(labels)}
    blank_label = len(labels) - 1  # 27

    # Convert decoded_pred to characters, excluding blanks
    characters = [label_to_char.get(label, '') for label in decoded_pred if label != blank_label]

    # Collapse repeated characters to prevent duplication
    collapsed = []
    prev_char = None
    for c in characters:
        if c != prev_char:
            collapsed.append(c)
            prev_char = c

    # Form the decoded string
    decoded_string = ''.join(collapsed)

    # Split the string into words
    words = decoded_string.strip().split(' ')

    # Assign speech onsets to words based on validated onsets
    word_timestamps = []
    last_onset = -min_separation  # Initialize to allow first word assignment

    for word in words:
        # Find the first audio onset that has a corresponding video onset within the tolerance
        suitable_onset = None
        for audio_onset in audio_onset_times:
            if audio_onset >= last_onset + min_separation:
                if validate_onset(audio_onset, video_onset_times, tolerance):
                    suitable_onset = audio_onset
                    break

        if suitable_onset is not None:
            word_timestamps.append({'word': word, 'start_time': suitable_onset})
            last_onset = suitable_onset  # Update the last assigned onset
        else:
            # If no suitable onset is found, assign the last available onset or mark as not available
            onset = audio_onset_times[-1] if audio_onset_times else 0
            word_timestamps.append({'word': word, 'start_time': onset})
            logging.debug(f"Word '{word}' could not be assigned a unique onset. Assigned to {onset:.2f}s.")

    # Optionally, log the word timestamps
    for word_info in word_timestamps:
        word = word_info['word']
        start_time = word_info['start_time']
        logging.debug(f"Word: {word}, Start Time: {start_time:.2f}s")

    return word_timestamps

def validate_onset(audio_onset, video_onsets, tolerance=0.2):
    """
    Validates if an audio onset has a corresponding video onset within the tolerance.

    Args:
        audio_onset (float): The audio onset time in seconds.
        video_onsets (list): List of video onset times in seconds.
        tolerance (float): Time tolerance in seconds.

    Returns:
        bool: True if a corresponding video onset is found, False otherwise.
    """
    for video_onset in video_onsets:
        if abs(video_onset - audio_onset) <= tolerance:
            return True
    return False



def predict(weight_path, video_path, absolute_max_string_len=32, output_size=28):
    """
    Performs lip-reading on the given video using the LipNet model.

    Args:
        weight_path (str): Path to the LipNet model weights.
        video_path (str): Path to the input video file.
        absolute_max_string_len (int): Maximum length of the predicted string.
        output_size (int): Number of classes for prediction.

    Returns:
        dict: Decoded text and word timestamps.
    """

    # Reset the default graph to avoid duplication
    tf.compat.v1.reset_default_graph()

    # Convert the parameters to integers if necessary
    absolute_max_string_len = int(absolute_max_string_len)
    output_size = int(output_size)

    logging.info("Loading video data from disk...")

    # Load the video data
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    if not os.path.isfile(video_path):
        logging.error(f"Video file not found: {video_path}")
        return None

    try:
        video.from_video(video_path)
    except Exception as e:
        logging.error(f"Failed to load video data: {e}")
        return None

    # Initialize video capture to extract FPS
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()  # Release the video capture
    logging.info(f"Video FPS: {fps}")

    if fps <= 0:
        fps = 25  # Default to 25 FPS
        logging.warning("Invalid FPS detected. Defaulting to 25 FPS.")

    # Detect speech onsets
    speech_onset_times = detect_speech_onsets(video_path)

    # Determine video dimensions based on the Keras image format
    if K.image_data_format() == 'channels_first':
        img_c, frames_n, img_w, img_h = video.data.shape
    else:
        frames_n, img_w, img_h, img_c = video.data.shape

    # Initialize LipNet model
    logging.info("Initializing LipNet model with parameters:")
    logging.info(f"       img_c={img_c}, img_w={img_w}, img_h={img_h}, frames_n={frames_n}, "
                 f"absolute_max_string_len={absolute_max_string_len}, output_size={output_size}")

    lipnet = LipNet(
        img_c=img_c,
        img_w=img_w,
        img_h=img_h,
        frames_n=frames_n,
        absolute_max_string_len=absolute_max_string_len,
        output_size=output_size
    )

    try:
        lipnet.build()
        lipnet.model.load_weights(weight_path)
        logging.info("LipNet model built and loaded successfully.\n")
    except Exception as e:
        logging.error(f"Failed to build/load LipNet model: {e}")
        return None

    # Debugging: Verify the type and summary of the model
    logging.debug(f"Type of lipnet.model: {type(lipnet.model)}")
    lipnet.model.summary()

    # Print all layer names to verify uniqueness
    layer_names = [layer.name for layer in lipnet.model.layers]
    for layer in lipnet.model.layers:
        logging.debug(f"Layer Name: {layer.name}")

    # Check for duplicate layer names
    duplicates = [name for name, count in Counter(layer_names).items() if count > 1]
    if duplicates:
        logging.debug(f"Duplicate layer names found: {duplicates}")
    else:
        logging.debug("All layer names are unique.")

    # Prepare the model for prediction
    try:
        # Access only the 'the_input' layer
        the_input = lipnet.model.get_layer('the_input').input

        # Access the 'softmax' layer's output
        softmax_output = lipnet.model.get_layer('softmax').output

        # Create a new model for prediction with only 'the_input' as input and 'softmax' as output
        prediction_model = Model(inputs=the_input, outputs=softmax_output)
        logging.info("Prediction model created successfully using the 'softmax' layer.")
        logging.debug(f"Type of prediction_model: {type(prediction_model)}")
        prediction_model.summary()
    except Exception as e:
        logging.error(f"Failed to prepare prediction model: {e}")
        return None

    # Normalize video data and prepare for input
    try:
        X_data = video.data.astype(np.float32) / 255.0
        X_data = np.expand_dims(X_data, axis=0)  # Add batch dimension
        logging.info(f"Shape of X_data: {X_data.shape}")
    except Exception as e:
        logging.error(f"Failed to preprocess video data: {e}")
        return None

    # Predict the word probabilities using LipNet
    logging.info("Starting prediction...")
    try:
        y_pred_probs = prediction_model.predict(X_data)
        logging.info("Prediction completed.")
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return None

    # Decode the predictions using CTC decoding
    try:
        decoded_pred_sparse, _ = ctc_decode(
            y_pred_probs,
            input_length=np.ones(y_pred_probs.shape[0]) * y_pred_probs.shape[1]
        )
        decoded_pred = get_value(decoded_pred_sparse[0])
        logging.debug(f"Decoded Predictions: {decoded_pred}")
    except Exception as e:
        logging.error(f"CTC decoding failed: {e}")
        return None

    # Flatten and filter out blanks (27) from the predictions
    try:
        decoded_pred = decoded_pred.flatten()  # Convert to 1D array
        blank_label = 27  # Define the correct blank label based on your labels list
        decoded_pred = [seq for seq in decoded_pred if seq != blank_label]
        logging.debug(f"Filtered Decoded Predictions (without blanks): {decoded_pred}")
    except Exception as e:
        logging.error(f"Failed to filter decoded predictions: {e}")
        return None

     # Detect speech onsets with adjusted parameters
    audio_onset_times, video_onset_times = detect_speech_onsets(
        video_path, 
        movement_multiplier=2.0,    # Increased multiplier for amplification
        desired_threshold=0.015,    # Lowered threshold for sensitivity
        initial_frame_count=15,      # Reduced to allow early onset detection
        verbose=True
    )


    # Map the decoded predictions to word timestamps
    word_timestamps = map_ctc_to_timestamps_lipnet(
        decoded_pred, 
        fps, 
        audio_onset_times, 
        video_onset_times,
        min_separation=0.05,   # Adjusted as per your requirement
        tolerance=0.2          # Time tolerance for validation
    )


    # Join the decoded words into a sentence
    decoded_text = ' '.join([w['word'] for w in word_timestamps])
    logging.info(f"Decoded Text: {decoded_text}")

    # Print each word with its corresponding start time
    logging.info("\nWord Timestamps:")
    for word_info in word_timestamps:
        logging.info(f"Word: {word_info['word']}, Start Time: {word_info['start_time']:.2f}s")

    return {'decoded_text': decoded_text, 'timestamps': word_timestamps}


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        weight_path = sys.argv[1]
        video_path = sys.argv[2]
        absolute_max_string_len = int(sys.argv[3]) if len(sys.argv) >= 4 else 32
        output_size = int(sys.argv[4]) if len(sys.argv) >= 5 else 28

        response = predict(weight_path, video_path, absolute_max_string_len, output_size)
    else:
        print("Usage: python predict.py [weights path] [video path] [optional: absolute_max_string_len] [optional: output_size]")
        sys.exit()

    if response and "decoded_text" in response:
        decoded_text = response["decoded_text"]
        timestamps = response["timestamps"]

        # Generate a visual representation of the decoded text
        stripe = "-" * len(decoded_text)
        print("\n __                   __  __          __      ")
        print("/\\ \\       __        /\\ \\/\\ \\        /\\ \\__   ")
        print("\\ \\ \\     /\\_\\  _____\\ \\ \\\\ \\     __\\ \\ ,_\\  ")
        print(" \\ \\ \\  __\\/\\ \\/\\ '__\\ \\ ,  \\  /'__\\ \\ \\/  ")
        print("  \\ \\ \\L\\ \\\\ \\ \\ \\ \\L\\ \\ \\ \\\\ \\/\\  __/\\ \\ \\_ ")
        print("   \\ \\____/ \\ \\_\\ \\ ,__/\\ \\_\\ \\_\\ \\____\\\\ \\__\\")
        print("    \\/___/   \\/_/\\ \\ \\/  \\/_/\\/_/\\/____/ \\/__/")
        print("                  \\ \\_\\                       ")
        print("                   \\/_/                       ")
        print(f"             --{stripe}- ")
        print(f"[ DECODED ] |> {decoded_text} |")
        print(f"             --{stripe}- ")

        # Display word start times
        print("\nWord Timestamps:")
        for word_info in timestamps:
            print(f"Word: {word_info['word']}, Start Time: {word_info['start_time']:.2f}s")
