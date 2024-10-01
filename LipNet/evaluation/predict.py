from lipnet.lipreading.videos import Video
from lipnet.lipreading.visualization import show_video_subtitle
from lipnet.core.decoders import Decoder
from lipnet.lipreading.helpers import labels_to_text
from lipnet.utils.spell import Spell
from lipnet.model2 import LipNet
from utils.resize_video import resize_and_adjust_video
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import sys
import os
import tempfile


np.random.seed(55)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
FACE_PREDICTOR_PATH = os.path.join(CURRENT_PATH, '..', 'common', 'predictors', 'shape_predictor_68_face_landmarks.dat')
PREDICT_GREEDY = False
PREDICT_BEAM_WIDTH = 200
PREDICT_DICTIONARY = os.path.join(CURRENT_PATH, '..', 'common', 'dictionaries', 'grid.txt')

def predict(weight_path, video_path, absolute_max_string_len=32, output_size=28):
    print("\nResizing the video to match the model's expected input dimensions...")

    target_width = 100
    target_height = 50
    target_frames = 75

    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mpg')
    temp_video_path = temp_video_file.name
    temp_video_file.close()

    # Call the resize_and_adjust_video function
    resize_and_adjust_video(video_path, temp_video_path, target_width, target_height, target_frames)

    print(f"\nLoading resized video data from: {temp_video_path}")
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)

    if os.path.isfile(temp_video_path):
        video.from_video(temp_video_path)
    else:
        print(f"Video file not found: {temp_video_path}")
        return None, ""

    print("Data loaded successfully.\n")

    if K.image_data_format() == 'channels_first':
        img_c, frames_n, img_w, img_h = video.data.shape
    else:
        frames_n, img_w, img_h, img_c = video.data.shape

    lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                    absolute_max_string_len=absolute_max_string_len, output_size=output_size)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)




    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    lipnet.model.load_weights(weight_path)

    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])

    X_data = np.array([video.data]).astype(np.float32) / 255
    input_length = np.array([len(video.data)])

    print("Starting prediction...")
    y_pred = lipnet.predict(X_data)
    result = decoder.decode(y_pred, input_length)[0]

    os.unlink(temp_video_path)
    print(f"Temporary file {temp_video_path} removed.")

    return (video, result)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        video, result = predict(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        video, result = predict(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    elif len(sys.argv) == 5:
        video, result = predict(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))

    if video is not None:
        show_video_subtitle(video.face, result)

    stripe = "-" * len(result)
    print("")
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
    print(f"[ DECODED ] |> {result} |")
    print(f"             --{stripe}- ")
