import os
import numpy as np
from keras import backend as K
from scipy import ndimage
from scipy.misc import imresize
import skvideo.io
import dlib
from lipnet.lipreading.aligns import Align

class VideoAugmenter(object):
    @staticmethod
    def split_words(video, align):
        video_aligns = []
        for sub in align.align:
            _video = Video(video.vtype, video.face_predictor_path)
            _video.face = video.face[sub[0]:sub[1]]
            _video.mouth = video.mouth[sub[0]:sub[1]]
            _video.set_data(_video.mouth)
            _align = Align(align.absolute_max_string_len, align.label_func).from_array([(0, sub[1] - sub[0], sub[2])])
            video_aligns.append((_video, _align))
        return video_aligns

    @staticmethod
    def merge(video_aligns):
        vsample = video_aligns[0][0]
        asample = video_aligns[0][1]
        video = Video(vsample.vtype, vsample.face_predictor_path)
        video.face = np.ones((0, vsample.face.shape[1], vsample.face.shape[2], vsample.face.shape[3]), dtype=np.uint8)
        video.mouth = np.ones((0, vsample.mouth.shape[1], vsample.mouth.shape[2], vsample.mouth.shape[3]), dtype=np.uint8)
        align = []
        inc = 0
        for _video, _align in video_aligns:
            video.face = np.concatenate((video.face, _video.face), 0)
            video.mouth = np.concatenate((video.mouth, _video.mouth), 0)
            for sub in _align.align:
                _sub = (sub[0] + inc, sub[1] + inc, sub[2])
                align.append(_sub)
            inc = align[-1][1]
        video.set_data(video.mouth)
        align = Align(asample.absolute_max_string_len, asample.label_func).from_array(align)
        return (video, align)

    @staticmethod
    def horizontal_flip(video):
        _video = Video(video.vtype, video.face_predictor_path)
        _video.face = np.flip(video.face, 2)
        _video.mouth = np.flip(video.mouth, 2)
        _video.set_data(_video.mouth)
        return _video


class Video(object):
    def __init__(self, vtype='mouth', face_predictor_path=None):
        if vtype == 'face' and face_predictor_path is None:
            raise AttributeError('Face video requires a face predictor path.')
        self.face_predictor_path = face_predictor_path
        self.vtype = vtype

    def from_frames(self, path):
        frames_path = sorted([os.path.join(path, x) for x in os.listdir(path)])
        frames = [ndimage.imread(frame_path) for frame_path in frames_path]
        self.handle_type(frames)
        return self

    def from_video(self, path):
        frames = self.get_video_frames(path)
        print(f"Total frames extracted from video: {len(frames)}")  # Debugging statement
        self.handle_type(frames)
        return self

    def from_array(self, frames):
        self.handle_type(frames)
        return self

    def handle_type(self, frames):
        if self.vtype == 'mouth':
            self.process_frames_mouth(frames)
        elif self.vtype == 'face':
            self.process_frames_face(frames)
        else:
            raise Exception('Invalid video type.')

    def process_frames_face(self, frames):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.face_predictor_path)
        mouth_frames = self.get_frames_mouth(detector, predictor, frames)
        self.face = np.array(frames)
        self.mouth = np.array(mouth_frames)
        self.set_data(mouth_frames)

    def process_frames_mouth(self, frames):
        self.face = np.array(frames)
        self.mouth = np.array(frames)
        self.set_data(frames)

    def get_frames_mouth(self, detector, predictor, frames):
        MOUTH_WIDTH = 100
        MOUTH_HEIGHT = 50
        HORIZONTAL_PAD = 0.19
        normalize_ratio = None
        mouth_frames = []
        frame_count = 0  # Track frame index

        for idx, frame in enumerate(frames):
            dets = detector(frame, 1)
            shape = None
            for k, d in enumerate(dets):
                shape = predictor(frame, d)
                break  # Only process the first detected face

            if shape is None:
                print(f"Warning: Face not detected in frame {idx}")
                # Instead of returning immediately, append a black frame or the original frame
                # For simplicity, we'll append the original frame resized to the mouth dimensions
                mouth_frames.append(imresize(frame, (MOUTH_HEIGHT, MOUTH_WIDTH)))
                continue

            mouth_points = [(part.x, part.y) for i, part in enumerate(shape.parts()) if i >= 48]
            np_mouth_points = np.array(mouth_points)
            mouth_centroid = np.mean(np_mouth_points, axis=0)

            if normalize_ratio is None:
                mouth_left = np.min(np_mouth_points[:, 0]) * (1.0 - HORIZONTAL_PAD)
                mouth_right = np.max(np_mouth_points[:, 0]) * (1.0 + HORIZONTAL_PAD)
                normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

            resized_img = imresize(frame, (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio)))
            mouth_centroid_norm = mouth_centroid * normalize_ratio

            mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
            mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
            mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
            mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

            mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]
            mouth_frames.append(mouth_crop_image)
            frame_count += 1

        print(f"Total frames processed for mouth detection: {len(mouth_frames)}")  # Debugging statement
        return mouth_frames

    def get_video_frames(self, path):
        videogen = skvideo.io.vreader(path)
        frames = [frame for frame in videogen]
        print(f"Total frames extracted: {len(frames)}")  # Debugging statement
        return frames

    def set_data(self, frames):
        data_frames = []
        for idx, frame in enumerate(frames):
            frame = frame.swapaxes(0, 1)  # Swap width and height to form format W x H x C
            if len(frame.shape) < 3:
                frame = np.expand_dims(frame, axis=-1)
            data_frames.append(frame)
            # print(f"Processed frame {idx}: shape {frame.shape}")  # Optional debugging

        frames_n = len(data_frames)
        data_frames = np.array(data_frames)  # T x W x H x C
        if K.image_data_format() == 'channels_first':
            data_frames = np.rollaxis(data_frames, 3)  # C x T x W x H

        self.data = data_frames
        self.length = frames_n
        print(f"Total frames processed into data: {self.length}")  # Debugging statement