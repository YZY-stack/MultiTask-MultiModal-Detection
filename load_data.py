'''
Author: Zhiyuan Yan
Email: yanzhiyuan1114@gmail.com
Time: 2023-04-14
'''

import cv2
import os
import dlib
import numpy as np
from skimage import transform as trans
import matplotlib.pyplot as plt
import torch
import librosa
import moviepy.editor as mp


# Define face detector and predictor models
face_detector = dlib.get_frontal_face_detector()
predictor_path = '/home/yuanxinhang/SelfBlendedImages/src/preprocess/shape_predictor_81_face_landmarks.dat'
face_predictor = dlib.shape_predictor(predictor_path)


def visualize_spectrogram(spectrogram):
    spectrogram_np = spectrogram.squeeze(0).numpy()
    # Create a figure with a single subplot
    fig, ax = plt.subplots(1, 1)

    # Plot the waveform
    ax.imshow(spectrogram_np, origin='lower', aspect='auto', cmap='viridis')

    # Add axis labels and a title
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    ax.set_title('Waveform')

    fig.savefig('waveform—mel.png')


def get_keypts(image, face, predictor, face_detector):
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)
    
    # select the key points for the eyes, nose, and mouth
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts


def extract_aligned_face_dlib(face_detector, predictor, image, res=224, scale=1.1, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, mask=None):
        """ 
        align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
        """

        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        # M: use opencv
        # M = cv2.getAffineTransform(src[[0,1,2],:],dst[[0,1,2],:])

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img

    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with dlib
    faces = face_detector(rgb, 1)
    if len(faces):
        # For now only take the biggest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(rgb, face, predictor, face_detector)

        # Align and crop the face
        cropped_face = img_align_crop(rgb, landmarks, outsize=(res, res))
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        
        return cropped_face
    
    else:
        return rgb


def preprocess_audio(waveform: np.ndarray, fixed_duration: int) -> np.ndarray:
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    sr = 22050
    
    # Compute the Mel-scaled spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform, sr=sr, 
        n_fft=n_fft, hop_length=hop_length, 
        n_mels=n_mels,
    )
    
    # Convert to log scale
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Pad or truncate the spectrogram to a fixed duration
    current_length = log_mel_spectrogram.shape[1]
    if current_length < fixed_duration:
        padding = fixed_duration - current_length
        log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, padding)), mode='constant')
    else:
        log_mel_spectrogram = log_mel_spectrogram[:, :fixed_duration]

    return log_mel_spectrogram


def load_video_data(
    video_path: str,
    num_sampled_frames: int = 32,
    fixed_duration: int = 300,
    resize_shape: tuple = (224, 224)
) -> tuple:

    # Check if the video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' not found.")

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        raise IOError(f"Unable to open video file '{video_path}'.")

    # Extract audio from the video file using moviepy
    try:
        video = mp.VideoFileClip(video_path)
    except Exception as e:
        raise IOError(f"Unable to read audio from the video file '{video_path}'. Error: {e}")

    audio = video.audio

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)

    # Calculate the frame sampling interval
    frame_sampling_interval = max(1, num_frames // num_sampled_frames)

    # Calculate the ratio of sampled video
    ratio = frame_sampling_interval * num_sampled_frames / num_frames

    # Initialize the output tensors for video and audio
    video_data = np.zeros((num_sampled_frames, *resize_shape, 3), dtype=np.uint8)  # num_sampled_frames, h, w, c

    # Loop over the video frames and extract sampled frames
    frame_idx = 0
    while frame_idx < num_sampled_frames:
        # Calculate the current frame position in the video
        frame_pos = int(frame_idx * frame_sampling_interval)

        # Read frame at the current position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            # End of video file
            break

        # Resize the frame
        frame = cv2.resize(frame, resize_shape)

        # Convert to RGB format and add to the video data tensor
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_data[frame_idx] = frame

        # Increment frame_idx for the next iteration
        frame_idx += 1

    # Extract corresponding audio segment
    audio_duration = video.duration * ratio
    start_time = 0
    end_time = min(audio_duration, video.duration)  # Clamp end_time to the video duration
    try:
        waveform = audio.subclip(start_time, end_time).to_soundarray(fps=22050)
    except:
        waveform = audio.subclip(start_time, video.duration).to_soundarray(fps=22050)
    
    # Preprocess the audio waveform to obtain a log-scaled Mel spectrogram
    log_mel_spectrogram = preprocess_audio(waveform[:, 0], fixed_duration=fixed_duration)  # Assuming mono audio

    # Add one dimension to the audio data
    audio_data = np.expand_dims(log_mel_spectrogram, axis=0)

    # Close video and audio objects
    cap.release()
    video.close()

    # shape of video_data: (num_sampled_frames, h, w, c), shape of audio_data: (1, n_mels, fixed_duration)
    return video_data, audio_data


if __name__ == '__main__':
    load_video_data(video_path='/mntcephfs/lab_data/zhiyuanyan/FakeAVCeleb_v1.2/FakeVideo-FakeAudio/African/men/id00076/00109_2_id01236_wavtolip.mp4')