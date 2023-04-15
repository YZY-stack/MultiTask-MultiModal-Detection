import os
import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from load_data import load_video_data
from cmdd import VideoAudioEncoder
from dataset import VideoAudioDataset


type_label_dict = {
    0: 'RealVideo-RealAudio',
    1: 'FakeVideo-FakeAudio',
    2: 'RealVideo-FakeAudio',
    3: 'FakeVideo-RealAudio',
}

label_dict = {
    0: 'Real',
    1: 'Fake',
}


# Parameters for inference
batch_size = 1


def transform_func(frame_clip):
        """ Apply data augmentation and normalization to a video clip.

        Args:
        frame_clip (np.ndarray): Array of video frames with shape (num_frames, c, h, w).

        Returns:
        tuple: A tuple of transformed video frames.
        """
        frame_normalized = []
        for i in range(frame_clip.shape[0]):
            one_frame = Image.fromarray(frame_clip[i])
            one_frame = T.ToTensor()(one_frame)
            one_frame = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(one_frame)
            frame_normalized.append(one_frame)
        frame_normalized = torch.stack(frame_normalized, dim=0)
        return frame_normalized


# Load the pre-trained model
def load_pretrained_model(model_path, model_class, device):
    model = model_class(
        batch_size, 32,
    ).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


# Inference function
def infer_single_video(model, video_path):
    with torch.no_grad():
        video_nplist, audio_data_np = load_video_data(
            video_path,
            num_sampled_frames=32,
            resize_shape=(224, 224),
        )
        video_frames = transform_func(video_nplist)
        audio_data = torch.from_numpy(audio_data_np).permute(0, 2, 1).float().to(device)

        video_frames = video_frames.unsqueeze(0).to(device)
        classification_score, classification_score_binary = model(video_frames, audio_data)
        predictions = torch.argmax(classification_score, dim=1)
        predictions_binary = torch.argmax(classification_score_binary, dim=1)
        fakeness = torch.softmax(classification_score_binary, dim=1)[0][1].item()
        type_prob = torch.softmax(classification_score_binary, dim=1)[0][1].item()
        return int(predictions.cpu().numpy()), int(predictions_binary.cpu().numpy()), type_prob, fakeness


# Load the pre-trained model
model_path = '/home/tianshuoge/aicomp/checkpoints/best_checkpoint.pth'
print(f'Loading the pre-trained model from {model_path}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: {}'.format(device))
pretrained_model = load_pretrained_model(model_path, VideoAudioEncoder, device)
print('Pre-trained model loaded successfully!')

# Collect video files
video_folder = '/mntcephfs/lab_data/zhiyuanyan/FakeAVCeleb_v1.2/FakeVideo-RealAudio/African/men/id00076'
video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')]
print('Inference on {} video files'.format(len(video_files)))

# Inference
count = 0
t1 = time.time()
for video_file in video_files:
    count += 1
    prediction, prediction_binary, type_prob, fakeness = infer_single_video(pretrained_model, video_file)
    type_name = type_label_dict[prediction]
    class_name = label_dict[prediction_binary]
    print(
        f'{count}/{len(video_files)}: '
        f'Video {video_file} is predicted as {class_name} with fakeness {fakeness:.4f}, '
        f'which belongs to {type_name} with confidence score: {type_prob:.4f}'
    )
t2 = time.time()
print('Inference time: {:.4f} seconds and {:.4f} seconds per video'.format(t2-t1, (t2-t1)/len(video_files)))

print('Done!')