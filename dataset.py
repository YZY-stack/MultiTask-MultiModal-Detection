'''
Author: Zhiyuan Yan
Email: yanzhiyuan1114@gmail.com
Time: 2023-04-14
'''

import os
import glob
import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch.nn.functional as F
from torchvision import transforms as T
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import numpy as np
from load_data import load_video_data

label_dict = {
    'RealVideo-RealAudio': 0,
    'FakeVideo-FakeAudio': 1,
    'RealVideo-FakeAudio': 2,
    'FakeVideo-RealAudio': 3,
}


class VideoAudioDataset(Dataset):
    def __init__(self, root_dir, data_aug=None, clip_size=16, resize_shape=(224, 224)):
        """ Video and audio dataset.
        
        Args:
        root_dir (str): Root directory of the dataset.
        transform (callable): Optional transform to be applied on a video clip.
        clip_size (int): Number of frames per clip.
        resize_shape (tuple): Target size for video frames.
        """
        self.root_dir = root_dir
        self.data_aug = data_aug
        self.clip_size = clip_size
        self.resize_shape = resize_shape
        self.classes = ['FakeVideo-FakeAudio', 'RealVideo-FakeAudio', 'FakeVideo-RealAudio', 'RealVideo-RealAudio']
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.videos_list = glob.glob(os.path.join(root_dir, '**/*.mp4'), recursive=True)  # using glob to recursively obtain all videos in given folder
        random.shuffle(self.videos_list)

    def get_label(self, video_path):
        """ Get the label of a video clip.

        Args:
        video_path (str): Path to the video clip.

        Returns:
        int: Label of the video clip.
        """
        target_info = video_path.split('/')[5]
        if target_info not in label_dict:
            raise ValueError(
            f"Expect label to be one of the FakeVideo-FakeAudio, RealVideo-FakeAudio, FakeVideo-RealAudio, RealVideo-RealAudio, but got {target_info}")
        target = label_dict[target_info]
        return target
    
    def transform_func(self, frame_clip):
        """ Apply data augmentation and normalization to a video clip.

        Args:
        frame_clip (np.ndarray): Array of video frames with shape (num_frames, c, h, w).

        Returns:
        tuple: A tuple of transformed video frames.
        """
        # Apply frame clip transformation
        if self.data_aug:
            raise ValueError("Data augmentation is not supported yet.")
        
        frame_normalized = []
        for i in range(frame_clip.shape[0]):
            one_frame = Image.fromarray(frame_clip[i])
            one_frame = T.ToTensor()(one_frame)
            one_frame = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(one_frame)
            frame_normalized.append(one_frame)
        frame_normalized = torch.stack(frame_normalized, dim=0)
        return frame_normalized


    def __len__(self):
        """Get the number of video clips in the dataset.

        Returns:
        int: Number of video clips in the dataset.
        """
        return len(self.videos_list)

    def __getitem__(self, idx):
        """Return the video frames and audio clip and its corresponding label for a given index.

        Args:
            idx (int): Index of the item to be returned.

        Returns:
            Tuple of video frames, audio clip, and target label.
        """
        video_path = self.videos_list[idx]
        video_nplist, audio_data_np = load_video_data(
            video_path, 
            num_sampled_frames=32, 
            resize_shape=self.resize_shape
        )
        target = self.get_label(video_path)
        
        # Convert audio_data from numpy to PyTorch tensor and reorder dimensions to match the shape desired by the audioEncoder
        audio_one_torch_vid = torch.from_numpy(audio_data_np).permute(0, 2, 1)  # Shape: (1, n_frames, n_mels)

        # Apply transform to the frame_clip
        frame_one_torch_vid = self.transform_func(video_nplist)

        return frame_one_torch_vid, audio_one_torch_vid, target
    

    @staticmethod
    def collate_fn(batch):
        """Collate function for VideoAudioDataset that collates a batch of samples into a batch of tensors.

        Args:
            batch (list): A list of tuples, each containing a frame_clip, an audio_spectrogram, and a target label.

        Returns:
            Tuple of collated frame_clips, audio_spectrograms, and target labels.
        """
        # Collect the frame_clips, audio_spectrograms, and target labels into separate lists
        frame_clips, audio_spectrograms, targets = [], [], []
        for sample in batch:
            frame_clip, audio_spectrogram, target = sample
            frame_clips.append(frame_clip.squeeze())
            audio_spectrograms.append(audio_spectrogram.squeeze().float())  # Permute the dimension for padding
            targets.append(target)

        # Stack the frame_clips into batch tensors
        frame_clips = torch.stack(frame_clips, dim=0)  # Shape: (batch_size, num_frames, c, h, w)
        
        # Stack the audio_spectrograms into batch tensors
        audio_spectrograms = torch.stack(audio_spectrograms, dim=0)  # Shape: (batch_size, 1, n_frames, n_mels)
        
        # Stack the targets into batch tensors
        targets = torch.tensor(targets)  # Shape: (batch_size)

        return frame_clips, audio_spectrograms, targets


if __name__ == '__main__':
    root_dir = '/mntcephfs/lab_data/zhiyuanyan/FakeAVCeleb_v1.2/'
    dataset = VideoAudioDataset(root_dir, data_aug=None)

    # Example usage in a DataLoader
    from torch.utils.data import DataLoader
    from torch.utils.data import WeightedRandomSampler

    # Compute the weights
    label_counts = [0] * len(label_dict)
    for video_path in dataset.videos_list:
        label = dataset.get_label(video_path)
        label_counts[label] += 1

    weights = [0] * len(dataset.videos_list)
    for idx, video_path in enumerate(dataset.videos_list):
        label = dataset.get_label(video_path)
        weights[idx] = 1.0 / label_counts[label]

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8, collate_fn=dataset.collate_fn)
    for i, batch in enumerate(dataloader):
        video_data, audio_data, targets = batch
        print(targets.cpu().numpy())
        # Do something with the data
