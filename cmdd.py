'''
Author: Zhiyuan Yan
Email: yanzhiyuan1114@gmail.com
Time: 2023-04-14
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VideoAudioEncoder(nn.Module):
    def __init__(self, bs, clip_size, num_classes=4):
        super(VideoAudioEncoder, self).__init__()
        # Pooling layer
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Video Encoder
        resnet34 = models.resnet34(pretrained=True)
        self.video_cnn = nn.Sequential(*list(resnet34.children())[:-2]) # Remove last two layers (avgpool and fc)
        self.video_lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=3, batch_first=True)

        # Audio Encoder
        self.audio_conv1 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.audio_bn1 = nn.BatchNorm1d(256)
        self.audio_conv2 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.audio_bn2 = nn.BatchNorm1d(512)
        self.audio_lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=3, batch_first=True)

        # Fusion Layer
        self.fusion_layer = nn.Linear(128 * 2, 128)

        # Classifier
        self.fc = nn.Linear(128, num_classes)
        self.fc_binary = nn.Linear(128, 2)

        self.clip_size = clip_size

    def forward(self, video_data, audio_data):
        ## video_data: batchsize, clip_size, c, h, w
        # audio_data: batchsize, duration, 128
        bs, clip_size, c, h, w = video_data.shape

        # Pass video data through pretrained ResNet34
        video_data = self.video_cnn(video_data.view(-1, c, h, w))  # Shape: (batch_size * clip_size, 512, h//32, w//32)

        # Reshape for LSTM
        video_data = self.pool(video_data).reshape(bs, clip_size, -1)  # Shape: (batch_size, clip_size, 512)

        # Pass video data through LSTM layer
        video_data, _ = self.video_lstm(video_data)  # Shape: (batch_size, clip_size, 128)
        video_data = video_data[:, -1, :]  # Shape: (batch_size, 128)

        # Pass audio data through CNN layers
        audio_data = audio_data.transpose(1, 2)  # Shape: (batch_size, 128, duration)
        audio_data = self.audio_bn1(self.audio_conv1(audio_data))
        audio_data = self.audio_bn2(self.audio_conv2(audio_data))  # Shape: (batch_size, 32, duration)

        # Pass audio data through LSTM layer
        audio_data, _ = self.audio_lstm(audio_data.transpose(1, 2))  # Shape: (batch_size, duration, 128)
        audio_data = audio_data[:, -1, :]  # Shape: (batch_size, 128)

        # Fuse video and audio features
        fused_features = self.fusion_layer(torch.cat((video_data, audio_data), dim=1))  # Shape: (batch_size, 128)

        # Pass through the classifier
        output = self.fc(fused_features)  # Shape: (batch_size, num_classes)
        output_binary = self.fc_binary(fused_features)  # Shape: (batch_size, 2)

        return output, output_binary
