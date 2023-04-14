'''
Author: Zhiyuan Yan
Email: yanzhiyuan1114@gmail.com
Time: 2023-04-14
'''

import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from cmdd import AudioEncoder, \
    VideoAudioEncoder, CrossModalContrastiveLearningModel, MLPProjectionHead
from dataset import VideoAudioDataset
from metrics import *


label_dict = {
    'RealVideo-RealAudio': 0,
    'FakeVideo-FakeAudio': 1,
    'RealVideo-FakeAudio': 2,
    'FakeVideo-RealAudio': 3,
}


if __name__ == '__main__':
    # Set hyperparameters
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: ", device)
    
    log_dir = 'runs'
    current_time = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = 'checkpoints/{}'.format(current_time)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Initialize dataset and dataloader
    root_dir = '/mntcephfs/lab_data/zhiyuanyan/FakeAVCeleb_v1.2/'
    train_dataset = VideoAudioDataset(root_dir, data_aug=None)
    
    # Compute the weights for weighted random sampler
    label_counts = [0] * len(label_dict)
    for video_path in train_dataset.videos_list:
        label = train_dataset.get_label(video_path)
        label_counts[label] += 1
    weights = [0] * len(train_dataset.videos_list)
    for idx, video_path in enumerate(train_dataset.videos_list):
        label = train_dataset.get_label(video_path)
        weights[idx] = 1.0 / label_counts[label]

    # Initialize dataloader with weighted random sampler
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=8, collate_fn=train_dataset.collate_fn)
    print(f"train_dataset: {len(train_dataset)} samples, {len(train_dataloader)} batches")

    # Initialize model, loss function, and optimizer
    model = VideoAudioEncoder(
        batch_size, 32, 
        # d_model=512, nhead=8, num_layers=6
    ).to(device)
    loss_function = nn.CrossEntropyLoss()
    
    # Filter the parameters to only include those with gradients
    params_with_grad = filter(lambda p: p.requires_grad, model.parameters())
    # Create the optimizer with the filtered parameters
    optimizer = torch.optim.Adam(params_with_grad, lr=learning_rate)
    print("Model initialized")

    # Initialize metrics for tracking training progress
    train_loss = []
    train_acc = []
    train_auc = []
    train_eer = []

    # Train model
    best_epoch = 0
    best_auc = 0.0
    print("Training model...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Set model to train mode
        model.train()

        # Initialize metrics for this epoch
        epoch_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_classification_loss = 0.0
        epoch_classification_loss_binary = 0.0
        epoch_correct = 0.0
        epoch_correct_binary = 0.0
        epoch_total = 0.0
        epoch_scores = []
        epoch_labels = []

        # Train on mini-batches
        train_progress = tqdm(train_dataloader)
        for i, batch in enumerate(train_progress):
            # Move data to device
            video_data, audio_data, targets = batch
            video_data, audio_data, targets = video_data.to(device), audio_data.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            classification_score, classification_score_binary = model(video_data, audio_data)
            classification_loss = loss_function(classification_score, targets)
            targets_binary = torch.where(targets == 0, torch.zeros_like(targets), torch.ones_like(targets))
            classification_loss_binary = loss_function(classification_score_binary, targets_binary)
            loss = classification_loss + classification_loss_binary

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update batch metrics
            _, predicted = torch.max(classification_score.data, 1)
            _, predicted_binary = torch.max(classification_score_binary.data, 1)
            batch_acc = (predicted == targets).sum().item() / targets.size(0)
            batch_acc_binary = (predicted_binary == targets_binary).sum().item() / targets_binary.size(0)
            batch_size = targets.size(0)

            # Update epoch metrics
            epoch_loss += loss.item() * batch_size
            # epoch_contrastive_loss += contrastive_loss.item() * batch_size
            epoch_classification_loss += classification_loss.item() * batch_size
            epoch_classification_loss_binary += classification_loss_binary.item() * batch_size
            epoch_correct += (predicted == targets).sum().item()
            epoch_correct_binary += (predicted_binary == targets_binary).sum().item()
            epoch_total += batch_size
            epoch_scores.append(classification_score_binary.detach().cpu().numpy())
            epoch_labels.append(targets_binary.detach().cpu().numpy())

            # Update tqdm progress bar description and postfix
            train_progress.set_description(f"Epoch {epoch+1}/{num_epochs}")
            train_progress.set_postfix(Loss=loss.item(), Acc=batch_acc, AccBinary=batch_acc_binary)

        # Calculate epoch metrics
        epoch_loss /= epoch_total
        # epoch_contrastive_loss /= epoch_total
        epoch_classification_loss /= epoch_total
        epoch_classification_loss_binary /= epoch_total
        epoch_acc = epoch_correct / epoch_total
        epoch_acc_binary = epoch_correct_binary / epoch_total
        epoch_scores = np.concatenate(epoch_scores)
        epoch_labels = np.concatenate(epoch_labels)
        epoch_auc = roc_auc_score(epoch_labels, epoch_scores[:, 1])
        epoch_eer = compute_eer(epoch_labels, epoch_scores[:, 1])

        print(
            f"Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.4f}, "
            f"ClassificationLoss={epoch_classification_loss:.4f}, " 
            f"ClassificationLoss_binary={epoch_classification_loss_binary:.4f}, " 
            f"Acc={epoch_acc:.4f}, "
            f"Acc_binary={epoch_acc_binary:.4f}, "
            f"AUC={epoch_auc:.4f}, EER={epoch_eer:.4f} "
        )

        # Save the last checkpoint for this epoch
        checkpoint_name = f'checkpoint_{epoch+1}.pth'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'acc': epoch_acc,
            'auc': epoch_auc,
            'eer': epoch_eer
        }, checkpoint_path)

        # Save the best checkpoint based on AUC
        if epoch_auc > best_auc:
            best_epoch = epoch+1
            best_auc = epoch_auc
            print(f'Current Best AUC: {best_auc}')
            best_checkpoint_name = f'best_checkpoint.pth'
            best_checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint_name)
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'acc': epoch_acc,
                'auc': epoch_auc,
                'eer': epoch_eer
            }, best_checkpoint_path)

    print(f"After training, Best epoch: {best_epoch}, AUC: {best_auc:.4f}")
    print("Finish training.")