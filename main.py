import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import random
from tqdm import tqdm

import torch.nn.functional as F

# Configuration
BDD_DIR = "bdd100k" 
TRAIN_IMAGES = os.path.join(BDD_DIR, "images", "100k", "train")
TRAIN_LABELS = os.path.join(BDD_DIR, "labels", "drivable", "colormaps", "train")
VAL_IMAGES = os.path.join(BDD_DIR, "images", "100k", "val")
VAL_LABELS = os.path.join(BDD_DIR, "labels", "drivable", "colormaps", "val")
BATCH_SIZE = 8
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom dataset class for BDD
class BDD_Dataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        # self.label_transform = label_transform
        self.image_files = sorted(os.listdir(img_dir))
        self.label_files = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.image_files)
        # return np.minimum(len(self.image_files), 5000)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("1")
        # label = Image.open(label_path).convert("RGB")
        
        # label.show()
        # label.save(f"test_{idx}.png")
        
        # only mark the red channel as 1 (since red is the lane color)
        # label = np.asarray(label)
        
        # labelnew = np.zeros_like(label)
        # labelnew[label[..., 0] > 100] = [255, 255, 255]
        # labelnew[label[..., 0] <= 100] = [0, 0, 0]
        
        # label = (red_channel > 0).astype(np.uint8)
        
        # labelnew = Image.fromarray(labelnew)
        # labelnew = labelnew.convert("1")
        
        labelnew = label

        if self.transform:
            image = self.transform(image)
            # label = transforms.ToTensor()(label)
            labelnew = self.transform(labelnew)
        # if self.transform:
        #     image = self.transform(image)
        # if self.label_transform: 
        #     label = self.label_transform(label)
        
        # print(image.shape, label.shape)
        
        
        
        # print(label)
        # savePlot(label, "test")

        return image, labelnew
    
class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.ReLU(inplace=True))

        block.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_channels, out_channels, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out

class LaneSegmentationModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, depth=5, start_filts=64, up_mode='upconv', padding=True, batch_norm=False):
        super(LaneSegmentationModel, self).__init__()

        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth

        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, start_filts * (2 ** i), padding, batch_norm))
            prev_channels = start_filts * (2 ** i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, start_filts * (2 ** i), up_mode, padding, batch_norm))
            prev_channels = start_filts * (2 ** i)

        self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        # Remove the channel dimension
        return self.last(x).squeeze(1)
        # return self.last(x)


def display_image(image, label):
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.imshow(label.squeeze().cpu().numpy(), alpha=0.4, cmap='jet')
    # plt.imshow(pred.squeeze().cpu().numpy(), alpha=0.4, cmap='jet_r')
    plt.show()
    
def save_result(image, label, pred, save_path, idx):
    os.makedirs(save_path, exist_ok=True)
    save_image(image, os.path.join(save_path, f"image_{idx}.png"))
    save_image(label.float(), os.path.join(save_path, f"ground_truth_{idx}.png"))
    save_image(pred.float(), os.path.join(save_path, f"prediction_{idx}.png"))
    
def display_and_save(image, label, pred, save_path, idx):
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    
    ax[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    ax[0].set_title('Image')
    
    ax[1].imshow(label.squeeze().cpu().numpy(), cmap='jet')
    ax[1].set_title('Ground Truth')
    
    ax[2].imshow(pred.squeeze().cpu().numpy(), cmap='jet')
    ax[2].set_title('Prediction')
    
    ax[3].imshow(image.permute(1, 2, 0).cpu().numpy())
    ax[3].imshow(pred.squeeze().cpu().numpy(), alpha=0.4, cmap='jet_r')
    ax[3].set_title('Prediction Overlayed')
    
    # plt.show()
    plt.savefig(os.path.join(save_path, f"image_{idx}.png"))

    # save_result(image, label, pred, save_path, idx)
    
def savePlot(image, save_path):
    # # plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    # fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # ax[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    
    # plt.savefig(os.path.join(save_path, f"image.png"))
    image.save(os.path.join(save_path, f"image.png"))
    

def main():
    # Preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Preprocessing transforms
    # image_transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor()
    # ])

    # label_transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.squeeze(0))  # Remove the channel dimension
    # ])

    # Create datasets and data loaders
    train_dataset = BDD_Dataset(TRAIN_IMAGES, TRAIN_LABELS, transform=transform)
    val_dataset = BDD_Dataset(VAL_IMAGES, VAL_LABELS, transform=transform)
    # train_dataset = BDD_Dataset(TRAIN_IMAGES, TRAIN_LABELS, transform=image_transform, label_transform=label_transform)
    # val_dataset = BDD_Dataset(VAL_IMAGES, VAL_LABELS, transform=image_transform, label_transform=label_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize the model, loss, and optimizer
    model = LaneSegmentationModel().to(DEVICE)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            # indicate progress
            # print(f"\rEpoch {epoch + 1}/{EPOCHS}, Batch {batch_idx + 1}/{len(train_loader)}", end="")
            
            # idx = batch_idx + 1
            
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            # print(outputs.shape, labels.squeeze().shape)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # if batch_idx % 1000 == 0:
            #     display_image(images[0], labels[0])

        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss}")

        # Validate the model
        num_images_to_display = 10
        random_indices = random.sample(range(len(val_dataset)), num_images_to_display)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(val_loader)):
                # indicate progress
                # print(f"\rValidation Progress: {100 * (i + 1) / len(val_loader):.2f}%", end="")
                
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels.squeeze())
                val_loss += loss.item()
                
                # if i == 0:
                #     pred = torch.sigmoid(outputs)
                #     pred = (pred > 0.5).float()  # Apply a threshold to convert the probabilities to binary values
                #     display_image(images[0], labels[0], pred[0])
                
                for idx, (img, gt, pred) in enumerate(zip(images, labels, outputs)):
                    global_idx = i * BATCH_SIZE + idx
                    if global_idx in random_indices:
                        # labels.show()
                        # print(labels)
                        pred = torch.sigmoid(pred)
                        # print(pred)
                        pred = (pred > 0.5).float()  # Apply a threshold to convert the probabilities to binary values
                        # print(pred)
                        display_and_save(img, gt, pred, "results", global_idx)

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Validation Loss: {val_loss}")
        
        # display the first image in the batch
        
        # Save the model
        torch.save(model.state_dict(), f'newmodel{epoch}.pth')
    
    
if __name__ == '__main__':
    main()