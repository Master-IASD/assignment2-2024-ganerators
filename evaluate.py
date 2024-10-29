from sklearn.metrics import precision_score, recall_score
import torch
from model import Discriminator, Generator
from utils import load_discriminator_model, load_model
from torchmetrics.image.fid import FrechetInceptionDistance
import argparse
import numpy as np
from scipy.linalg import sqrtm
from torchvision import models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

# def load_fake_images(directory, transform):
#     fake_images = []
    
#     for filename in os.listdir(directory):
#         if filename.endswith(".png") or filename.endswith(".jpg"):
#             path = os.path.join(directory, filename)
#             image = Image.open(path)#.convert("L")
#             # image = transform(image)
#             fake_images.append(image)
    
#     fake_images_tensor = torch.stack(fake_images)
#     return fake_images_tensor

class FakeImageDataset(Dataset):
    def __init__(self, directory, transform):
        self.directory = directory
        self.transform = transform
        self.image_files = [f for f in os.listdir(directory) if f.endswith(".png") or f.endswith(".jpg")]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.image_files[idx])
        image = Image.open(img_path).convert("L")  # Convert to grayscale for MNIST
        return self.transform(image)

def calculate_precision_recall(discriminator, real_data, fake_data):
    # Generate samples
    # noise = torch.randn(num_samples, noise_dim).to(device)
    # fake_data = generator(noise)

    # Combine real and fake samples
    # print(fake_data.shape, real_data.shape)
    all_data = torch.cat((real_data, fake_data), dim=0)
    all_data = all_data.view(-1, 784)
    # print(all_data.shape)
    labels = torch.cat((torch.ones(real_data.size(0)), torch.zeros(fake_data.size(0))), dim=0)
    # labels = labels.view(-1, 784)
    # print(labels.shape)

    # Evaluate discriminator
    with torch.no_grad():
        preds = discriminator(all_data)
        preds_labels = (preds > 0).float()  # Use a threshold to classify

    # Calculate precision and recall
    precision = precision_score(labels.cpu(), preds_labels.cpu())
    recall = recall_score(labels.cpu(), preds_labels.cpu())
    
    return precision, recall

def get_inception_features(images):
    # Load Inception v3 model
    inception_model = models.inception_v3(pretrained=True, transform_input=True)
    inception_model.eval()

    with torch.no_grad():
        features = inception_model(images)
        
    return features

def calculate_fid(real_images, fake_images):
    # Get features from real and fake images
    real_features = get_inception_features(real_images)
    fake_features = get_inception_features(fake_images)

    # Calculate mean and covariance for real and fake features
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # Calculate FID score
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real.dot(sigma_fake))

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid

def fid_calc_fun(test_loader, G):
    
    fid = FrechetInceptionDistance(feature=64).cuda()
    fid.reset()
    G.eval()

    for real_data, _ in test_loader:
        real_data = real_data.view(-1, mnist_dim).cuda()
        noise = torch.randn(real_data.size(0), 100).cuda()
        fake_data = G(noise)
                
        real_data_rgb = real_data.view(-1, 1, 28, 28).repeat(1, 3, 1, 1)
        fake_data_rgb = fake_data.view(-1, 1, 28, 28).repeat(1, 3, 1, 1)

        real_data_rgb_uint8 = ((real_data_rgb + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        fake_data_uint8 = ((fake_data_rgb + 1) * 127.5).clamp(0, 255).to(torch.uint8)

        fid.update(real_data_rgb_uint8, real=True)
        fid.update(fake_data_uint8, real=False)

    return fid.compute().item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    parser.add_argument("--dir", help="Directory of fake images.")
    parser.add_argument("--checkpoint", help="Checkpoint path", default="checkpoints")
    args = parser.parse_args()

    mnist_dim = 784

    model = Discriminator(mnist_dim).cuda()
    model = load_discriminator_model(model, args.checkpoint)
    model = torch.nn.DataParallel(model).cuda()

    print("Model loaded")

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print("Real dataset loaded")

    # Set up a DataLoader for the fake images in batches
    fake_dataset = FakeImageDataset(args.dir, transform=transform)
    fake_loader = DataLoader(fake_dataset, batch_size=args.batch_size, shuffle=False)
    print("Fake dataset loaded")

    precision_scores = []
    recall_scores = []
    fid_scores = []
    print("Calculating scores")

    for fake_batch in fake_loader:
        real_batch = next(iter(test_loader))[0]
        
        # Calculate metrics for the current batch
        precision, recall = calculate_precision_recall(model, real_batch, fake_batch)
        # fid = calculate_fid(real_batch, fake_batch)
        
        # Append results for averaging later
        precision_scores.append(precision)
        recall_scores.append(recall)
        # fid_scores.append(fid)
    
    print("Averaging scores")
    # Average the metrics over all batches
    average_precision = np.mean(precision_scores)
    average_recall = np.mean(recall_scores)
    average_fid = np.mean(fid_scores)

    G = load_model(Generator(g_output_dim=mnist_dim).cuda(), args.checkpoint)
    fid = fid_calc_fun(test_loader, G)

    print(f'Precision: {average_precision}, Recall: {average_recall}, FID: {fid}')
