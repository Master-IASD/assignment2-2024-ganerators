import torch
import torchvision
import os
import argparse
import numpy as np

from model import Generator
from utils import load_model

def generate_interpolated_samples_old(G, num_samples=10000, interpolation_steps=10):
    generated_images = []
    
    for _ in range(num_samples // interpolation_steps):
        z1 = torch.randn(1, 100).cuda()
        z2 = torch.randn(1, 100).cuda()
        
        for alpha in np.linspace(0, 1, interpolation_steps):
            z = alpha * z1 + (1 - alpha) * z2
            gen_image = G(z).detach().cpu()
            gen_image = gen_image.view(28, 28)  
            generated_images.append(gen_image)
            
    return torch.stack(generated_images)

def generate_interpolated_samples(G, latent_dim, num_samples=10000, interpolation_steps=10):
    generated_images = []
    
    for _ in range(num_samples // interpolation_steps):
        # Generate 10 random vectors in the latent space
        z_vectors = [torch.randn(1, latent_dim).cuda() for _ in range(10)]
        
        for _ in range(interpolation_steps):
            # Generate random weights that sum to 1 for the convex combination
            weights = np.random.dirichlet(np.ones(10))  # Dirichlet distribution ensures the sum is 1
            weights_tensor = torch.tensor(weights, dtype=torch.float32).cuda().view(-1, 1)
            
            # Compute the convex combination
            z = sum(w * z for w, z in zip(weights_tensor, z_vectors))
            
            # Generate and reshape the image
            gen_image = G(z).detach().cpu()
            gen_image = gen_image.view(28, 28)  # Reshape to image dimensions
            generated_images.append(gen_image)
            
    return torch.stack(generated_images)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    args = parser.parse_args()

    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    model = Generator(g_output_dim=mnist_dim).cuda()
    model = load_model(model, 'checkpoints')
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    print('Model loaded.')

    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    # Generate images using interpolation technique
    generated_samples = generate_interpolated_samples(model, num_samples=10000, interpolation_steps=10)

    # Save generated images
    for n_samples in range(len(generated_samples)):
        torchvision.utils.save_image(generated_samples[n_samples], os.path.join('samples', f'{n_samples}.png'))         

    print(f'Generated and saved {len(generated_samples)} images to the "samples" directory.')
