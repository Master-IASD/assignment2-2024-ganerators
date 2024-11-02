import torch
import torchvision
import os
import argparse
import numpy as np

from model import Generator
from utils import load_model

def generate_interpolated_samples(G, num_samples=10000, interpolation_steps=10):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    args = parser.parse_args()

    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    model = Generator(g_output_dim=mnist_dim).cuda()
    model = load_model(model, 'checkpoints-test')
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
