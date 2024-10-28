import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim


from model import Generator, Discriminator
from utils import D_train, G_train, save_models
from utils import load_model, load_discriminator_model


from datasets import NoiseDataset


if __name__ == '__main__':
    modes = ["train", "collab"]
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")
    parser.add_argument('--mode', type=str, default="train", choices=modes,
                        help=f'Options: {modes}')

    args = parser.parse_args()


    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')


    print('Model Loading...')
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).cuda()
    D = torch.nn.DataParallel(Discriminator(mnist_dim)).cuda()


    # model = DataParallel(model).cuda()
    print('Model loaded.')
    # Optimizer 



    # define loss
    criterion = nn.BCELoss() 

    if args.mode == 'train':
        print("Running in mode TRAIN")

        # define optimizers
        G_optimizer = optim.Adam(G.parameters(), lr = args.lr)
        D_optimizer = optim.Adam(D.parameters(), lr = args.lr)

        print('Start Training :')
        
        n_epoch = args.epochs
        for epoch in trange(1, n_epoch+1, leave=True):           
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.view(-1, mnist_dim)
                D_train(x, G, D, D_optimizer, criterion)
                G_train(x, G, D, G_optimizer, criterion)

            if epoch % 10 == 0:
                save_models(G, D, 'checkpoints')
                    
        print('Training done')
    
    elif args.mode == "collab":

        print("Running in mode COLLAB")

        # load generator
        G = Generator(g_output_dim = mnist_dim).cuda()
        G = load_model(G, 'checkpoints')
        G = torch.nn.DataParallel(G).cuda()

        # load discriminator
        D = Discriminator(mnist_dim).cuda()
        D = load_discriminator_model(D, 'checkpoints')
        D = torch.nn.DataParallel(D).cuda()

        # create random dataset
        noise = NoiseDataset(dim=100)
        rollout_rate = 0.1 # This is the learning rate used by the optimizer (optim_r) for refining the perturbations (delta_refine).
        rollout_steps = 50 # number of iterations over which the refinement (adjustment of the generated samples) takes place.

        optim_d = optim.SGD(D.parameters(), lr=args.lr)

        n_epoch = args.epochs
        for epoch in trange(1, n_epoch+1, leave=True):   
            print(f"EPOCH {epoch}")        
            for batch_idx, (x, _) in enumerate(train_loader):
                # print batch id
                if batch_idx % 100 == 0:
                    print(f"BATCH {batch_idx}")

                # resize x from (batch size, 28, 28) to (batch size, 784)
                x = x.view(-1, mnist_dim)

                # synthesize noisy samples
                noise_batch = noise.next_batch(args.batch_size).cuda()
                fake_batch = G(noise_batch)

                # probabilistic refinement
                proba_refine = torch.zeros([args.batch_size, mnist_dim], requires_grad=False, device="cuda")
                proba_steps = torch.LongTensor(args.batch_size,1).random_() % rollout_steps
                # Create a one-hot encoded matrix indicating in which iteration each batch item will be assigned some perturbation.
                proba_steps_one_hot = torch.LongTensor(args.batch_size, rollout_steps)
                proba_steps_one_hot.zero_()
                proba_steps_one_hot.scatter_(1, proba_steps, 1)

                # create tensor of small perturbations
                delta_refine = torch.zeros([args.batch_size, mnist_dim], requires_grad=True, device="cuda")
                optim_r = optim.Adam([delta_refine], lr=rollout_rate)

                # Define a target label tensor filled with ones, indicating the desired outcome for the discriminator.
                label = torch.full((args.batch_size,1), 1, dtype=torch.float, device="cuda")

                # Refinement loop to iteratively adjust `delta_refine` over a set number of steps.
                for k in range(rollout_steps):
                    optim_r.zero_grad()
                    output = D(fake_batch.detach() + delta_refine) # add perturbations to fake samples
                    loss_r = criterion(output, label)
                    loss_r.backward() # improve the discriminator
                    optim_r.step()

                    # probabilistic assignment: apply the refined perturbation only at the designated step
                    proba_refine[proba_steps_one_hot[:,k] == 1, :] = delta_refine[proba_steps_one_hot[:,k] == 1, :]

                ############################
                # Shape D network: maximize log(D(x)) + log(1 - D(R(G(z))))
                ###########################
                optim_d.zero_grad()

                # train with real
                real_batch = x
                output = D(real_batch)
                loss_d_real = criterion(output, label)
                loss_d_real.backward()

                # train with refined
                label.fill_(0)
                output = D((fake_batch+proba_refine).detach())
                loss_d_fake = criterion(output, label)
                loss_d_fake.backward()

                loss_d = loss_d_real + loss_d_fake
                optim_d.step()

            if epoch % 10 == 0:
                save_models(G, D, 'checkpoints')

        print('Refinement done')
