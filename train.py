import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim


from model import Generator, Discriminator
from utils import D_train, G_train, save_models, load_model, load_discriminator_model


from datasets import NoiseDataset


if __name__ == '__main__':
    modes = ["train", "refine", "collab"]
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
        # G_scheduler = optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[args.epochs // 2, 3 * args.epochs // 4, 7 * args.epochs // 8], gamma=0.1)
        D_optimizer = optim.Adam(D.parameters(), lr = args.lr)
        # D_scheduler = optim.lr_scheduler.MultiStepLR(D_optimizer, milestones=[args.epochs // 2, 3 * args.epochs // 4, 7 * args.epochs // 8], gamma=0.1)

        print('Start Training :')
        
        n_epoch = args.epochs
        for epoch in trange(1, n_epoch+1, leave=True):           
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.view(-1, mnist_dim)
                D_train(x, G, D, D_optimizer, criterion)
                G_train(x, G, D, G_optimizer, criterion)
            # G_scheduler.step()
            # D_scheduler.step()

            if epoch % 10 == 0:
                save_models(G, D, 'checkpoints')
                    
        print('Training done')

    # elif args.mode == 'refine':

    #     print("Running in mode REFINE")

    #     G = Generator(g_output_dim = mnist_dim).cuda()
    #     G = load_model(G, 'checkpoints')
    #     G = torch.nn.DataParallel(G).cuda()

    #     D = Discriminator(mnist_dim).cuda()
    #     D = load_discriminator_model(D, 'checkpoints')
    #     D = torch.nn.DataParallel(D).cuda()

    #     noise = NoiseDataset(dim=100)
    #     rollout_rate = 0.1
    #     rollout_steps = 30

    #     noise_batch = noise.next_batch(args.batch_size).cuda()
    #     print(noise_batch.shape)
    #     fake_batch = G(noise_batch)

    #     delta_refine = torch.zeros([args.batch_size, mnist_dim], requires_grad=True, device="cuda")
    #     optim_r = optim.Adam([delta_refine], lr=rollout_rate)
    #     label = torch.full((args.batch_size,1), 1, dtype=torch.float, device="cuda")
    #     for k in range(rollout_steps):
    #         optim_r.zero_grad()
    #         output = D(fake_batch.detach() + delta_refine)
    #         loss_r = criterion(output, label)
    #         loss_r.backward()
    #         optim_r.step()

    #     os.makedirs('checkpoints-refined', exist_ok=True)
    #     save_models(G, D, 'checkpoints-refined')

    
    elif args.mode == "collab":

        print("Running in mode REFINE")

        G = Generator(g_output_dim = mnist_dim).cuda()
        G = load_model(G, 'checkpoints')
        G = torch.nn.DataParallel(G).cuda()

        D = Discriminator(mnist_dim).cuda()
        D = load_discriminator_model(D, 'checkpoints')
        D = torch.nn.DataParallel(D).cuda()

        noise = NoiseDataset(dim=100)
        rollout_rate = 0.1
        rollout_steps = 50

        optim_d = optim.SGD(D.parameters(), lr=args.lr)

        n_epoch = args.epochs
        for epoch in trange(1, n_epoch+1, leave=True):   
            print(f"EPOCH {epoch}")        
            for batch_idx, (x, _) in enumerate(train_loader):
                if batch_idx % 100 == 0:
                    print(f"BATCH {batch_idx}")
                x = x.view(-1, mnist_dim)

                # synthesize refined samples
                noise_batch = noise.next_batch(args.batch_size).cuda()
                fake_batch = G(noise_batch)

                # probabilistic refinement
                proba_refine = torch.zeros([args.batch_size, mnist_dim], requires_grad=False, device="cuda")
                proba_steps = torch.LongTensor(args.batch_size,1).random_() % rollout_steps
                proba_steps_one_hot = torch.LongTensor(args.batch_size, rollout_steps)
                proba_steps_one_hot.zero_()
                proba_steps_one_hot.scatter_(1, proba_steps, 1)

                delta_refine = torch.zeros([args.batch_size, mnist_dim], requires_grad=True, device="cuda")
                optim_r = optim.Adam([delta_refine], lr=rollout_rate)
                label = torch.full((args.batch_size,1), 1, dtype=torch.float, device="cuda")
                for k in range(rollout_steps):
                    optim_r.zero_grad()
                    output = D(fake_batch.detach() + delta_refine)
                    loss_r = criterion(output, label)
                    loss_r.backward()
                    optim_r.step()

                    # probabilistic assignment
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
                save_models(G, D, 'checkpoints-refined')

        print('Refinement done')
