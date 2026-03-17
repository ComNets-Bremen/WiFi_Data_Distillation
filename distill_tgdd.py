import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import wandb
import copy
import random
from reparam_module import ReparamModule
import pandas as pd
import sys
from tgdd_step import tgdd_update_step
import torchvision          # <-- also add this (needed later in the script)
import warnings  

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    # Check args
    print("hello")
    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")
    if args.texture and args.pix_init == "real":
        print("WARNING: texture with real init may require long smoothing")

    # Device, datasets
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes, class_names, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = \
        get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    if args.dataset == 'xrf55':
        im_size = (270, 1000); num_classes = 55; channel = 1
    elif args.dataset == 'widar':
        im_size = (20, 20); num_classes = 6; channel = 22
    elif args.dataset == 'mmfi':
        im_size = (342, 350); num_classes = 27; channel = 1
    else:
        raise ValueError("Unsupported dataset")
    args.im_size = im_size

    # Default override: TGDD mode lr_img
    if args.distill_mode == 'tgdd' and args.lr_img == 10:
        args.lr_img = 10
        print("[TGDD] distill_mode=tgdd: setting lr_img=0.02 (default override)")


    # Reload args from wandb.config
    wandb.init(project="DatasetDistillation", job_type="Run", config=vars(args))



    # Build real dataset indices
    images_all = []; labels_all = []; indices_class = [[] for _ in range(num_classes)]
    print("BUILDING DATASET")
    for i in range(len(dst_train)):
        img, lab = dst_train[i]
        c = class_map[int(lab)]
        images_all.append(img.unsqueeze(0)); labels_all.append(c)
        indices_class[c].append(i)
    images_all = torch.cat(images_all, dim=0).cpu()
    labels_all = torch.tensor(labels_all, dtype=torch.long)

    for c in range(num_classes):
        print(f"class {c}: {len(indices_class[c])} real images")

    def get_images(c, n):
        idx = np.random.choice(indices_class[c], size=n, replace=False)
        return images_all[idx]

    # Initialize synthetic images
    label_syn = torch.tensor([c for c in range(num_classes) for _ in range(args.ipc)],
                             dtype=torch.long, device=args.device)
    if args.texture:
        image_syn = torch.randn((num_classes*args.ipc, channel,
                                 im_size[0]*args.canvas_size, im_size[1]*args.canvas_size))
    else:
        image_syn = torch.randn((num_classes*args.ipc, channel, im_size[0], im_size[1]))

    if args.pix_init == 'real':
        print("Initializing synthetic from real images")
        for c in range(num_classes):
            idx = np.random.choice(indices_class[c], size=args.ipc, replace=False)
            image_syn.data[c*args.ipc:(c+1)*args.ipc] = images_all[idx].to(args.device)

    image_syn = image_syn.to(args.device).detach().requires_grad_(True)
    syn_lr = torch.tensor(args.lr_teacher).to(args.device).detach()  # constant
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5) if args.distill_mode=='mtt' else None

    criterion = nn.CrossEntropyLoss().to(args.device)

    # Load expert buffers
    expert_dir = os.path.join(args.buffer_path, args.dataset, args.model)
    expert_files = []
    n=0
    while os.path.exists(os.path.join(expert_dir, f"replay_buffer_{n}.pt")):
        expert_files.append(os.path.join(expert_dir, f"replay_buffer_{n}.pt")); n+=1
    assert expert_files, f"No buffers in {expert_dir}"
    random.shuffle(expert_files)
    buffer = torch.load(expert_files[0])
    expert_idx = 0; file_idx = 0

    best_acc = {m:0 for m in get_eval_pool(args.eval_mode,args.model,args.model)}
    best_std = {m:0 for m in best_acc}

    for it in range(args.Iteration+1):
        save_this_it=False
        wandb.log({"Iteration": it}, step=it)

        # Periodic evaluation
        if it % args.eval_it == 0:
            for model_eval in best_acc:
                accs = []; for_train=[]
                for _ in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                    image_eval = image_syn.detach().clone()
                    label_eval = label_syn.detach().clone()
                    args.lr_net = float(syn_lr)
                    _, acc_train, acc_test = evaluate_synset(0, net_eval, image_eval, label_eval, testloader, args, texture=args.texture)
                    accs.append(acc_test); for_train.append(acc_train)
                mean_test = float(np.mean(accs)); std_test = float(np.std(accs))
                if mean_test > best_acc[model_eval]:
                    best_acc[model_eval]=mean_test; best_std[model_eval]=std_test; save_this_it=True
                wandb.log({f"TestAcc/{model_eval}": mean_test, f"TestStd/{model_eval}": std_test}, step=it)

        # Save best & progress images
        if it in [0, args.Iteration] or save_this_it:
            torch.save(image_syn.cpu(), f"images_{it}.pt")
            torch.save(label_syn.cpu(), f"labels_{it}.pt")
            # NEW - handles 22-channel Widar data
            _vis = image_syn.detach().cpu()
            if _vis.shape[1] > 3:
               _vis = _vis[:, :1, :, :]
            wandb.log({"SynthImages": wandb.Image(torchvision.utils.make_grid(_vis, normalize=True))}, step=it)
            #wandb.log({"SynthImages": wandb.Image(torchvision.utils.make_grid(image_syn.detach().cpu(), normalize=True))}, step=it)

        # Log synthetic learning rate
        wandb.log({"Synthetic_LR": float(syn_lr)}, step=it)

        # Get expert trajectory (same as original code)
        if args.load_all:
            expert_trajectory = random.choice(buffer)
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0; file_idx = (file_idx+1) % len(expert_files)
                buffer = torch.load(expert_files[file_idx])
        # -- 

        # TGDD mode update
        if args.distill_mode == 'tgdd':
            loss_total, stats = tgdd_update_step(
                args=args,
                expert_trajectory=expert_trajectory,
                image_syn=image_syn,
                label_syn=label_syn,
                get_images=get_images,
                criterion=criterion,
                net_ctor=lambda: get_network(args.model, channel, num_classes, im_size),
                optimizer_img=optimizer_img
            )
            wandb.log(stats, step=it)
            continue

        # MTT mode update (unchanged)
        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)
        student_net = ReparamModule(student_net)
        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)
        student_net.train()
        num_params = sum(p.numel() for p in student_net.parameters())

        start_epoch = np.random.randint(0, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]
        target_params = expert_trajectory[start_epoch + args.expert_epochs]
        target_params = torch.cat([p.data.reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.reshape(-1) for p in starting_params], 0).requires_grad_(True)]
        for step in range(args.syn_steps):
            if step == 0:
                indices = torch.randperm(image_syn.size(0))
            x = image_syn[indices[step % len(indices)]].unsqueeze(0).to(args.device)
            y = label_syn[indices[step % len(indices)]].unsqueeze(0).to(args.device)
            if args.texture:
                x = augment_texture(x, args)  # unspecified utility
            out = student_net(x, flat_param=student_params[-1])
            ce = criterion(out, y)
            grad = torch.autograd.grad(ce, student_params[-1], create_graph=True)[0]
            student_params.append(student_params[-1] - syn_lr * grad)
        loss = F.mse_loss(student_params[-1], target_params, reduction='sum') / num_params
        loss /= (F.mse_loss(torch.cat([p.data.reshape(-1) for p in starting_params],0), target_params, reduction='sum') / num_params)
        optimizer_img.zero_grad(); optimizer_lr.zero_grad()
        loss.backward()
        optimizer_img.step()
        optimizer_lr.step()
        wandb.log({"Grand_Loss": float(loss.detach()), "Start_Epoch": start_epoch}, step=it)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette',
                        help='ImageNet subset. This only does anything when --dataset=ImageNet')
    parser.add_argument('--model', type=str, default='widar_CNN', help='model')
    parser.add_argument('--res', type=int, default=128, help='resolution for imagenet')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000,
                        help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=0.02,
                        help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-06,
                        help='learning rate for updating learning rate (used in MTT mode only)')
    parser.add_argument('--lr_teacher', type=float, default=0.001,
                        help='initialization for synthetic learning rate')
    parser.add_argument('--lr_init', type=float, default=0.01,
                        help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None,
                        help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256,
                        help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3,
                        help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20,
                        help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25,
                        help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--load_all', action='store_true',
                        help="only use if you can fit all expert trajectories into RAM")
    parser.add_argument('--no_aug', type=bool, default=False,
                        help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1,
                        help='number of canvas samples per iteration')

    parser.add_argument('--max_files', type=int, default=None,
                        help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None,
                        help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    # =========================
    # TGDD-specific arguments
    # =========================
    parser.add_argument('--distill_mode', type=str, default='tgdd', choices=['mtt', 'tgdd'],
                        help='distillation mode: original MTT or TGDD-style distribution matching')
    parser.add_argument('--alpha_sdc', type=float, default=0.5,
                        help='weight for stage-wise distribution constraint')
    parser.add_argument('--expert_region_len', type=int, default=7,
                        help='L: length of expert region used for TGDD')
    parser.add_argument('--b_real_per_class', type=int, default=64,
                        help='number of real samples per class per TGDD step')
    parser.add_argument('--b_syn_per_class', type=int, default=32,
                        help='number of synthetic samples per class per TGDD step')
    parser.add_argument('--tgdd_use_penultimate', action='store_true',
                        help='use penultimate features if available, otherwise fallback safely')
    parser.add_argument('--tgdd_match_aug', action='store_true',
                        help='apply same augmentation to real and synthetic batches during TGDD matching')
    parser.add_argument('--distributed', action='store_true', 
                    help='use DataParallel for multi-GPU training')

    args = parser.parse_args()
    main(args)

