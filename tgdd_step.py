# tgdd_step.py
import torch
import torch.nn.functional as F
from tgdd_utils import load_snapshot_params_into_net_, forward_logits_and_feats, classwise_mean

def tgdd_update_step(args, expert_trajectory, image_syn, label_syn,
                     get_images, criterion, net_ctor, optimizer_img):
    """
    Perform one TGDD update step.
    - expert_trajectory: list of (snapshot_params_list) for each epoch
    - image_syn: [N, C, H, W] synthetic images (requires_grad=True)
    - label_syn: [N] labels
    - get_images(c, n): function to get n real images of class c (CPU tensor)
    - criterion: CrossEntropyLoss
    - net_ctor: function returning a fresh network to load params
    - optimizer_img: optimizer for image_syn

    Returns: (loss_total, stats_dict)
    """
    device = args.device
    C = int(label_syn.max().item() + 1)
    ipc = int(args.ipc)
    L = int(args.expert_region_len)
    alpha = float(args.alpha_sdc)

    # Choose random stage j such that [j, j+L-1] is in range
    max_j = len(expert_trajectory) - L
    j = int(torch.randint(0, max_j + 1, (1,)).item())
    k = int(torch.randint(0, L, (1,)).item())

    theta_ext = expert_trajectory[j]
    theta_exp = expert_trajectory[j+k]

    # Build two nets and optionally wrap with DataParallel
    net_ext = net_ctor().to(device).eval()
    net_exp = net_ctor().to(device).eval()
    if args.distributed:
        net_ext = torch.nn.DataParallel(net_ext)
        net_exp = torch.nn.DataParallel(net_exp)

    for p in net_ext.parameters():
        p.requires_grad_(False)
    for p in net_exp.parameters():
        p.requires_grad_(False)

    load_snapshot_params_into_net_(net_ext, theta_ext)
    load_snapshot_params_into_net_(net_exp, theta_exp)

    # Sample class-balanced real and synthetic batches
    b_syn = min(ipc, args.b_syn_per_class)
    b_real = args.b_real_per_class
    syn_idx = []
    real_x = []
    real_y = []
    for c in range(C):
        st = c*ipc; ed = (c+1)*ipc
        perm = torch.randperm(ed-st, device=device)[:b_syn] + st
        syn_idx.append(perm)
        x_c = get_images(c, b_real).to(device)
        y_c = torch.full((b_real,), c, device=device, dtype=torch.long)
        real_x.append(x_c); real_y.append(y_c)

    syn_idx = torch.cat(syn_idx)
    syn_x = image_syn[syn_idx]
    syn_y = label_syn[syn_idx].to(device)
    real_x = torch.cat(real_x, dim=0)
    real_y = torch.cat(real_y, dim=0)

    # (Optional) DiffAugment here, if valid for your data
    if not args.no_aug:
        # Unspecified: domain-specific augmentations (placeholder)
        pass

    # Compute features
    with torch.no_grad():
        _, feats_real = forward_logits_and_feats(net_ext, real_x, use_penultimate=args.tgdd_use_penultimate)
    _, feats_syn = forward_logits_and_feats(net_ext, syn_x, use_penultimate=args.tgdd_use_penultimate)
    feats_real = feats_real.view(feats_real.size(0), -1)
    feats_syn = feats_syn.view(feats_syn.size(0), -1)

    mu_real = classwise_mean(feats_real, real_y, C)
    mu_syn  = classwise_mean(feats_syn, syn_y, C)

    # MMD (centroid) loss
    loss_mmd = torch.mean((mu_syn - mu_real)**2)

    # SDC: cross-entropy on synthetic
    logits = net_exp(syn_x)
    loss_sdc = criterion(logits, syn_y)

    loss_total = loss_mmd + alpha * loss_sdc
    optimizer_img.zero_grad(set_to_none=True)
    loss_total.backward()
    grad_norm = float(image_syn.grad.detach().norm().cpu()) if image_syn.grad is not None else 0.0
    optimizer_img.step()

    # Debug stats
    with torch.no_grad():
        gaps = torch.mean((mu_syn - mu_real)**2, dim=1).cpu().numpy().tolist()
    stats = {
        "Loss_total": float(loss_total.detach().cpu()),
        "Loss_mmd": float(loss_mmd.detach().cpu()),
        "Loss_sdc": float(loss_sdc.detach().cpu()),
        "alpha": alpha,
        "stage_j": j,
        "delta_k": k,
        "grad_norm": grad_norm,
        "b_syn_per_class": b_syn,
        "b_real_per_class": b_real,
    }
    for c, gap in enumerate(gaps):
        stats[f"gap_class_{c}"] = float(gap)
    return loss_total, stats

