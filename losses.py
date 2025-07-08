import torch

def compute_losses_diff(opts, src_pcd0, src_pcd, pred_transforms, transform_gt, loss_type = 'mae', reduction = 'mean'):

    losses = {}
    num_iter = len(pred_transforms)

    # Compute losses
    gt_src_transformed = transform(transform_gt, src_pcd0)
    if loss_type == 'mse':
        # MSE loss to the groundtruth (does not take into account possible symmetries)
        criterion = torch.nn.MSELoss(reduction=reduction)
        for i in range(num_iter):
            pred_src_transformed = transform(pred_transforms[i], src_pcd)
            if reduction.lower() == 'mean':
                losses['mse_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
            elif reduction.lower() == 'none':
                losses['mse_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed),
                                                        dim=[-1, -2])
    elif loss_type == 'mae':
        # MSE loss to the groundtruth (does not take into account possible symmetries)
        criterion = torch.nn.L1Loss(reduction=reduction)
        for i in range(num_iter):
            pred_src_transformed = transform(pred_transforms[i], src_pcd)
            if reduction.lower() == 'mean':
                losses['mae_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
            elif reduction.lower() == 'none':
                losses['mae_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed), dim=[-1, -2])
    elif loss_type == 'chamfer':
        for i in range(num_iter):
            pred_src_transformed = transform(pred_transforms[i], src_pcd)
            losses['chamfer_{}'.format(i)] = chamfer_distance(pred_src_transformed, gt_src_transformed)
    else:
        raise NotImplementedError

    discount_factor = 0.5  # Early iterations will be discounted
    total_losses = []
    for k in losses:
        discount = discount_factor ** (num_iter - int(k[k.rfind('_')+1:]) - 1)
        total_losses.append(losses[k] * discount)
    losses['total'] = torch.sum(torch.stack(total_losses), dim=0)

    return losses

def compute_losses(opts, src_pcd, pred_transforms, transform_gt, loss_type = 'mae', reduction = 'mean'):

    losses = {}
    num_iter = len(pred_transforms)

    # Compute losses
    gt_src_transformed = transform(transform_gt, src_pcd)
    if loss_type == 'mse':
        # MSE loss to the groundtruth (does not take into account possible symmetries)
        criterion = torch.nn.MSELoss(reduction=reduction)
        for i in range(num_iter):
            pred_src_transformed = transform(pred_transforms[i], src_pcd)
            if reduction.lower() == 'mean':
                losses['mse_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
            elif reduction.lower() == 'none':
                losses['mse_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed),
                                                        dim=[-1, -2])
    elif loss_type == 'mae':
        # MSE loss to the groundtruth (does not take into account possible symmetries)
        criterion = torch.nn.L1Loss(reduction=reduction)
        for i in range(num_iter):
            pred_src_transformed = transform(pred_transforms[i], src_pcd)
            if reduction.lower() == 'mean':
                losses['mae_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
            elif reduction.lower() == 'none':
                losses['mae_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed), dim=[-1, -2])
    elif loss_type == 'chamfer':
        for i in range(num_iter):
            pred_src_transformed = transform(pred_transforms[i], src_pcd)
            losses['chamfer_{}'.format(i)] = chamfer_distance(pred_src_transformed, gt_src_transformed)
    else:
        raise NotImplementedError

    discount_factor = 0.5  # Early iterations will be discounted
    total_losses = []
    for k in losses:
        discount = discount_factor ** (num_iter - int(k[k.rfind('_')+1:]) - 1)
        total_losses.append(losses[k] * discount)
    losses['total'] = torch.sum(torch.stack(total_losses), dim=0)

    return losses

def chamfer_distance(pc1, pc2):
    """
    Computes Chamfer Distance between two point clouds.
    Args:
        pc1: (B, N, 3) - batch of source point clouds
        pc2: (B, M, 3) - batch of target point clouds
    Returns:
        chamfer_loss: scalar
    """
    B, N, _ = pc1.shape
    _, M, _ = pc2.shape
    pc1_expand = pc1.unsqueeze(2)  # (B, N, 1, 3)
    pc2_expand = pc2.unsqueeze(1)  # (B, 1, M, 3)
    dist = torch.norm(pc1_expand - pc2_expand, dim=3)  # (B, N, M)
    min_dist_pc1, _ = torch.min(dist, dim=2)  # (B, N)
    min_dist_pc2, _ = torch.min(dist, dim=1)  # (B, M)
    chamfer_loss = min_dist_pc1.mean(dim=1) + min_dist_pc2.mean(dim=1)  # (B,)
    return chamfer_loss.mean()  # scalar

def transform(g, a, normals=None):
    R = g[..., :3, :3]  # (B, 3, 3)
    p = g[..., :3, 3]  # (B, 3)

    if len(g.size()) == len(a.size()):
        b = torch.matmul(a, R.transpose(-1, -2)) + p[..., None, :]
    else:
        raise NotImplementedError
        b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p  # No batch. Not checked

    if normals is not None:
        rotated_normals = normals @ R.transpose(-1, -2)
        return b, rotated_normals

    else:
        return b