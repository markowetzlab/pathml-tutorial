import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, loader, criterion, device):
    """Evaluation without the densecrf with the dice coefficient and loss"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot_dice = 0
    tot_loss = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                masks_pred = net(imgs)

            loss = criterion(masks_pred, true_masks)
            tot_loss += loss.item()

            if net.n_classes > 1:
                tot_dice += F.cross_entropy(masks_pred, true_masks).item()
            else:
                pred = torch.sigmoid(masks_pred)
                pred = (pred > 0.5).float()
                tot_dice += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot_dice / n_val, tot_loss / n_val
