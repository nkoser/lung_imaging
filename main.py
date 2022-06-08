import os
from glob import glob
import matplotlib.pyplot as plt
import pkbar as pkbar
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from data_loader import Lung_loader
from resUnet import ResUnet3D
import pytorch_ssim #!pip install git+https://github.com/jinh0park/pytorch-ssim-3D.git

from utils import PSNR

down_img_paths = glob(os.path.join('/Users/niklaskoser/Desktop/down_samesize/Training', '*.nii.gz'))
down_img_paths.sort()
org_img_paths = glob(os.path.join('/Users/niklaskoser/Desktop/down_samesize/Training', '*.nii.gz'))
org_img_paths.sort()

exp_name = 'test_I'
folder_path = 'Users/niklaskoser/Desktop/results'


train_set = Lung_loader(down_img_paths[:30], org_img_paths[:30])
val_set = Lung_loader(down_img_paths[30:], org_img_paths[30:])
imgs = val_set[0]
plt.imshow(imgs[0][0, :, 32, :].cpu().rot90().numpy(), 'gray')
plt.show()

plt.imshow(imgs[1][0, :, 32, :].cpu().rot90().numpy(), 'gray')
plt.show()


train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=True)



model = ResUnet3D(channel=1)

torch.manual_seed(101)

nr_epx = 25

# initialize the optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
criterion = nn.MSELoss()

run_loss = 0
run_acc = 0

# Save loss of each epoch
losses_training = []
losses_validate = []
psnrs = []
ssims = []

iterate = len(train_loader)
bar = pkbar.Pbar(name="Progress", target=iterate)
best_ssim = 0
for epoch in range(nr_epx):

    ########################################
    #               TRAINING               #
    ########################################

    sum_loss = 0

    # Parameters must be trainable
    model.train()
    with torch.set_grad_enabled(True):
        print(f"{20 * '-'} Epoche: {epoch + 1} {20 * '-'}")
        # loop to process all training samples (packed into batches)
        for batch_ndx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = x.float(), y.float()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            bar.update(batch_ndx)

        losses_training.append(sum_loss / len(train_loader))

    ########################################
    #              VALIDATION              #
    ########################################

    sum_loss, psnr_sum, ssim_sum = 0, 0, 0

    # Parameters must not be trainable
    model.eval()
    with torch.set_grad_enabled(False):  # TODO add bool value

        for batch_ndx, (x, y) in enumerate(val_loader):
            x, y = x.float(), y.float()
            # print(f"Size: {x.shape}")
            pred = model(x)
            predicted = torch.argmax(pred)
            loss = criterion(pred, y)
            sum_loss += loss.item()
            psnr_sum += PSNR()(pred, y).item()
            ssim_sum += pytorch_ssim.ssim3D(pred, y).item()
        losses_validate.append(sum_loss / len(val_loader))
        ssims.append(ssim_sum / len(val_loader))
        psnrs.append(psnr_sum / len(val_loader))

        if ssims[-1] > best_ssim:
            print(f" Best Validation SSIM: {ssims[-1]}")
            best_ssim = ssims[-1]
            torch.save(model, os.path.join(folder_path, exp_name + "best.pth"))
        print(f"Epoche {epoch} -- Trainings Loss {losses_training[-1]}")
        print(f"Epoche {epoch} -- Validation Loss {losses_validate[-1]}")
        print(f"Epoche {epoch} -- Validation PSNR {psnrs[-1]}")
        print(f"Epoche {epoch} -- Validation SSIM {ssims[-1]}")

torch.save(model, os.path.join(folder_path, exp_name + ".pth"))
df = pd.DataFrame([losses_validate, losses_training, psnrs, ssims])
df = (df.transpose())
df.columns = ["Validation", "Training", "PSNR", "SSIM"]
df.to_csv(os.path.join(folder_path, "val_loss.csv"))
