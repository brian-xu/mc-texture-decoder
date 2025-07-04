import argparse
import os

import torch
import torch.nn.functional as F
from dataset import TextureDataset
from model import TextureDecoder
from torch.utils.data import DataLoader
from tqdm import tqdm


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()


num_epochs = 10000
checkpoint_path = "ckpt"
os.makedirs(checkpoint_path, exist_ok=True)
train_dataset = TextureDataset()
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=train_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    drop_last=False,
)

model = TextureDecoder().to("cuda:0")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_eval_loss = 100.0
best_epoch = 0
for epoch in tqdm(range(num_epochs)):
    model.train()
    for idx, item in enumerate(train_loader):
        targets, labels = item
        targets = targets.to("cuda:0")
        labels = labels.to("cuda:0")
        outputs = model(labels)

        l2loss = l2_loss(outputs, targets)
        cosloss = cos_loss(outputs, targets)
        loss = l2loss + cosloss * 0.001

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_iter = epoch * len(train_loader) + idx

    if num_epochs - epoch < 5:
        eval_loss = 0.0
        model.eval()
        for idx, item in enumerate(test_loader):
            targets, labels = item
            targets = targets.to("cuda:0")
            labels = labels.to("cuda:0")
            with torch.no_grad():
                outputs = model(labels)
            loss = l2_loss(outputs, targets) + cos_loss(outputs, targets)
            eval_loss += loss * len(targets)
        eval_loss = eval_loss / len(train_dataset)
        print("eval_loss:{:.8f}".format(eval_loss))
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_path, "best_ckpt.pth"),
            )

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_path, f"{epoch}_ckpt.pth"),
            )

print(f"best_epoch: {best_epoch}")
print("best_loss: {:.8f}".format(best_eval_loss))
