"""stage2_classifier.py — DINOv2 wireframe visual classifier.

Only needed if Stage 1 precision < 85% at recall >= 80%.
Excludes 'unsure' labels from both training and evaluation.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.metrics import roc_auc_score
import numpy as np

from config import RENDERS_DIR, STAGE2_MODEL_PATH


class WireframeDataset(Dataset):
    def __init__(self, records, renders_dir, n_views=8):
        self.records = records
        self.renders_dir = Path(renders_dir)
        self.n_views = n_views
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        label = 1 if rec['label'] == 'artist' else 0
        views = []
        rdir = self.renders_dir / str(rec['uid'])
        for i in range(self.n_views):
            p = rdir / f'view_{i:02d}.png'
            if p.exists():
                views.append(self.transform(Image.open(p).convert('RGB')))
            else:
                views.append(torch.zeros(3, 224, 224))
        return torch.stack(views), label


class WireframeClassifier(nn.Module):
    def __init__(self, n_views=8):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.view_attn = nn.Sequential(nn.Linear(384, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(
            nn.Linear(384, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        B, V, C, H, W = x.shape
        feats = self.backbone(x.view(B * V, C, H, W)).view(B, V, -1)
        attn = torch.softmax(self.view_attn(feats), dim=1)
        pooled = (feats * attn).sum(dim=1)
        return self.head(pooled).squeeze(-1)


def train_stage2(records, renders_dir=None, epochs=20, batch_size=16, lr=1e-3):
    """
    Train Stage 2 classifier.

    Args:
        records: list of dicts with 'uid' and 'label' ('artist' or 'not_artist').
                 'unsure' labels must be excluded by the caller.
    """
    if renders_dir is None:
        renders_dir = RENDERS_DIR
    STAGE2_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = WireframeDataset(records, renders_dir)

    n_train = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, len(dataset) - n_train],
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4)

    model = WireframeClassifier().to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01,
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    best_auc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for views, labels in train_loader:
            views, labels = views.to(device), labels.float().to(device)
            loss = criterion(model(views), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for views, labels in val_loader:
                logits = model(views.to(device))
                preds.extend(torch.sigmoid(logits).cpu().numpy())
                gts.extend(labels.numpy())

        if len(set(gts)) > 1:
            auc = roc_auc_score(gts, preds)
        else:
            auc = 0.0
        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader):.4f} - Val AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), str(STAGE2_MODEL_PATH))

    print(f"Best Val AUC: {best_auc:.4f}")
    print(f"Saved: {STAGE2_MODEL_PATH}")
