import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, ConcatDataset, Dataset

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
#IDデータ
# CIFAR-100
cifar100_train = datasets.CIFAR100(root='/workspace/data', train=True, download=True, transform=transform)
cifar100_test  = datasets.CIFAR100(root='/workspace/data', train=False, download=True, transform=transform)
train_loader_cifar100 = DataLoader(cifar100_train, batch_size=64, shuffle=True)
test_loader_cifar100  = DataLoader(cifar100_test, batch_size=64)

#OODデータ
# CIFAR-10(NEAR)
cifar10_test  = datasets.CIFAR10(root='/workspace/data', train=False, download=True, transform=transform)
test_loader_cifar10  = DataLoader(cifar10_test, batch_size=64)
# SVHN(FAR)
svhn_test  = datasets.SVHN(root='/workspace/data', split='test', download=True, transform=transform)
test_loader_svhn   = DataLoader(svhn_test, batch_size=64)
# LSUN(近めなFAR)
lsun_test = datasets.LSUN(
    root='/workspace/data',
    classes=['test'],
    transform=transform
)
test_loader_lsun   = DataLoader(lsun_test, batch_size=64)
# MNIST(より遠いFAR)
mnist_test = datasets.MNIST(
    root='/workspace/data',
    train=False,
    download=True,
    transform=transform
)
test_loader_mnist   = DataLoader(mnist_test, batch_size=64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ood = models.resnet18(pretrained=False)
model_ood.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
model_ood.maxpool = nn.Identity()
model_ood.fc = nn.Linear(model_ood.fc.in_features, 100)
model_ood = model_ood.to(device)

def train(model, loader, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    print(next(model.parameters()).device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.3f}")

print("=== OODモデル学習 ===")
train(model_ood, train_loader_cifar100)


def test_ood(model, id_loader, ood_loader):
    model.eval()
    id_scores = []
    ood_scores = []

    with torch.no_grad():
        # ID
        for images, _ in id_loader:
            images = images.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            conf, _ = torch.max(probs, 1)

            id_scores.extend(conf.cpu().numpy())

        # OOD
        for images, _ in ood_loader:
            images = images.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            conf, _ = torch.max(probs, 1)

            ood_scores.extend(conf.cpu().numpy())

    id_scores = np.array(id_scores)
    ood_scores = np.array(ood_scores)

    # ===== 向き確認 =====
    print("ID mean:", id_scores.mean())
    print("OOD mean:", ood_scores.mean())

    # ===== AUROC =====
    y_true = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores))
    ])
    y_score = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(y_true, y_score)

    # ===== FPR95 =====
    thresholds = np.sort(y_score)
    fpr95 = 1.0  # 初期値

    for th in thresholds:
        tp = np.sum(id_scores >= th)
        fn = np.sum(id_scores < th)
        fp = np.sum(ood_scores >= th)
        tn = np.sum(ood_scores < th)

        tpr = tp / (tp + fn)

        if tpr >= 0.95:
            fpr = fp / (fp + tn)
            fpr95 = min(fpr95, fpr)

    print("AUROC:", auroc)
    print("FPR95:", fpr95)

print("=== テスト①（NEAR_OOD:CIFAR-10）===")
test_ood(model_ood, test_loader_cifar100, test_loader_cifar10)

print("=== テスト②（FAR_OOD:SVHN）===")
test_ood(model_ood, test_loader_cifar100, test_loader_svhn)

print("=== テスト③（FAR_OOD:LSUN）===")
test_ood(model_ood, test_loader_cifar100, test_loader_lsun)

print("=== テスト②（FAR_OOD:MNIST）===")
test_ood(model_ood, test_loader_cifar100, test_loader_mnist)