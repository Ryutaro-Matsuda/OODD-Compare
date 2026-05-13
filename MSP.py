import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torch.nn.functional as F


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.dropout = nn.Dropout(p=dropout_rate)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=True
                )
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()

        self.in_planes = 16

        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(
            3,
            nStages[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        self.layer1 = self._wide_layer(
            WideBasic,
            nStages[1],
            n,
            dropout_rate,
            stride=1
        )

        self.layer2 = self._wide_layer(
            WideBasic,
            nStages[2],
            n,
            dropout_rate,
            stride=2
        )

        self.layer3 = self._wide_layer(
            WideBasic,
            nStages[3],
            n,
            dropout_rate,
            stride=2
        )

        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks,
                    dropout_rate, stride):

        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    dropout_rate,
                    stride
                )
            )
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.relu(self.bn1(out))

        out = F.avg_pool2d(out, 8)

        out = out.view(out.size(0), -1)

        out = self.linear(out)

        return out


gray_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
])
color_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
])
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
])
#IDデータ
# CIFAR-100
cifar100_train = datasets.CIFAR100(root='/workspace/data', train=True, download=True, transform=train_transform)
cifar100_test  = datasets.CIFAR100(root='/workspace/data', train=False, download=True, transform=color_transform)
train_loader_cifar100 = DataLoader(cifar100_train, batch_size=64, shuffle=True)
test_loader_cifar100  = DataLoader(cifar100_test, batch_size=64)

#OODデータ
# CIFAR-10(NEAR)
cifar10_test  = datasets.CIFAR10(root='/workspace/data', train=False, download=True, transform=color_transform)
test_loader_cifar10  = DataLoader(cifar10_test, batch_size=64)
# SVHN(FAR)
svhn_test  = datasets.SVHN(root='/workspace/data', split='test', download=True, transform=color_transform)
test_loader_svhn   = DataLoader(svhn_test, batch_size=64)
# LSUN(近めなFAR)
# lsun_test = datasets.LSUN(
#     root='/workspace/data',
#     classes='test',
#     transform=transform
# )
# test_loader_lsun   = DataLoader(lsun_test, batch_size=64)
# MNIST(より遠いFAR)
mnist_test = datasets.MNIST(
    root='/workspace/data',
    train=False,
    download=True,
    transform=gray_transform
)
test_loader_mnist   = DataLoader(mnist_test, batch_size=64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ood = WideResNet(
    depth=28,
    widen_factor=10,
    dropout_rate=0.3,
    num_classes=100
)
model_ood = model_ood.to(device)

def train(model, loader, epochs=200):
    start = time.time()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150],
        gamma=0.1
    )

    print(next(model.parameters()).device)

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

        scheduler.step()

        acc = correct / total

        print(
            f"Epoch {epoch+1} | "
            f"Loss: {total_loss:.3f} | "
            f"Train Acc: {acc:.4f}"
        )
    end = time.time()
    elapsed = end - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"{hours}h {minutes}m {seconds:.2f}s")

print("=== OODモデル学習 ===")
train(model_ood, train_loader_cifar100)


def test_msp(model, id_loader, ood_loader):
    start = time.time()
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
    end = time.time()
    elapsed = end - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"{hours}h {minutes}m {seconds:.2f}s")

import torch.nn.functional as F
def odin_score(model, images, T=100.0, epsilon=0.001):
    model.eval()
    images = images.to(device)
    images.requires_grad = True

    # 1回目のforward（勾配計算用）
    logits = model(images)
    logits = logits / T

    preds = logits.argmax(dim=1)
    loss = F.cross_entropy(logits, preds)
    loss.backward()

    # 入力に微小ノイズを加える
    grad = images.grad.data
    perturbed = images - epsilon * torch.sign(grad)

    # 2回目のforward（スコア取得）
    logits = model(perturbed)
    logits = logits / T
    probs = F.softmax(logits, dim=1)

    conf, _ = probs.max(dim=1)
    return conf.detach()

def test_odin(model, id_loader, ood_loader, T=100.0, epsilon=0.001):
    start = time.time()
    model.eval()
    id_scores = []
    ood_scores = []

    # ===== ID =====
    for images, _ in id_loader:
        scores = odin_score(model, images, T, epsilon)
        id_scores.extend(scores.cpu().numpy())

    # ===== OOD =====
    for images, _ in ood_loader:
        scores = odin_score(model, images, T, epsilon)
        ood_scores.extend(scores.cpu().numpy())

    id_scores = np.array(id_scores)
    ood_scores = np.array(ood_scores)

    # 向き確認
    print("ID mean:", id_scores.mean())
    print("OOD mean:", ood_scores.mean())

    # AUROC
    y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    y_score = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(y_true, y_score)

    # FPR95
    thresholds = np.sort(y_score)
    tpr_list, fpr_list = [], []

    for th in thresholds:
        tp = np.sum(id_scores >= th)
        fn = np.sum(id_scores < th)
        fp = np.sum(ood_scores >= th)
        tn = np.sum(ood_scores < th)

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    valid = [fpr for tpr, fpr in zip(tpr_list, fpr_list) if tpr >= 0.95]
    fpr95 = min(valid) if len(valid) > 0 else 1.0

    print("AUROC:", auroc)
    print("FPR95:", fpr95)
    end = time.time()
    elapsed = end - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"{hours}h {minutes}m {seconds:.2f}s")



def energy_score(model, images, T=1.0):
    model.eval()
    images = images.to(device)

    with torch.no_grad():
        logits = model(images)
        energy = -T * torch.logsumexp(logits / T, dim=1)

    return energy.cpu().numpy()

def test_energy(model, id_loader, ood_loader, T=1.0):
    start = time.time()
    model.eval()
    id_scores = []
    ood_scores = []

    # ===== ID =====
    for images, _ in id_loader:
        scores = energy_score(model, images, T)
        id_scores.extend(scores)

    # ===== OOD =====
    for images, _ in ood_loader:
        scores = energy_score(model, images, T)
        ood_scores.extend(scores)

    id_scores = np.array(id_scores)
    ood_scores = np.array(ood_scores)

    # ===== 向き確認（重要）=====
    print("ID mean:", id_scores.mean())
    print("OOD mean:", ood_scores.mean())

    # ===== AUROC =====
    # Energyは小さいほどID → 符号反転
    y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    y_score = -np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(y_true, y_score)

    # ===== FPR95 =====
    thresholds = np.sort(y_score)
    fpr95 = 1.0

    for th in thresholds:
        tp = np.sum(y_score[:len(id_scores)] >= th)
        fn = np.sum(y_score[:len(id_scores)] < th)
        fp = np.sum(y_score[len(id_scores):] >= th)
        tn = np.sum(y_score[len(id_scores):] < th)

        tpr = tp / (tp + fn)

        if tpr >= 0.95:
            fpr = fp / (fp + tn)
            fpr95 = min(fpr95, fpr)

    print("AUROC:", auroc)
    print("FPR95:", fpr95)
    end = time.time()
    elapsed = end - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"{hours}h {minutes}m {seconds:.2f}s")

def extract_features(model, x):
    # ResNet18用
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x

def compute_class_stats(model, loader, num_classes=100):
    model.eval()

    features = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = extract_features(model, images)

            for f, y in zip(feats, labels):
                features[y].append(f.cpu().numpy())

    class_means = []
    for c in range(num_classes):
        class_means.append(np.mean(features[c], axis=0))

    class_means = np.array(class_means)
    return class_means

def compute_precision(class_means, model, loader):
    model.eval()

    diffs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            feats = extract_features(model, images)
            feats = feats.cpu().numpy()

            labels = labels.numpy()

            for f, y in zip(feats, labels):
                diff = f - class_means[y]
                diffs.append(diff)

    diffs = np.array(diffs)

    # クラス平均との差の共分散
    cov = np.cov(diffs, rowvar=False)

    # 数値安定化
    cov += 1e-6 * np.eye(cov.shape[0])

    precision = np.linalg.inv(cov)

    return precision

def mahalanobis_score(model, images, class_means, precision):

    model.eval()
    images = images.to(device)

    with torch.no_grad():
        feats = extract_features(model, images).cpu().numpy()

    scores = []

    for f in feats:
        dists = []
        for mean in class_means:
            diff = f - mean
            dist = np.dot(np.dot(diff, precision), diff.T)
            dists.append(dist)

        score = -min(dists)  # 小さい距離 → ID
        scores.append(score)

    return np.array(scores)

def test_mahalanobis(model, id_loader, ood_loader, class_means, precision):
    start = time.time()
    id_scores = []
    ood_scores = []

    # ID
    for images, _ in id_loader:
        scores = mahalanobis_score(model, images, class_means, precision)
        id_scores.extend(scores)

    # OOD
    for images, _ in ood_loader:
        scores = mahalanobis_score(model, images, class_means, precision)
        ood_scores.extend(scores)

    id_scores = np.array(id_scores)
    ood_scores = np.array(ood_scores)

    print("ID mean:", id_scores.mean())
    print("OOD mean:", ood_scores.mean())

    y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    y_score = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(y_true, y_score)

    # FPR95
    thresholds = np.sort(y_score)
    fpr95 = 1.0

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
    end = time.time()
    elapsed = end - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"{hours}h {minutes}m {seconds:.2f}s")

def test_accuracy(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print("Accuracy:", acc)

print("MSP_SCORE")
print("=== テスト①（NEAR_OOD:CIFAR-10）===")
test_msp(model_ood, test_loader_cifar100, test_loader_cifar10)

print("=== テスト②（FAR_OOD:SVHN）===")
test_msp(model_ood, test_loader_cifar100, test_loader_svhn)

# print("=== テスト③（FAR_OOD:LSUN）===")
# test_msp(model_ood, test_loader_cifar100, test_loader_lsun)

print("=== テスト④（FAR_OOD:MNIST）===")
test_msp(model_ood, test_loader_cifar100, test_loader_mnist)

print("ODIN_SCORE")
print("=== テスト①（NEAR_OOD:CIFAR-10）===")
test_odin(model_ood, test_loader_cifar100, test_loader_cifar10)

print("=== テスト②（FAR_OOD:SVHN）===")
test_odin(model_ood, test_loader_cifar100, test_loader_svhn)

# print("=== テスト③（FAR_OOD:LSUN）===")
# test_odin(model_ood, test_loader_cifar100, test_loader_lsun)

print("=== テスト④（FAR_OOD:MNIST）===")
test_odin(model_ood, test_loader_cifar100, test_loader_mnist)

print("ENERGY_SCORE")
print("=== テスト①（NEAR_OOD:CIFAR-10）===")
test_energy(model_ood, test_loader_cifar100, test_loader_cifar10)

print("=== テスト②（FAR_OOD:SVHN）===")
test_energy(model_ood, test_loader_cifar100, test_loader_svhn)

# print("=== テスト③（FAR_OOD:LSUN）===")
# test_energy(model_ood, test_loader_cifar100, test_loader_lsun)

print("=== テスト④（FAR_OOD:MNIST）===")
test_energy(model_ood, test_loader_cifar100, test_loader_mnist)

print("MAHALANOBIS_SCORE")
class_means = compute_class_stats(model_ood, train_loader_cifar100)
precision = compute_precision(class_means, model_ood, train_loader_cifar100)
print("=== テスト①（NEAR_OOD:CIFAR-10）===")
test_mahalanobis(model_ood, test_loader_cifar100, test_loader_cifar10,class_means,precision)

print("=== テスト②（FAR_OOD:SVHN）===")
test_mahalanobis(model_ood, test_loader_cifar100, test_loader_svhn,class_means,precision)

# print("=== テスト③（FAR_OOD:LSUN）===")
# test_mahalanobis(model_ood, test_loader_cifar100, test_loader_lsun,class_means,precision)

print("=== テスト④（FAR_OOD:MNIST）===")
test_mahalanobis(model_ood, test_loader_cifar100, test_loader_mnist,class_means,precision)

print("Model Accuracy")
test_accuracy(model_ood, test_loader_cifar100)