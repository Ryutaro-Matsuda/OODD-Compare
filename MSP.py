import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, ConcatDataset, Dataset

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
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
])
#IDデータ
# CIFAR-100
cifar100_train = datasets.CIFAR100(root='/workspace/data', train=True, download=True, transform=color_transform)
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

model_ood = models.resnet18(pretrained=False)
model_ood.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
model_ood.maxpool = nn.Identity()
model_ood.fc = nn.Linear(model_ood.fc.in_features, 100)
model_ood = model_ood.to(device)

def train(model, loader, epochs=300):
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


def test_msp(model, id_loader, ood_loader):
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

import torch.nn.functional as F
def odin_score(model, images, T=1000.0, epsilon=0.001):
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

def test_odin(model, id_loader, ood_loader, T=1000.0, epsilon=0.001):
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



def energy_score(model, images, T=1.0):
    model.eval()
    images = images.to(device)

    with torch.no_grad():
        logits = model(images)
        energy = -T * torch.logsumexp(logits / T, dim=1)

    return energy.cpu().numpy()

def test_energy(model, id_loader, ood_loader, T=1.0):
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

    all_features = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            feats = extract_features(model, images)
            all_features.append(feats.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)

    cov = np.cov(all_features, rowvar=False)
    precision = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))

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