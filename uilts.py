import random
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from scipy import io
import shutil


device = torch.device('cuda')


def get_dataset(dataset_path, dataset_gt_path):
    img = io.loadmat(dataset_path)['indian_pines_corrected']
    gt = io.loadmat(dataset_gt_path)['indian_pines_gt']
    ignored_labels = [0]
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print(
            "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)
    ignored_labels = list(set(ignored_labels))
    data = np.asarray(img, dtype='float32')
    shapeor = data.shape
    data = data.reshape(-1, data.shape[-1])
    data = StandardScaler().fit_transform(data)
    data = data.reshape(shapeor)
    return data, gt, ignored_labels


def get_X(x, model):
    output, _, _ = model(x)
    w = dict(model.named_parameters())['fc.weight']
    num_classes = output.shape[1]
    ws = []
    w_grad1 = torch.zeros_like(w)
    for i in range(0, num_classes):
        logits_trace = output[0][i]
        w_grad, = torch.autograd.grad(logits_trace, w, retain_graph=True)
        ws.append(w_grad)
        w_grad1 += w_grad
    W = torch.cat(ws, 0)
    W = W[:num_classes]
    W = W.view(num_classes, -1).t()
    return W


def cal_hessian(model, x):
    output, logit, _ = model(x)
    probs = F.softmax(output, 1)
    X = get_X(x, model)
    D = torch.diag(probs[0, :])
    A = (D - probs.t().mm(probs))
    hessian = X.mm(A).mm(X.t())
    return hessian


def compare(A, B):
    if A.shape != B.shape:
        return 0
    w, h = A.size()
    count = 0
    for i in range(w):
        for j in range(h):
            if A[i][j] == 0 and B[i][j] == 1:
                count += 1
    return count


def coverAbyB(A, B):
    w, h = A.size()
    for i in range(w):
        for j in range(h):
            if A[i][j] == 0 and B[i][j] == 1:
                A[i][j] = 1


class CategoryConsistencyLoss(nn.Module):
    def __init__(self, num_classes, embedding_size):
        super(CategoryConsistencyLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.weightcenters = nn.Parameter(torch.normal(0, 1, (num_classes, embedding_size)))

    def forward(self, x, labels):
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        dist_metric = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                      torch.pow(self.weightcenters, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        dist_metric.addmm_(x, self.weightcenters.t(), beta=1, alpha=-2)

        dist = dist_metric[range(batch_size), labels]
        loss = dist.clamp(1e-12, 1e+12).sum() / batch_size
        return loss


ce_criterion = torch.nn.CrossEntropyLoss().to(torch.device('cuda'))
cc_criterion = CategoryConsistencyLoss(num_classes=16, embedding_size=200).to(torch.device('cuda'))


def uniform(A):
    min = A.min()
    max = A.max()
    A = (A - min) / (max - min)
    return A


def myway(x, model, lr, n_iter, ori_gt, path1, path2, alpha, beta, xmin, xmax, grad_threshold=10):
    model.eval()
    device = torch.device('cuda')
    x = torch.from_numpy(x)
    x = x.transpose(2, 0).transpose(2, 1)
    x = x.unsqueeze(0).to(device)
    x.requires_grad = True

    output, features, band_weights = model(x)
    loss1 = ce_criterion(output, ori_gt)
    loss2 = cc_criterion(band_weights, ori_gt.squeeze())
    loss_ori = loss1 + 10.0 * loss2  # The training loss function of the original model.
    w = dict(model.named_parameters())['fc.weight']
    y = torch.autograd.grad(loss_ori, w, create_graph=True)
    ti = torch.sum(y[0] ** 2)
    if ti < grad_threshold:
        return None

    optimizer = torch.optim.Adam(
        params=[x],
        lr=lr,
    )
    maxentropy = 0
    for i in range(n_iter):
        output, logit, band_weights = model(x)
        gt = torch.argmax(output)
        gt = gt.unsqueeze(0)
        loss1 = ce_criterion(output, gt)
        loss2 = cc_criterion(band_weights, gt.squeeze(0))
        loss = loss1 + 10.0 * loss2
        w = dict(model.named_parameters())['fc.weight']
        dl_dw = torch.autograd.grad(loss, w, create_graph=True)
        sensitivity_r1 = torch.sum(dl_dw[0] ** 2)

        hessian = cal_hessian(model, x)
        sensitivity_r2 = torch.sum(hessian ** 2)
        loss_sen = alpha * (1 / sensitivity_r1) + (1 - alpha) * (1 / sensitivity_r2)
        P = F.softmax(output, dim=-1)
        D_P = P.var()
        loss_en = D_P
        loss = beta * loss_sen + loss_en
        loss.backward()
        if sensitivity_r2.detach().cpu().numpy() > 1000:
            entropy = -torch.sum(P * torch.log(P))
            if i == n_iter - 1:
                torch.save(x, path2)
            if entropy > maxentropy:
                torch.save(x, path1)
                maxentropy = entropy
        x.data = torch.clamp(x.data, xmin, xmax)
        optimizer.step()
        with torch.no_grad():
            x.data = torch.clamp(x.data, xmin, xmax)
        optimizer.zero_grad()
        del output, logit, loss, w, dl_dw, hessian


def select_sample_tensor(net, path, threshold_cov=100, threshold_sen=0.1, path_group=None):
    net = net.to(device)
    w = dict(net.named_parameters())['fc.weight']
    c, h = w.size()
    cover = torch.zeros((c, h)).to(device)
    if (path_group == None):
        path_group = os.path.join(path, f"{threshold_cov}_{threshold_sen}")
        if not os.path.exists(path_group):
            os.mkdir(path_group)
    file_list = []
    for root, dirs, files in os.walk(path, topdown=True):
        dirs[:] = [d for d in dirs if not os.path.isdir(os.path.join(root, d))]
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(".pth"):
                file_list.append(file_path)
    random.shuffle(file_list)
    for file_path in file_list:
        img = torch.load(file_path)
        output, _, _ = net(img)
        loss_fn = nn.CrossEntropyLoss()
        gt = torch.argmax(output)
        gt = gt.unsqueeze(0)
        loss = loss_fn(output, gt)
        dl_dw = torch.autograd.grad(loss, w, create_graph=True)
        binary_tensor = (abs(dl_dw[0]) > threshold_sen).float()
        num_cover = compare(cover, binary_tensor)
        if num_cover >= threshold_cov:
            shutil.copy(file_path, path_group)
            coverAbyB(cover, binary_tensor)
        ones_mask = (cover == 1)
        count_ones = torch.sum(ones_mask).item()
        print("total parameters：", c * h, "covered parameters：", count_ones)
        if count_ones >= c * h * 1:
            break
