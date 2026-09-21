import torch


def compute_squared_dists(xyz1, xyz2, chunk_size=2048):
    """纯 PyTorch 分块计算平方距离 (防 OOM, 完全替代 C++ 算子)"""
    B, N, _ = xyz1.shape
    _, M, _ = xyz2.shape
    dist1 = torch.zeros(B, N, device=xyz1.device)
    dist2 = torch.zeros(B, M, device=xyz2.device)

    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        d = torch.sum((xyz1[:, i:end].unsqueeze(2) - xyz2.unsqueeze(1)) ** 2, dim=-1)
        dist1[:, i:end] = torch.min(d, dim=2)[0]

    for i in range(0, M, chunk_size):
        end = min(i + chunk_size, M)
        d = torch.sum((xyz2[:, i:end].unsqueeze(2) - xyz1.unsqueeze(1)) ** 2, dim=-1)
        dist2[:, i:end] = torch.min(d, dim=2)[0]

    return dist1, dist2


def calc_pcn_cd(dist1, dist2, scale=1000.0):
    """
    学术标准 PCN L1 Chamfer Distance
    注意：必须加 1e-12 防止 sqrt(0) 产生的 NaN 溢出
    """
    cd1 = torch.mean(torch.sqrt(dist1 + 1e-12), dim=1)
    cd2 = torch.mean(torch.sqrt(dist2 + 1e-12), dim=1)
    return ((cd1 + cd2) / 2.0) * scale


def calc_fscore(dist1, dist2, threshold=0.01):
    """
    学术标准 F-Score @ 1%
    必须将平方距离转换为真实 Euclidean 距离后，再与 0.01 比较
    """
    d1 = torch.sqrt(dist1 + 1e-12)
    d2 = torch.sqrt(dist2 + 1e-12)

    precision = torch.mean((d1 < threshold).float(), dim=1)
    recall = torch.mean((d2 < threshold).float(), dim=1)

    # 同样加上 1e-8 防止分母为 0
    fscore = 2 * (precision * recall) / (precision + recall + 1e-8)
    return fscore


def calc_acc(logits, labels):
    """计算分类准确率"""
    if logits is None:
        return 0.0
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


class AverageMeter(object):
    def __init__(self): self.reset()

    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0

    def update(self, val, n=1): self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count