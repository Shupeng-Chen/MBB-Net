import torch

# =====================================================================
# 放弃极其不稳定的 C++ 拓展算子格式
# 直接使用极度安全的纯 PyTorch 分块算子，保证 100% 拿到 dist1 和 dist2
# =====================================================================
def compute_squared_dists_chunked(xyz1, xyz2, chunk_size=256):
    """纯 PyTorch 分块计算平方距离 (防 OOM，且稳定返回 d1, d2)"""
    B, N, _ = xyz1.shape
    _, M, _ = xyz2.shape
    dist1 = torch.zeros(B, N, device=xyz1.device)
    dist2 = torch.zeros(B, M, device=xyz2.device)

    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        diff = xyz1[:, i:end].unsqueeze(2) - xyz2.unsqueeze(1)
        d = torch.sum(diff ** 2, dim=-1)
        dist1[:, i:end] = torch.min(d, dim=2)[0]
        del diff, d

    for i in range(0, M, chunk_size):
        end = min(i + chunk_size, M)
        diff = xyz2[:, i:end].unsqueeze(2) - xyz1.unsqueeze(1)
        d = torch.sum(diff ** 2, dim=-1)
        dist2[:, i:end] = torch.min(d, dim=2)[0]
        del diff, d

    return dist1, dist2

def calc_shapenet_metrics(pred, gt):
    """计算 ShapeNet 的 CD-L2 和 F-Score"""
    pred = pred.contiguous().float()
    gt = gt.contiguous().float()

    with torch.no_grad():
        # 直接使用自带的绝对安全算子
        dist1, dist2 = compute_squared_dists_chunked(pred, gt, chunk_size=256)

        # 1. 计算 ShapeNet CD-L2 (* 1000)
        cd_val = (torch.mean(dist1) + torch.mean(dist2)) * 1000.0

        # 2. 计算 F-Score @ 1% (需要换算回真实 Euclidean 距离比较)
        d1 = torch.sqrt(dist1 + 1e-12)
        d2 = torch.sqrt(dist2 + 1e-12)

        precision = torch.mean((d1 < 0.01).float(), dim=1)
        recall = torch.mean((d2 < 0.01).float(), dim=1)

        fscore = 2 * (precision * recall) / (precision + recall + 1e-8)

        return cd_val.item(), fscore.mean().item()

def calc_acc(logits, labels):
    """计算分类准确率"""
    if logits is None:
        return 0.0
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

class AverageMeter(object):
    """记录和计算平均值的工具类"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count