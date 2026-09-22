import torch
import torch.nn as nn

# 尝试导入你的 C++ 算子。如果有 L2 就用 L2，如果没有，默认的 ChamferDistance 就是 L2
try:
    from extensions.chamfer_dist import ChamferDistanceL2
except ImportError:
    from extensions.chamfer_dist import ChamferDistance as ChamferDistanceL2


class ShapeNetLoss(nn.Module):
    def __init__(self, alpha=0.4, cd_scale=1.0):
        super().__init__()
        self.alpha = alpha
        self.cd_scale = cd_scale
        self.cd_l2 = ChamferDistanceL2()  # 调用 C++ 极速算子
        self.ce = nn.CrossEntropyLoss()

    def _safe_cd_l2(self, pc1, pc2):
        """安全的 C++ CD-L2 计算，彻底告别 OOM"""
        # 必须 contiguous()，否则 CUDA 拓展容易报错
        res = self.cd_l2(pc1.contiguous().float(), pc2.contiguous().float())
        if isinstance(res, (list, tuple)):
            d1, d2 = res
            # ShapeNet 标准 CD-L2: 不要 sqrt!
            cd = torch.mean(d1, dim=1) + torch.mean(d2, dim=1)
            return cd.mean()
        else:
            return res.mean()

    def forward(self, out_list, logits, gt, label):
        loss_comp = torch.tensor(0.0).cuda()

        # SnowflakeNet 核心：对多级输出全部施加 C++ CD 监督
        for pc in out_list:
            loss_comp += self._safe_cd_l2(pc, gt)

        loss_comp = loss_comp * self.cd_scale

        loss_ce = torch.tensor(0.0).cuda()
        if logits is not None:
            loss_ce = self.ce(logits, label)

        total_loss = loss_comp + self.alpha * loss_ce

        return total_loss, loss_comp, loss_ce