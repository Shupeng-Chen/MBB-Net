import torch
import torch.nn as nn
from extensions.chamfer_dist import ChamferDistanceL1

class PCNLoss(nn.Module):
    # �� 修改 1：引入 alpha 参数，默认使用最佳排雷结果 0.4
    def __init__(self, cd_scale=1000.0, use_mbb=True, alpha=0.4):
        super().__init__()
        self.cd_scale = cd_scale
        self.use_mbb = use_mbb
        self.alpha = alpha
        self.cd_l1 = ChamferDistanceL1()
        self.ce = nn.CrossEntropyLoss()

    def _safe_cd(self, pc1, pc2):
        """自适应算子返回格式，防止拆包报错"""
        res = self.cd_l1(pc1.float(), pc2.float())
        if isinstance(res, (list, tuple)):
            # 如果返回的是 (dist1, dist2)
            d1, d2 = res
            cd = (torch.mean(torch.sqrt(d1 + 1e-12), dim=1) +
                  torch.mean(torch.sqrt(d2 + 1e-12), dim=1)) / 2.0
            return cd.mean()
        else:
            # 如果直接返回了计算好的 CD 值
            return res.mean()

    def forward(self, seeds, fine, logits, gt, label):
        # 1. 计算 Fine CD (16384 pts)
        l_fine = self._safe_cd(fine, gt)

        # 2. 计算 Coarse CD (256 pts)
        l_coarse = self._safe_cd(seeds, gt)

        # 3. 组合几何损失
        l_cd = (l_fine + 0.5 * l_coarse) * self.cd_scale

        # 4. 语义损失
        l_ce = torch.tensor(0.0).cuda()
        if self.use_mbb and logits is not None:
            l_ce = self.ce(logits, label)

        # �� 修改 2：使用传入的 alpha 动态权重
        total_loss = l_cd + self.alpha * l_ce
        return total_loss, l_cd