import sys
import os
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '../../..'))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

# 核心魔法：直接继承 ShapeNet55 的 Trainer
# 因为点云补全的 CD Loss 和 F1 评测指标对于 55 类和 34 类在数学上是完全一致的
from reproduce.ShapeNet.ShapeNet34_21.support.shapenet55_trainer import ShapeNet55Trainer


class ShapeNet34Trainer(ShapeNet55Trainer):
    def __init__(self, model, train_loader, test_loader, cfg):
        """
        继承 55 的 Trainer，所有优化器、Loss 计算、日志记录直接复用，
        保持代码的极简与绝对安全。
        """
        super().__init__(model, train_loader, test_loader, cfg)

    # 如果未来你想给 34 加特殊的 log 逻辑，可以直接在这里重写方法
    # 目前直接沿用父类即可