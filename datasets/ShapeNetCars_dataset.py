import os
import json
import torch
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class ShapeNetCarsDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        # 使用 Easydict 传进来的参数
        self.dataset_root = Path(config.DATA_ROOT)
        self.subset = config.subset
        self.npoints = config.get('N_POINTS', 16384)

        self.file_list = []
        self.pcn_json_path = self.dataset_root / "PCN.json"

        if not self.pcn_json_path.exists():
            raise FileNotFoundError(f"Missing {self.pcn_json_path}. Please check your PCN dataset path.")

        with open(self.pcn_json_path, 'r', encoding='utf-8') as f:
            category_json = json.load(f)

        for cat in category_json:
            # 02958343 是 ShapeNet 中 Car (汽车) 的专属 ID
            if cat['taxonomy_id'] != '02958343':
                continue

            folder = 'test' if self.subset == 'test' else self.subset
            if folder in cat:
                for model_id in cat[folder]:
                    # 严格按照你原来 PCN_dataset 的路径拼写逻辑
                    c_path = self.dataset_root / folder / "complete" / cat['taxonomy_id'] / f"{model_id}.pcd"
                    self.file_list.append({
                        'taxonomy_id': cat['taxonomy_id'],
                        'model_id': model_id,
                        'gt_path': str(c_path)
                    })

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        # 读取完整车辆 (Ground Truth)
        if not os.path.exists(sample['gt_path']):
            raise FileNotFoundError(f"Missing PCD: {sample['gt_path']}")

        pcd = o3d.io.read_point_cloud(sample['gt_path'])
        pts = np.asarray(pcd.points).astype(np.float32)

        # 固定采样到 16384 个点
        if len(pts) > self.npoints:
            choice = np.random.choice(len(pts), self.npoints, replace=False)
            pts = pts[choice, :]
        elif len(pts) < self.npoints and len(pts) > 0:
            choice = np.random.choice(len(pts), self.npoints, replace=True)
            pts = pts[choice, :]

        gt_tensor = torch.from_numpy(pts)

        # �� 核心：为了兼容 metric.py 里 dataset[i][-1][1] 的解包格式，
        # 我们返回: taxonomy_id, model_id, (dummy_partial, gt_tensor)
        dummy_partial = torch.zeros_like(gt_tensor)

        return sample['taxonomy_id'], sample['model_id'], (dummy_partial, gt_tensor)