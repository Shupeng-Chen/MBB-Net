import torch
import numpy as np
import json
import random
import os
from pathlib import Path
import open3d as o3d
from torch.utils.data import Dataset
from tqdm import tqdm


class PCNDataset(Dataset):
    def __init__(self, config, subset='train'):
        self.config = config
        self.subset = subset

        # �� [核心防线]：环境自适应路径漂移修复 (开源友好)
        # 绝对不影响任何加载逻辑、数据增强或网络结构，仅做字符串替换
        primary_path = Path(config['DATASET_ROOT'])
        fallback_path = Path(__file__).resolve().parents[1] / "data" / "PCN" / "ShapeNetCompletion"

        if primary_path.exists():
            self.dataset_root = primary_path
        elif fallback_path.exists():
            if subset == 'test':  # 只在测试时打印一次警告，防止训练时刷屏
                print(f"\n⚠️ [环境自适应] 检测到 YAML 配置路径 ({primary_path}) 失效。")
                print(f"   已自动重定向至物理备用路径: {fallback_path}\n")
            self.dataset_root = fallback_path
        else:
            self.dataset_root = primary_path  # 如果两个都找不到，让它自然报错

        self.pcn_json_path = self.dataset_root / "PCN.json"

        self.CLASS_MAPPING = {'02691156': 0, '02933112': 1, '02958343': 2, '03001627': 3,
                              '03636649': 4, '04256520': 5, '04379243': 6, '04530566': 7}

        self.samples = self._load_metadata()
        self.preload = config.get('PRELOAD', False)

        if self.preload and subset == 'train':
            print(f"�� [Academic Mode] 正在将 {subset} 集 8 视角预加载至内存...")
            self.pc_data = []
            for s in tqdm(self.samples):
                try:
                    p_pts_list = [self._load_raw_pcd(p) for p in s['partial_paths']]
                    gt_pts = self._load_raw_pcd(s['complete_path'])
                    self.pc_data.append((p_pts_list, gt_pts, s['label']))
                except Exception:
                    continue
            print(f"✅ 预加载完成！有效样本：{len(self.pc_data)}")
        else:
            self.preload = False

    def _load_metadata(self):
        with open(self.pcn_json_path, 'r', encoding='utf-8') as f:
            pcn_data = json.load(f)
        samples = []
        for cat in pcn_data:
            tax_id = cat['taxonomy_id']
            if tax_id not in self.CLASS_MAPPING: continue
            folder = 'test' if self.subset == 'test' else self.subset
            label = self.CLASS_MAPPING[tax_id]
            for model_id in cat[folder]:
                p_paths = [self.dataset_root / folder / "partial" / tax_id / model_id / f"{i:02d}.pcd" for i in
                           range(8)]
                c_path = self.dataset_root / folder / "complete" / tax_id / f"{model_id}.pcd"
                samples.append(
                    {'label': label, 'partial_paths': [str(p) for p in p_paths], 'complete_path': str(c_path)})
        return samples

    def _load_raw_pcd(self, path):
        if not os.path.exists(path): raise FileNotFoundError(f"Missing: {path}")
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points).astype(np.float32)
        if len(pts) == 0: raise ValueError(f"Empty PCD: {path}")
        return pts

    def _sample_points(self, pts, n):
        idx = np.random.choice(len(pts), n, replace=(len(pts) < n))
        return pts[idx]

    def __len__(self):
        return len(self.pc_data) if self.preload else len(self.samples)

    def __getitem__(self, idx):
        sample_seed = idx + torch.initial_seed() % (2 ** 32)
        random.seed(sample_seed)
        np.random.seed(sample_seed)

        if self.preload:
            p_list, gt_raw, label = self.pc_data[idx]
            view = np.random.randint(0, 8)
            partial = self._sample_points(p_list[view].copy(), self.config['NUM_PARTIAL_POINTS'])
            gt = self._sample_points(gt_raw.copy(), self.config['NUM_COMPLETE_POINTS'])
        else:
            s = self.samples[idx]
            view = np.random.randint(0, 8) if self.subset == 'train' else 0
            partial = self._sample_points(self._load_raw_pcd(s['partial_paths'][view]),
                                          self.config['NUM_PARTIAL_POINTS'])
            gt = self._sample_points(self._load_raw_pcd(s['complete_path']), self.config['NUM_COMPLETE_POINTS'])
            label = s['label']

        # �� 数据增强同时作用于 partial 和 gt
        if self.config.get('AUGMENT', False) and self.subset == 'train':
            if random.random() < 0.5:
                partial[:, 0], gt[:, 0] = -partial[:, 0], -gt[:, 0]
            angle = random.uniform(-0.1, 0.1)
            c, s_val = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s_val, 0], [s_val, c, 0], [0, 0, 1]], dtype=np.float32)
            partial, gt = partial @ R, gt @ R

        return torch.from_numpy(partial), torch.from_numpy(gt), torch.tensor(label)