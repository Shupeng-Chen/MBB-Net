import os
import torch
import numpy as np
import torch.utils.data as data


class ShapeNet34Dataset(data.Dataset):
    def __init__(self, cfg, subset='train'):
        super().__init__()
        self.subset = subset

        # 1. 路径配置 (点云实体文件路径 + TXT列表路径)
        self.pc_path = os.path.join(cfg['DATA_ROOT'], 'shapenet_pc')
        self.split_root = cfg['SPLIT_ROOT']  # 指向你的 my_data 目录
        self.dataset_type = str(cfg.get('DATASET_TYPE', '34'))  # '34' 或 '21'

        self.npoints = cfg.get('NUM_PARTIAL_POINTS', 2048)
        self.gt_npoints = cfg.get('NUM_GT_POINTS', 8192)

        # 2. 精准定位官方 TXT 划分文件
        if self.dataset_type == '21':
            assert subset == 'test', "[!] ShapeNet-Unseen21 only has evaluation test set!"
            split_file = os.path.join(self.split_root, 'ShapeNet-Unseen21', 'test.txt')
        else:
            split_file = os.path.join(self.split_root, 'ShapeNet-34', f'{subset}.txt')

        # 3. 读取文件列表
        with open(split_file, 'r') as f:
            self.file_list = [l.strip().replace('.npy', '') for l in f.readlines() if l.strip()]

        # 4. 动态构建 34 类的 label 映射 (为了与 PoinTr 的 34分类严格对齐)
        # 我们从 34 的 train.txt 中提取唯一的 34 个类，而不是写死
        train_34_file = os.path.join(self.split_root, 'ShapeNet-34', 'train.txt')
        with open(train_34_file, 'r') as f:
            train_34_list = [l.strip().replace('.npy', '') for l in f.readlines() if l.strip()]
        unique_classes_34 = sorted(list(set([f.split('-')[0] for f in train_34_list])))
        self.cat2label = {cat: i for i, cat in enumerate(unique_classes_34)}

        # 5. 测试时的固定 8 视角 (严格遵守学术标准)
        self.fixed_viewpoints = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ], dtype=np.float32)
        self.fixed_viewpoints /= (np.linalg.norm(self.fixed_viewpoints, axis=1, keepdims=True) + 1e-8)

    def pc_norm(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        return pc / (m + 1e-9)

    def crop_pc(self, pc, viewpoint, n_remove):
        vp_pos = viewpoint * 2.0
        dist = np.sum((pc - vp_pos) ** 2, axis=1)
        idx = np.argsort(dist)
        keep_idx = idx[:pc.shape[0] - n_remove]
        return pc[keep_idx]

    def random_sample(self, pc, n):
        idx = np.random.permutation(pc.shape[0])
        if pc.shape[0] < n:
            res_idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n - pc.shape[0])])
        else:
            res_idx = idx[:n]
        return pc[res_idx]

    def __getitem__(self, idx):
        model_id = self.file_list[idx]
        cat_id = model_id.split('-')[0]

        # 读取点云
        gt_pc = np.load(os.path.join(self.pc_path, model_id + '.npy')).astype(np.float32)
        gt_pc = self.pc_norm(gt_pc)
        gt_pc = self.random_sample(gt_pc, self.gt_npoints)

        # 获取类别标签 (如果是 Unseen21, 它不在 34 的字典里，默认给 0，反正也不测 Acc)
        label = self.cat2label.get(cat_id, 0)

        if self.subset == 'train':
            # 训练阶段：随机截断 25%~75%
            vp = np.random.randn(3)
            vp /= (np.linalg.norm(vp) + 1e-8)

            n_remove = np.random.randint(2048, 6144)
            partial_pc = self.crop_pc(gt_pc, vp, n_remove)
            partial_pc = self.random_sample(partial_pc, self.npoints)

            # 随机 Z 轴旋转 (增强泛化性)
            theta = np.random.uniform(0, 2 * np.pi)
            rot_mat = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ], dtype=np.float32)

            gt_pc = gt_pc @ rot_mat
            partial_pc = partial_pc @ rot_mat

            return torch.from_numpy(partial_pc).float(), \
                torch.from_numpy(gt_pc).float(), \
                torch.tensor(label, dtype=torch.long)
        else:
            # 测试阶段：使用 8 个固定视角生成 simple, moderate, hard
            vp = self.fixed_viewpoints[idx % 8]
            res = {}
            for k, n_remove in zip(['simple', 'moderate', 'hard'], [2048, 4096, 6144]):
                p = self.crop_pc(gt_pc, vp, n_remove)
                res[k] = torch.from_numpy(self.random_sample(p, self.npoints)).float()

            return res, \
                torch.from_numpy(gt_pc).float(), \
                torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.file_list)