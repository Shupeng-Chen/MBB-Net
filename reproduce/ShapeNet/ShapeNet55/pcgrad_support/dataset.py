import os
import torch
import numpy as np
import torch.utils.data as data


class ShapeNetDataset(data.Dataset):
    def __init__(self, cfg, subset='train'):
        super().__init__()
        self.subset = subset
        self.data_root = cfg['DATA_ROOT']
        self.pc_path = os.path.join(self.data_root, 'shapenet_pc')

        self.npoints = cfg.get('NUM_PARTIAL_POINTS', 2048)
        self.gt_npoints = cfg.get('NUM_GT_POINTS', 8192)  # PoinTr 标准

        self.classes = [
            '02691156', '02747177', '02773838', '02801938', '02808440', '02818832',
            '02828884', '02843684', '02871439', '02876657', '02880940', '02924116',
            '02933112', '02942699', '02946921', '02954340', '02958343', '02992529',
            '03001627', '03046257', '03085013', '03207941', '03211117', '03261776',
            '03325088', '03337140', '03467517', '03513137', '03593526', '03624134',
            '03636649', '03642806', '03691459', '03710193', '03759954', '03761084',
            '03790512', '03797390', '03928116', '03938244', '03948459', '03991062',
            '04004475', '04074963', '04090263', '04099429', '04225987', '04256520',
            '04330267', '04379243', '04401088', '04460130', '04468005', '04530566', '04554684'
        ]

        num_classes_cfg = cfg.get('NUM_CLASSES', 55)
        if num_classes_cfg == 34:
            self.classes = self.classes[:34]
        elif num_classes_cfg == 21:
            self.classes = self.classes[34:]

        self.cat2label = {cat: i for i, cat in enumerate(self.classes)}

        # 严格按照 PoinTr 标准，仅使用 train.txt 和 test.txt，无 val.txt
        split_file = os.path.join(self.data_root, f'{subset}.txt')
        with open(split_file, 'r') as f:
            self.file_list = [l.strip().replace('.npy', '') for l in f.readlines() if l.strip()]

        self.file_list = [f for f in self.file_list if f.split('-')[0] in self.cat2label]

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

        gt_pc = np.load(os.path.join(self.pc_path, model_id + '.npy')).astype(np.float32)
        gt_pc = self.pc_norm(gt_pc)
        gt_pc = self.random_sample(gt_pc, self.gt_npoints)

        label = self.cat2label.get(cat_id, 0)

        if self.subset == 'train':
            vp = np.random.randn(3)
            vp /= (np.linalg.norm(vp) + 1e-8)

            n_remove = np.random.randint(2048, 6144)
            partial_pc = self.crop_pc(gt_pc, vp, n_remove)
            partial_pc = self.random_sample(partial_pc, self.npoints)

            # 修正 1：严格对齐 PoinTr 的 Z 轴旋转
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
            vp = self.fixed_viewpoints[idx % 8]
            res = {}
            # 修正 2：测试难度命名严格对齐 PoinTr (simple, moderate, hard)
            for k, n_remove in zip(['simple', 'moderate', 'hard'], [2048, 4096, 6144]):
                p = self.crop_pc(gt_pc, vp, n_remove)
                res[k] = torch.from_numpy(self.random_sample(p, self.npoints)).float()

            return res, \
                torch.from_numpy(gt_pc).float(), \
                torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.file_list)