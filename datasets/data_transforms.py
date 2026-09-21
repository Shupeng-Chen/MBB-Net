import numpy as np
import torch


class NormalizeObjectPose(object):
    def __call__(self, data):
        ptcloud = data['partial_cloud']
        bbox = data['bounding_box']

        # 祖传标尺逻辑 (一字不改，原汁原味)
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                             [np.sin(yaw), np.cos(yaw), 0],
                             [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]

        ptcloud = np.dot(ptcloud - center, rotation) / scale
        # 祖传置换矩阵，保证 MMD = 32.845 的核心
        ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        data['partial_cloud'] = ptcloud
        return data


class RandomSamplePoints(object):
    def __init__(self, n_points=2048):
        self.n_points = n_points

    def __call__(self, data):
        ptcloud = data['partial_cloud']
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:self.n_points]]
        if ptcloud.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])
        data['partial_cloud'] = ptcloud
        return data


class ToTensor(object):
    def __call__(self, data):
        data['partial_cloud'] = torch.from_numpy(data['partial_cloud']).float()
        return data