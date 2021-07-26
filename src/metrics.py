import numpy as np
import torch
from sklearn.metrics import confusion_matrix

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

class SiameseAccuracyMetric(Metric):
    """
    Works with SiameseNet model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0
        self.threshold_dist = 0.1
        self.cm = np.zeros((2, 2))

    def normalize(self, vector):
        scale = 1.0 / (vector.max(dim=0, keepdim=True)[0] - vector.min(dim=0, keepdim=True)[0]) 
        vector.mul_(scale).sub_(vector.min(dim=0, keepdim=True)[0])
        return vector

    def mysigmoid(self, x):
        return 1 + (-1 / (1 + torch.exp(-x)))

    def __call__(self, outputs, target, loss):
        distances = (outputs[0].data - outputs[1].data).pow(2).sum(1)  # squared distances
        normalized_distances = self.normalize(distances).reshape((-1, 1))
        pred = torch.round(self.mysigmoid(normalized_distances - self.threshold_dist))
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        # confusion matrix calculation
        self.cm += confusion_matrix(target[0].data.view_as(pred).cpu(), pred.cpu())
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'

    def confusion_matrix(self):
        return self.cm
