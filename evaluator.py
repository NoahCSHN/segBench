import numpy as np

# ================= 辅助函数: 计算 IoU =================
class Evaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        label = self.num_classes * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_classes**2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def evaluate(self):
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusion_matrix) # TP
        union = (
            np.sum(self.confusion_matrix, axis=1) + 
            np.sum(self.confusion_matrix, axis=0) - 
            np.diag(self.confusion_matrix)
        )
        # 避免除以 0
        iou = intersection / (union + 1e-10)
        return iou
