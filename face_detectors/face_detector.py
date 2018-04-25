import numpy as np


class FaceDetector(object):

    name = None

    def _detect_face(self, img):
        raise NotImplementedError

    def evaluate_against(self, attack, eval_imgs):
        detections = []
        for img in eval_imgs:
            detections.append(self._detect_face(attack(img)))
        return sum(detections) / len(detections)
