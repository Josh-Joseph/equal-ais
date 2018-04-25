import numpy as np
import cv2
from .face_detector import FaceDetector


_face_cascade_name = './face_detectors/haarcascade_frontalface_default.xml'


class CV2CascadeFaceDetector(FaceDetector):

    name = 'cv2_haar_cascade_face_detector'

    def __init__(self):
        self._face_cascade = cv2.CascadeClassifier()
        self._face_cascade.load(_face_cascade_name)

    def _detect_face(self, img):
        return len(self._face_cascade.detectMultiScale(np.asarray(img),
                                                       1.1, 2, 0|cv2.CASCADE_SCALE_IMAGE, (5, 5))) > 0
