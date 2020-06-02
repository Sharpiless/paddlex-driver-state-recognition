import numpy as np
import sys
import cv2
import os
from yolo3tiny.detection import Detector

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

from yolo3tiny.imageSolver import draw_bbox, recover_img

fontC = ImageFont.truetype('platech.ttf', 16, 0)


class FaceDet(object):
    '''
    人脸检测+识别
    需要识别的脸放到那个faces文件夹里面
    '''

    def __init__(self, thread=0.5):

        self.input_size = 416
        self.face_det = Detector(USE_CUDA=True)

        self.thread = thread
        self.min_size = 400

    def scale_img(self, img, IMG_WH=416):
        h, w, c = img.shape
        dh, dh_e, dw, dw_e = 0, 0, 0, 0
        if w > h:
            dh = (w-h)//2
            dh_e = w-h-dh-dh
        else:
            dw = (h-w)//2
            dw_e = h-w-dw-dw
        img = cv2.copyMakeBorder(img, dh, dh+dh_e, dw,
                                 dw+dw_e, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        imgArray = cv2.resize(img, (IMG_WH, IMG_WH))/255.0
        return np.array(imgArray, dtype='float32').swapaxes(0, 2).swapaxes(1, 2)

    def detect(self, im):

        font = cv2.FONT_HERSHEY_DUPLEX

        im = self.scale_img(im, 416)

        bboxes_pre = self.face_det.detect(
            im, confidence_threshold=self.thread, nms_threshold=0.3)[0]
        
        self.frame = recover_img(im)

        if bboxes_pre.shape[0]:

            bboxes_pre = bboxes_pre[:, :4]
            w = bboxes_pre[:, 2] - bboxes_pre[:, 0]
            h = bboxes_pre[:, 3] - bboxes_pre[:, 1]
            face_loc = [v for v in list(bboxes_pre) if (
                v[2]-v[0])*(v[3]-v[1]) > self.min_size]

            self.frame = draw_bbox(self.frame, bboxes_pre)

        return self.frame


if __name__ == '__main__':
    
    det = FaceDet(thread=0.1)
    base = './test_images'

    for im in os.listdir(base):
        pt = os.path.join(base, im)
        im = cv2.imread(pt)
        result = det.detect(im)
        cv2.imshow('result', result)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
