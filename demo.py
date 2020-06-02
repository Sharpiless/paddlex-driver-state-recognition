import matplotlib
import paddlex as pdx

import paddle.fluid as fluid
import numpy as np
import cv2
import os

import matplotlib.pyplot as plt

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

from facedet import FaceDet

fontC = ImageFont.truetype('./platech.ttf', 20, 0)


def drawText(img, addText, x1, y1):

    color = (20, 255, 20)
    # img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((x1, y1),
              addText.encode("utf-8").decode("utf-8"),
              color, font=fontC)
    imagex = np.array(img)

    return imagex


save_dir = './best_model'
model = pdx.load_model(save_dir)

classes = {'c0': 'normal driving',
           'c1': 'texting-right',
           'c2': 'talking on the phone-right',
           'c3': 'texting-left',
           'c4': 'talking on the phone-left',
           'c5': 'operating the radio',
           'c6': 'drinking',
           'c7': 'reaching behind',
           'c8': 'hair and makeup',
           'c9': 'talking to passenger'}

base = './test_images'

det = FaceDet(thread=0.1)

for im in os.listdir(base):

    pt = os.path.join(base, im)

    result = model.predict(pt)
    print(result)
    lbl = classes[result[0]['category']]+' '+str(result[0]['score'])

    image = cv2.imread(pt)
    image = det.detect(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = drawText(image, lbl, 0, 10)

    plt.imshow(image)
    plt.show()