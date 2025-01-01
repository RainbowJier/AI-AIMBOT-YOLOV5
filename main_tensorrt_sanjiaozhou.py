import ctypes
import os
import time
from math import atan

import cupy as cp
import numpy as np
import pandas as pd
import torch
import win32api
import win32con

import gameSelection
from config_sjz import aaQuitKey, confidence, cpsDisplay, \
    visuals, centerOfScreen, aim_range
from models.common import DetectMultiBackend
from utils.general import (cv2, non_max_suppression, xyxy2xywh)

# 加载罗技驱动dll文件
try:
    root = os.path.abspath(os.path.dirname(__file__))
    driver = ctypes.CDLL(f'{root}/logitech.driver.dll')
    ok = driver.device_open() == 1  # 该驱动每个进程可打开一个实例
    if not ok:
        print('Error, GHUB or LGS driver not found')
except FileNotFoundError:
    print(f'Error, DLL file not found')


class Logitech:
    class mouse:

        @staticmethod
        def mouse_press(code):
            """ 鼠标按下 code: 1左 2中 3右 """
            if not ok:
                return
            driver.mouse_down(code)
            print("按下左键")

        @staticmethod
        def mouse_up(code):
            """ 鼠标松开 code: 左 中 右 """
            if not ok:
                return
            driver.mouse_up(code)
            print("松开左键")

        @staticmethod
        def move(x, y):
            if not ok:
                return
            if x == 0 and y == 0:
                return
            driver.moveR(x, y, True)


def main():
    # External Function for running the game selection menu (gameSelection.py)
    camera, cWidth, cHeight, region = gameSelection.gameSelection()

    # 识别的人物
    center_screen = [cWidth, cHeight]

    # Used for forcing garbage collection
    count = 0
    sTime = time.time()

    # Loading Yolo5 Small AI Model
    model = DetectMultiBackend('weights/sanjiaozhou/v2/320-FP16/sjz_yolov5.engine', device=torch.device(
        'cuda'), dnn=False, data='', fp16=True)
    stride, names, pt = model.stride, model.names, model.pt

    with torch.no_grad():
        while win32api.GetAsyncKeyState(ord(aaQuitKey)) == 0:
            npImg = cp.array([camera.get_latest_frame()])
            if npImg.shape[3] == 4:
                # If the image has an alpha channel, remove it
                npImg = npImg[:, :, :, :3]

            """
            获取检测到的目标数据
            """
            targets = detection(npImg, model)

            """
            移动鼠标
            """
            move_Mouse(targets, center_screen, camera)

            """
            Draw frame.
            """
            # See what the bot sees
            if visuals:
                npImg = cp.asnumpy(npImg[0])
                # Loops over every item identified and draws a bounding box
                for i in range(0, len(targets)):
                    halfW = round(targets["width"][i] / 2)
                    halfH = round(targets["height"][i] / 2)
                    midX = targets['current_mid_x'][i]
                    midY = targets['current_mid_y'][i]
                    (startX, startY, endX, endY) = int(
                        midX + halfW), int(midY + halfH), int(midX - halfW), int(midY - halfH)

                    # draw the bounding box and label on the frame
                    if targets["class"][0] in (0, 2):
                        label = "{}: {:.2f}%".format(
                            "Body", targets["confidence"][i] * 100)

                    else:
                        label = "{}: {:.2f}%".format(
                            "Head", targets["confidence"][i] * 100)

                    cv2.rectangle(npImg, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(npImg, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            """
            Forced garbage cleanup every second
            """
            count += 1
            if (time.time() - sTime) > 1:
                if cpsDisplay:
                    print("CPS: {}".format(count))
                count = 0
                sTime = time.time()

            # Uncomment if you keep running into memory issues
            # gc.collect(generation=0)

            # See visually what the Aimbot sees
            if visuals:
                cv2.imshow('Live Feed', npImg)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    exit()
    camera.stop()


def detection(npImg, model):
    im = npImg / 255
    im = im.astype(cp.half)

    im = cp.moveaxis(im, 3, 1)
    im = torch.from_numpy(cp.asnumpy(im)).to('cuda')

    # Detecting all the objects
    results = model(im)

    pred = non_max_suppression(
        results, confidence, confidence, [0, 1, 2], False, max_det=1)

    targets = []
    for i, det in enumerate(pred):
        gn = torch.tensor(im.shape)[[0, 0, 0, 0]]
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                if int(cls.item()) == 0 or int(cls.item()) == 1:
                    targets.append((xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() +
                                   [float(conf), int(cls)])  # normalized xywh

    targets = pd.DataFrame(
        targets, columns=['current_mid_x', 'current_mid_y', 'width', "height", "confidence", "class"])
    return targets


def move_Mouse(targets, center_screen, camera):
    """
    获取目标数据（坐标，高度）
    Returns:

    """
    # If there are people in the center bounding box
    if len(targets) > 0:
        if (centerOfScreen):
            # Compute the distance from the center
            """
            （current_mid_x,current_mid_y)：检测到方框的中心点
            """
            targets["dist_from_center"] = np.sqrt((targets.current_mid_x - center_screen[0]) ** 2 + (
                    targets.current_mid_y - center_screen[1]) ** 2)

            # Sort the data frame by distance from center
            targets = targets.sort_values("dist_from_center")

        # The center of the body
        box_xMid = targets.iloc[0].current_mid_x
        box_yMid = targets.iloc[0].current_mid_y

        box_height = targets.iloc[0].height
        # 瞄准头部
        headshot_offset = box_height

        # 最终移动的坐标
        mouseMove = [box_xMid - center_screen[0], box_yMid - center_screen[1]]

        headshot_offset = targets.iloc[0].height

        # Targets
        target_x = box_xMid
        target_y = box_yMid - headshot_offset
        if win32api.GetKeyState(0x14):
            # Logitech.mouse.move(int(mouseMove[0]), int(mouseMove[1]))
            if (targets["dist_from_center"][0] <= aim_range):
                if (win32api.GetKeyState(win32con.VK_LBUTTON) < 0):
                    capture_screen(camera)
                else:
                    Logitech.mouse.move(int(mouseMove[0]), int(mouseMove[1]))

def capture_screen(camera):
    # while pressing left button
    if (win32api.GetKeyState(win32con.VK_LBUTTON) < 0):
        frame = camera.get_latest_frame()
        from PIL import Image
        pil_img = Image.fromarray(frame)
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        pil_img.save('C:\\Users\\30218\Desktop\SJZ\\train/' + str(timestamp) + '.png')


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        traceback.print_exception(e)
        print("ERROR: " + str(e))
