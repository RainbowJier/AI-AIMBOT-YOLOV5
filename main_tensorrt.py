import ctypes
import os
import time

import cupy as cp
import numpy as np
import pandas as pd
import torch
import win32api

import gameSelection
from config import aaMovementAmp, useMask, maskHeight, maskWidth, aaQuitKey, confidence, headshot_mode, cpsDisplay, \
    visuals, centerOfScreen, maskSide
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
        def move(x, y):
            if not ok:
                return
            if x == 0 and y == 0:
                return
            driver.moveR(x, y, True)


def main():
    # External Function for running the game selection menu (gameSelection.py)
    camera, cWidth, cHeight = gameSelection.gameSelection()

    # 自瞄的范围
    center_screen = [cWidth, cHeight]

    # Used for forcing garbage collection
    count = 0
    sTime = time.time()

    # Loading Yolo5 Small AI Model
    model = DetectMultiBackend('weights/10w/10w-v5.engine', device=torch.device(
        'cuda'), dnn=False, data='', fp16=True)
    stride, names, pt = model.stride, model.names, model.pt

    # Main loop Quit if exit key is pressed
    last_mid_coord = None
    with torch.no_grad():
        while win32api.GetAsyncKeyState(ord(aaQuitKey)) == 0:

            npImg = cp.array([camera.get_latest_frame()])
            if npImg.shape[3] == 4:
                # If the image has an alpha channel, remove it
                npImg = npImg[:, :, :, :3]

            """
            # Use "left" or "right" for the mask side depending on where the interfering object is, useful for 3rd player models or large guns
            """
            Mask(useMask, maskSide, npImg)

            """
            获取检测到的目标数据
            """
            targets = detection(npImg, model, names)

            """
            移动鼠标
            """
            move_Mouse(targets, center_screen, last_mid_coord, cWidth, cHeight)

            """
            绘制方框
            """
            drawframe(npImg, targets)

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


def Mask(useMask, maskSide, npImg):
    if useMask:
        maskSide = maskSide.lower()
        if maskSide == "right":
            npImg[:, -maskHeight:, -maskWidth:, :] = 0
        elif maskSide == "left":
            npImg[:, -maskHeight:, :maskWidth, :] = 0
        else:
            raise Exception('ERROR: Invalid maskSide! Please use "left" or "right"')


def detection(npImg, model, names):
    """
    获取检测到目标的数据（坐标，高度）
    Args:
        npImg:
        model:
        names:

    Returns:

    """
    im = npImg / 255
    im = im.astype(cp.half)

    im = cp.moveaxis(im, 3, 1)
    im = torch.from_numpy(cp.asnumpy(im)).to('cuda')

    # Detecting all the objects
    results = model(im)

    pred = non_max_suppression(
        results, confidence, confidence, [0, 1, 2, 3], False, max_det=2)

    targets = []
    for i, det in enumerate(pred):
        s = ""
        gn = torch.tensor(im.shape)[[0, 0, 0, 0]]
        if len(det):
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}, "  # add to string

            for *xyxy, conf, cls in reversed(det):
                targets.append((xyxy2xywh(torch.tensor(xyxy).view(
                    1, 4)) / gn).view(-1).tolist() + [float(conf)])  # normalized xywh

    targets = pd.DataFrame(
        targets, columns=['current_mid_x', 'current_mid_y', 'width', "height", "confidence"])

    return targets


def move_Mouse(targets, center_screen, last_mid_coord, cWidth, cHeight):
    """
    获取目标数据（坐标，高度）
    Returns:

    """
    # If there are people in the center bounding box
    if len(targets) > 0:
        if (centerOfScreen):
            # Compute the distance from the center
            targets["dist_from_center"] = np.sqrt((targets.current_mid_x - center_screen[0]) ** 2 + (
                    targets.current_mid_y - center_screen[1]) ** 2)

            # Sort the data frame by distance from center
            targets = targets.sort_values("dist_from_center")

        # Get the last persons mid coordinate if it exists
        if last_mid_coord:
            targets['last_mid_x'] = last_mid_coord[0]
            targets['last_mid_y'] = last_mid_coord[1]
            # Take distance between current person mid coordinate and last person mid coordinate
            targets['dist'] = np.linalg.norm(
                targets.iloc[:, [0, 1]].values - targets.iloc[:, [4, 5]], axis=1)
            targets.sort_values(by="dist", ascending=False)

        # Take the first person that shows up in the dataframe (Recall that we sort based on Euclidean distance)
        xMid = targets.iloc[0].current_mid_x
        yMid = targets.iloc[0].current_mid_y

        box_height = targets.iloc[0].height
        if headshot_mode:
            headshot_offset = box_height * 0.4
        else:
            headshot_offset = box_height * 0.2

        mouseMove = [xMid - cWidth, (yMid - headshot_offset) - cHeight]

        # 修改开启自瞄开关
        if win32api.GetKeyState(0x14):
            # 定义起始值、结束值和步长
            # 构建递减序列的列表
            for number in [aaMovementAmp - 0.05 * i for i in range(int((0.5 - 0.1) / 0.05) + 1)]:
                # 分多次移动可一定程度解决超调问题
                Logitech.mouse.move(int(mouseMove[0] * number), int(mouseMove[1] * number))

        last_mid_coord = [xMid, yMid]

    else:
        last_mid_coord = None


def drawframe(npImg, targets):
    """
    绘制方框
    Args:
        npImg:
        targets:

    Returns:

    """
    # Used for colors drawn on bounding boxes
    COLORS = np.random.uniform(0, 255, size=(1500, 3))
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

            idx = 0
            # draw the bounding box and label on the frame
            label = "{}: {:.2f}%".format(
                "Human", targets["confidence"][i] * 100)
            cv2.rectangle(npImg, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(npImg, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        traceback.print_exception(e)
        print("ERROR: " + str(e))
        print("Ask @Wonder for help in our Discord in the #ai-aimbot channel ONLY: https://discord.gg/rootkitorg")
