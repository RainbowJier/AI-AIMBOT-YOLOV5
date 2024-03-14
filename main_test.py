import ctypes
import os
import time

import numpy as np
import pandas as pd
import torch
import win32api
from pynput import mouse

import gameSelection
# Could be do with
# from config import *
# But we are writing it out for clarity for new devs
from config import aaMovementAmp, useMask, maskWidth, maskHeight, aaQuitKey, screenShotHeight, confidence, \
    headshot_mode, cpsDisplay, visuals, centerOfScreen
from utils.general import (cv2, non_max_suppression, xyxy2xywh)

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
            """
            相对移动, 绝对移动需配合 pywin32 的 win32gui 中的 GetCursorPos 计算位置
            pip install pywin32 -i https://pypi.tuna.tsinghua.edu.cn/simple
            x: 水平移动的方向和距离, 正数向右, 负数向左
            y: 垂直移动的方向和距离
            """
            if not ok:
                return
            if x == 0 and y == 0:
                return
            driver.moveR(x, y, True)


# @smart_inference_mode()
def main():
    # External Function for running the game selection menu (gameSelection.py)
    camera, cWidth, cHeight = gameSelection.gameSelection()

    # Used for forcing garbage collection
    count = 0
    sTime = time.time()

    # Loading Yolo5 Small AI Model, for better results use yolov5m or yolov5l
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s',
                           pretrained=True, force_reload=True)
    stride, names, pt = model.stride, model.names, model.pt

    if torch.cuda.is_available():
        model.half()

    # Used for colors drawn on bounding boxes
    COLORS = np.random.uniform(0, 255, size=(1500, 3))

    # Main loop Quit if Q is pressed
    last_mid_coord = None
    with torch.no_grad():
        while win32api.GetAsyncKeyState(ord(aaQuitKey)) == 0:

            # Getting Frame
            npImg = np.array(camera.get_latest_frame())

            from config import maskSide  # "temporary" workaround for bad syntax
            if useMask:
                maskSide = maskSide.lower()
                if maskSide == "right":
                    npImg[-maskHeight:, -maskWidth:, :] = 0
                elif maskSide == "left":
                    npImg[-maskHeight:, :maskWidth, :] = 0
                else:
                    raise Exception('ERROR: Invalid maskSide! Please use "left" or "right"')

            # Normalizing Data
            im = torch.from_numpy(npImg)
            if im.shape[2] == 4:
                # If the image has an alpha channel, remove it
                im = im[:, :, :3, ]

            im = torch.movedim(im, 2, 0)
            if torch.cuda.is_available():
                im = im.half()
                im /= 255
            if len(im.shape) == 3:
                im = im[None]

            # 检测目标
            results = model(im, size=screenShotHeight)

            # 抑制不符合阈值的结果
            pred = non_max_suppression(
                results, confidence, confidence, 0, False, max_det=1000)

            # Converting output to usable cords
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

            # 自动获取游戏窗口大小
            center_screen = [cWidth, cHeight]

            # 如果中心边界框中有人
            if len(targets) > 0:
                if (centerOfScreen):
                    # 计算距中心的距离
                    targets["dist_from_center"] = np.sqrt((targets.current_mid_x - center_screen[0]) ** 2 + (
                            targets.current_mid_y - center_screen[1]) ** 2)

                    # 按距中心的距离对数据框进行排序
                    targets = targets.sort_values("dist_from_center")

                # 获取最后一个人的中间坐标（如果存在）
                if last_mid_coord:
                    targets['last_mid_x'] = last_mid_coord[0]
                    targets['last_mid_y'] = last_mid_coord[1]
                    # 获取当前人的中间坐标和最后一个人的中间坐标之间的距离
                    targets['dist'] = np.linalg.norm(
                        targets.iloc[:, [0, 1]].values - targets.iloc[:, [4, 5]], axis=1)
                    targets.sort_values(by="dist", ascending=False)

                # 鼠标的位置
                mouse_x, mouse_y = m.position

                # 选取数据框中出现的第一个人（回想一下，我们根据欧几里德距离进行排序）
                xMid = targets.iloc[0].current_mid_x
                yMid = targets.iloc[0].current_mid_y

                box_height = targets.iloc[0].height
                # 不同模式，瞄准的位置高度
                if headshot_mode:
                    headshot_offset = box_height * 0.4  # 头部
                    # headshot_offset = box_height * 0.4  # 头部
                else:
                    headshot_offset = box_height * 0.2  # 胸部

                mouseMove = [xMid - cWidth, (yMid - headshot_offset) - cHeight]

                # 点击按钮移动鼠标------------->后续可以修改无需按钮直接移动鼠标
                if win32api.GetKeyState(0x14):
                    Logitech.mouse.move(int(mouseMove[0] * aaMovementAmp), int(mouseMove[1] * aaMovementAmp))

                    # win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(
                    #     mouseMove[0] * aaMovementAmp), int(mouseMove[1] * aaMovementAmp), 0, 0)
                last_mid_coord = [xMid, yMid]



            # 框中没有人
            else:
                last_mid_coord = None

            # 看看机器人看到了什么
            if visuals:
                # 循环遍历每个已识别的项目并绘制边界框
                for i in range(0, len(targets)):
                    halfW = round(targets["width"][i] / 2)
                    halfH = round(targets["height"][i] / 2)
                    midX = targets['current_mid_x'][i]
                    midY = targets['current_mid_y'][i]
                    (startX, startY, endX, endY) = int(
                        midX + halfW), int(midY + halfH), int(midX - halfW), int(midY - halfH)

                    idx = 0

                    # 在框架上绘制边界框和标签
                    label = "{}: {:.2f}%".format(
                        "Human", targets["confidence"][i] * 100)
                    cv2.rectangle(npImg, (startX, startY), (endX, endY),
                                  COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(npImg, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            # Forced garbage cleanup every second
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


if __name__ == "__main__":
    try:
        m = mouse.Controller()
        main()
    except Exception as e:
        import traceback

        traceback.print_exception(e)
        print("ERROR: " + str(e))
        print("Ask @Wonder for help in our Discord in the #ai-aimbot channel ONLY: https://discord.gg/rootkitorg")
