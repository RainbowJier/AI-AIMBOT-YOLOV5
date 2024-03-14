import ctypes
import os
import pathlib
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
from config import useMask, maskWidth, maskHeight, aaQuitKey, screenShotHeight, confidence, \
    headshot_mode, cpsDisplay, visuals, centerOfScreen
from utils.general import (cv2, non_max_suppression, xyxy2xywh)

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

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


def main():
    # External Function for running the game selection menu (gameSelection.py)
    camera, cWidth, cHeight = gameSelection.gameSelection()

    # Used for forcing garbage collection
    count = 0
    sTime = time.time()

    # Loading Yolo5 Small AI Model, for better results use yolov5m or yolov5l
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5m',
    #                        pretrained=True, force_reload=True)
    # 加载本地模型
    model = torch.hub.load('.', 'custom', source='local',
                           path='weights/CS2-2/yolov5s.pt', force_reload=True
                           )
    """
    names：模型分析的类别列表，例如{0: 'ct', 1: 'hat ct', 2: 'hat t', 3: 't'}
    """
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

            # 检测所有对象
            results = model(im, size=screenShotHeight)

            # 目标识别，抑制不符合阈值的结果
            pred = non_max_suppression(
                results, confidence, confidence, [0, 1, 2, 3], False, max_det=30)

            # 初始化一个空列表，用于存储处理后的目标检测结果
            targets = []
            # 遍历模型预测的所有检测结果
            for i, det in enumerate(pred):
                # 初始化一个空字符串，用于记录每个类别的检测数量信息
                s = ""

                # 获取输入图像的尺寸信息，并将其转换为PyTorch张量
                # 用于后续计算时对检测框坐标进行归一化
                gn = torch.tensor(im.shape)[[0, 0, 0, 0]]
                # 如果当前批次的检测结果不为空
                if len(det):
                    # 遍历当前批次检测结果中所有不同的类别
                    for c in det[:, -1].unique():  # det[:, -1].unique().tolist(),类别的下标，对应yaml文件顺序[0.0, 1.0]
                        # 计算属于类别c的检测目标数量
                        print(det[:, -1] == c)
                        n = (det[:, -1] == c).sum()  # tensor(1, device='cuda:0')

                        # 将类别c的检测数量和类别名添加到字符串s中
                        # 假设names是一个字典或列表，存储了类别ID到类别名称的映射
                        s += f"{n} {names[int(c)]}, "

                    for *xyxy, conf, cls in reversed(det):
                        targets.append((xyxy2xywh(torch.tensor(xyxy).view(
                            1, 4)) / gn).view(-1).tolist() + [float(conf)])  # normalized xywh

            targets = pd.DataFrame(
                targets, columns=['current_mid_x', 'current_mid_y', 'width', "height", "confidence"])

            # 自瞄框的范围
            center_screen = [cWidth, cHeight]

            """
            移动鼠标部分
            
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
                    headshot_offset = box_height * 0.3
                else:
                    headshot_offset = box_height * 0.2

                mouseMove = [xMid - cWidth, (yMid - headshot_offset) - cHeight]

                # 点击按钮移动鼠标------------->后续可以修改无需按钮直接移动鼠标
                if win32api.GetKeyState(0x14):
                    # Logitech.mouse.move(int(mouseMove[0] * aaMovementAmp), int(mouseMove[1] * aaMovementAmp))
                    Logitech.mouse.move(int(mouseMove[0] * 1.1), int(mouseMove[1] * 1.1))
                    Logitech.mouse.move(int(mouseMove[0] * 1.0), int(mouseMove[1] * 1.0))
                    Logitech.mouse.move(int(mouseMove[0] * 0.9), int(mouseMove[1] * 0.9))
                    Logitech.mouse.move(int(mouseMove[0] * 0.8), int(mouseMove[1] * 0.8))
                    Logitech.mouse.move(int(mouseMove[0] * 0.6), int(mouseMove[1] * 0.6))
                    Logitech.mouse.move(int(mouseMove[0] * 0.5), int(mouseMove[1] * 0.5))

                last_mid_coord = [xMid, yMid]
            else:
                last_mid_coord = None

            # See what the bot sees
            if visuals:
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
