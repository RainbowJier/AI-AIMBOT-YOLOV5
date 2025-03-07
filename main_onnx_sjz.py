import ctypes
import os
import pathlib
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import win32api
import win32con
from pynput import mouse

import gameSelection
# Could be do with
# from config import *
# But we are writing it out for clarity for new devs
from config_sjz import aaQuitKey, confidence, cpsDisplay, \
    visuals, onnxChoice, centerOfScreen, aim_range
from utils.general import (cv2, non_max_suppression, xyxy2xywh)

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

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
    # 用于运行游戏选择菜单的外部功能（gameSelection.py）
    camera, cWidth, cHeight, region = gameSelection.gameSelection()

    #  用于强制垃圾回收
    count = 0
    sTime = time.time()

    # 基于config.py选择正确的ONNX提供程序
    onnxProvider = ""
    if onnxChoice == 1:
        onnxProvider = "CPUExecutionProvider"
    elif onnxChoice == 2:
        onnxProvider = "DmlExecutionProvider"
    elif onnxChoice == 3:
        import cupy as cp
        onnxProvider = "CUDAExecutionProvider"

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_sess = ort.InferenceSession('weights/sanjiaozhou/v1/320-FP16/sjz.onnx',
                                    sess_options=so, providers=[onnxProvider])

    # 获取模型的输入信息
    input_meta = ort_sess.get_inputs()
    input_type = input_meta[0].type  # TypE
    input_shape = input_meta[0].shape  # shape
    print("模型的tensor：", input_type)
    print("模型的shape：", input_shape)

    # 用于在边界框上绘制的颜色
    COLORS = np.random.uniform(0, 255, size=(1500, 3))

    # 如果按下Q，主循环退出
    last_mid_coord = None
    while win32api.GetAsyncKeyState(ord(aaQuitKey)) == 0:
        # 获取框架
        npImg = np.array(camera.get_latest_frame())  # 获取截图

        # If Nvidia, do this
        if onnxChoice == 3:
            # 归一化数据
            im = torch.from_numpy(npImg).to('cuda')
            if im.shape[2] == 4:
                # If the image has an alpha channel, remove it
                im = im[:, :, :3, ]

            im = torch.movedim(im, 2, 0)
            # 根据onnx模型，转换tensor是float/float16
            if (input_type == "tensor(float16)"):
                im = im.half()
            else:
                im = im.float()

            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        # If AMD or CPU, do this
        else:
            # Normalizing Data
            im = np.array([npImg])
            if im.shape[3] == 4:
                # If the image has an alpha channel, remove it
                im = im[:, :, :, :3]
            im = im / 255
            im = im.astype(np.half)
            im = np.moveaxis(im, 3, 1)

        # If Nvidia, do this
        if onnxChoice == 3:
            outputs = ort_sess.run(None, {'images': cp.asnumpy(im)})

        # If AMD or CPU, do this
        else:
            outputs = ort_sess.run(None, {'images': np.array(im)})

        im = torch.from_numpy(outputs[0]).to('cpu')

        """模型识别"""
        pred = non_max_suppression(im, confidence, confidence, [0, 1, 2, 3], False, max_det=3)

        # 将预测结果转换为目标框和置信度
        targets = []
        for i, det in enumerate(pred):
            s = ""
            # 获取图像尺寸
            gn = torch.tensor(im.shape)[[0, 0, 0, 0]]
            # 如果检测结果不为空，遍历类别和检测框
            if len(det):
                # 获取类别统计
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # 检测框数量 per 类别
                    s += f"{n} {int(c)}, "  # 添加到字符串

                # 遍历检测框，添加normalized xywh和置信度
                for *xyxy, conf, cls in reversed(det):
                    # 将 xyxy 转化为 normalized xywh
                    targets.append((xyxy2xywh(torch.tensor(xyxy).view(
                        1, 4)) / gn).view(-1).tolist() + [float(conf)])  # normalized xywh

        # 将目标框和置信度转换为 DataFrame
        targets = pd.DataFrame(
            targets, columns=['current_mid_x', 'current_mid_y', 'width', "height", "confidence"])

        # 自动瞄准范围
        center_screen = [cWidth, cHeight]

        # 如果中心边界框中有人
        if len(targets) > 0:
            if (centerOfScreen):
                # 计算到中心的距离
                targets["dist_from_center"] = np.sqrt(
                    (targets.current_mid_x - center_screen[0]) ** 2 + (targets.current_mid_y - center_screen[1]) ** 2)

                # Sort the data frame by distance from center
                targets = targets.sort_values("dist_from_center")

            # 获取最后一个人的中间坐标（如果存在）
            if last_mid_coord:
                targets['last_mid_x'] = last_mid_coord[0]
                targets['last_mid_y'] = last_mid_coord[1]
                # Take distance between current person mid coordinate and last person mid coordinate
                targets['dist'] = np.linalg.norm(
                    targets.iloc[:, [0, 1]].values - targets.iloc[:, [4, 5]], axis=1)
                targets.sort_values(by="dist", ascending=False)

            # 根据欧几里得距离进行排序，获取距离准心最近的人的坐标
            xMid = targets.iloc[0].current_mid_x
            yMid = targets.iloc[0].current_mid_y

            box_height = targets.iloc[0].height
            # 瞄准头部
            headshot_offset = box_height * 0.1

            # 最终移动的坐标
            mouseMove = [xMid - cWidth, (yMid - headshot_offset) - cHeight]

            # 修改开启自瞄开关,0x43:c按钮
            if win32api.GetKeyState(0x14):
                if (win32api.GetKeyState(win32con.VK_LBUTTON) < 0):
                    # capture_screen(camera)
                    Logitech.mouse.move(int(mouseMove[0]), int(mouseMove[1]))
                else:
                    Logitech.mouse.move(int(mouseMove[0]), int(mouseMove[1]))

            # 当点击右键
            last_mid_coord = [xMid, yMid]


        else:
            last_mid_coord = None

        # 描绘人物
        if visuals:
            # Loops over every item identified and draws a bounding box
            for i in range(0, len(targets)):
                halfW = round(targets["width"][i] / 2)
                halfH = round(targets["height"][i] / 2)
                midX = targets['current_mid_x'][i]
                midY = targets['current_mid_y'][i]
                (startX, startY, endX, endY) = (int(midX + halfW),
                                                int(midY + halfH),
                                                int(midX - halfW),
                                                int(midY - halfH))

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
        m = mouse.Controller()
        main()
    except Exception as e:
        import traceback

        traceback.print_exception(e)
        print("ERROR: " + str(e))
        print("Ask @Wonder for help in our Discord in the #ai-aimbot channel ONLY: https://discord.gg/rootkitorg")
