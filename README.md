﻿# 🎯 AI Aimbot 🎮

## There are 3 Versions 🚀🚦🖥️

- Fast 🏃‍♂️ - `main.py` ✅ Easy to set up, Works on any computer 💻
- Faster 🏃‍♂️💨 - `main_onnx.py` ⚙️ May need to edit a file, Works on any computer 💻
- Fastest 🚀 - `main_tensorrt.py` 🏢 Enterprise level hard, Works on computers with Nvidia GPUs only 🎮

*** 

## 🧰 Requirements

- Nvidia RTX 980 🆙, higher or equivalent
- And one of the following:
    - Nvidia CUDA Toolkit 11.8 [DOWNLOAD HERE](https://developer.nvidia.com/cuda-11-8-0-download-archive)

*** 

## 🚀 Pre-setup Steps

1. Download and Unzip the AI Aimbot and stash the folder somewhere handy 🗂️.
2. Ensure you've got Python installed (like a pet python 🐍) – grab version
   3.11 [HERE](https://www.python.org/downloads/release/python-3116/).
3. Fire up `PowerShell` or `Command Prompt` on Windows 🔍.
4. To install `PyTorch`, select the appropriate command based on your GPU.
   -
   Nvidia `pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118`
    - AMD or CPU `pip install torch torchvision torchaudio`
5. 📦 Run the command below to install the required Open Source packages:
   ```
   pip install -r requirements.txt
   ```

## 🔌 How to Run (Fast 🏃‍♂️ Version)

Follow these steps **after** Python and all packages have been installed:

1. Open `PowerShell` ⚡ or `Command Prompt` 💻.
2. Input `cd `, then drag & drop the folder containing the bot code into the terminal.
3. Hit Enter ↩️.
4. Type `python main.py` and press Enter.
5. Use **CAPS_LOCK** to toggle the aimbot 🎯. It begins in the *off* state.
6. Pressing `q` 💣 at **ANY TIME** will shut down the program.

## 🔌 How to Run (Faster 🏃‍♂️💨 Version)

Follow these steps **after** Python and all packages have been installed:

1. Open the `config.py` 📄 file and tweak the `onnxChoice` variable to correspond with your hardware specs:
    - `onnxChoice = 1` # CPU ONLY 🖥
    - `onnxChoice = 2` # AMD/NVIDIA ONLY 🎮
    - `onnxChoice = 3` # NVIDIA ONLY 🏎️
2. IF you have an NVIDIA set up, run the following
    ``` 
    pip install onnxruntime-gpu
    pip install cupy-cuda11x
    ```
2. Follow the same steps as for the Fast 🏃‍♂️ Version above except for step 4, you will run `python main_onnx.py`
   instead.

## 🔌 How to Run (Fastest 🚀 Version)

Follow these sparkly steps to get your TensorRT ready for action! 🛠️✨

1. **Install Cupy**
   Run the following `pip install cupy-cuda11x`

2. **CUDNN Installation** 🧩
   Click to
   install [CUDNN 📥](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.6/local_installers/11.x/cudnn-windows-x86_64-8.9.6.50_cuda11-archive.zip/).
   You'll need a Nvidia account to proceed. Don't worry it's free.

3. **Unzip and Relocate** 📁➡️
   Open the .zip CuDNN file and move all the folders/files to where the CUDA Toolkit is on your machine, usually
   at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`.

4. **Get TensorRT 8.6 GA** 🔽
   Fetch [`TensorRT 8.6 GA 🛒`](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/zip/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip).

5. **Unzip and Relocate** 📁➡️
   Open the .zip TensorRT file and move all the folders/files to where the CUDA Toolkit is on your machine, usually
   at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`.

6. **Python TensorRT Installation** 🎡
   Once you have all the files copied over, you should have a folder
   at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\python`. If you do, good, then run the following command
   to install TensorRT in python.
   ```
   pip install "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\python\tensorrt-8.6.1-cp311-none-win_amd64.whl"
   ```
   🚨 If the following steps didn't work, don't stress out! 😅 The labeling of the files corresponds with the Python
   version you have installed on your machine. We're not looking for the 'lean' or 'dispatch' versions. 🔍 Just locate
   the correct file and replace the path with your new one. 🔄 You've got this! 💪
7. **Set Your Environmental Variables** 🌎
   Add these paths to your environment:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
   ```


8. **Download Pre-trained Models** 🤖
   You can use one of the .engine models we supply. But if it doesn't work, then you will need to re-export it. Grab
   the `.pt` file here for the model you want. We recommend `yolov5s.py`
   or `yolov5m.py` [HERE 🔗](https://github.com/ultralytics/yolov5/releases/tag/v7.0).

12. **Run the Export Script** 🏃‍♂️💻
    Time to execute `export.py` with the following command. Patience is key; it might look frozen, but it's just
    concentrating hard! Can take up to 20 mintues.
   ```bash
   python .\export.py --weights ./yolov5s.pt --include engine --half --imgsz 320 320 --device 0
   ```

Note: You can pick a different YOLOv5 model size. TensorRT's power allows for larger models if desired!

If you've followed these steps, you should be all set with TensorRT! ⚙️🚀

## ⚙️ Configurable Settings

*Default settings are generally great for most scenarios. Check out the comments in the code for more insights. 🔍 The
configuration settings are now located in the `config.py` file!<br>
**CAPS_LOCK is the default for flipping the switch on the autoaim superpower! ⚙️ 🎯**

`useMask` - Set to `True` or `False` to turn on and off 🎭

`maskWidth` - The width of the mask to use. Only used when `useMask` is `True` 📐

`maskHeight` - The height of the mask to use. Only used when `useMask` is `True` 📐

`aaQuitKey` - The go-to key is `q`, but if it clashes with your game style, swap it out! ⌨️♻️

`headshot_mode` - Set to `False` if you're aiming to keep things less head-on and more centered. 🎯➡️👕

`cpsDisplay` - Toggle off with `False` if you prefer not to display the CPS in your command station. 💻🚫

`visuals` - Flip to `True` to witness the AI's vision! Great for sleuthing out any hiccups. 🕵️‍♂️✅

`aaMovementAmp` - The preset should be on point for 99% of players. Lower the digits for smoother targeting. Recommended
doses: `0.5` - `2`. ⚖️🕹️

`confidence` - Stick with the script here unless you're the expert. 🧐✨

`screenShotHeight` - Same as above, no need for changes unless you've got a specific vision. 📏🖼️

`screenShotWidth` - Keep it constant as is, unless you've got reasons to adjust. 📐🖼️

`aaDetectionBox` - Default's your best bet, change only if you've got the know-how. 📦✅

`onnxChoice` - Gear up for the right graphics card—Nvidia, AMD, or CPU power! 💻👾

`centerOfScreen` - Keep this switched on to stay in the game's heart. ❤️🖥️

## 📊 Current Stats

The bot's efficiency depends on your setup. We achieved 100-150 CPS with our test specs below 🚀.

    - AMD Ryzen 7 2700
    - 64 GB DDR4
    - Nvidia RTX 3080

💡 Tip: Machine Learning can be tricky, so reboot if you keep hitting CUDA walls.

## ⚠️ Known Cheat-Detectable Games

Splitgate (reported by a Discord user 🕵️‍♂️), EQU8 detects win32 mouse movement library.

## 🚀 Custom Aimbots and Models

Show off your work or new models via Pull Requests in `customScripts` or `customModels` directories, respectively. Check
out the `example-user` folder for guidance.

## 🥐Problem

The Perfect/5e platform blocks win32 movement because pywin32 is the operating system's keyboard and mouse message, not
the driver's keyboard and mouse message.

1. Use python+ Logitech driver (perfect solution)
2. Use python+ GoBot driver
3. Increase the program’s permissions in the operating system to prevent it from being blocked

## 🌠 Future Ideas

- [x] Mask Player to avoid false positives

Happy Coding and Aiming! 🎉👾
