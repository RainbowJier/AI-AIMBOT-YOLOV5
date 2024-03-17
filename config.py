# Portion of screen to be captured (This forms a square/rectangle around the center of screen)
# 设置图片截取的大小
screenShotHeight = 320
screenShotWidth = 320

# Use "left" or "right" for the mask side depending on where the interfering object is, useful for 3rd player models or large guns
useMask = False
maskSide = "left"
maskWidth = 80
maskHeight = 200


# ct otherwise t
CT = False

# fps
target_fps = 160

# Auto-aimING mouse movement amplifier
aaMovementAmp = 0.4

# 鼠标平滑
# lock平滑系数；越大越平滑，最低1.0
lock_smooth = 4

# lock幅度系数；若在桌面试用请调成1，在游戏中(csgo)则为灵敏度
lock_sen = 0.7

# Person Class Confidence
confidence = 0.8

# What key to press to quit and shutdown the autoaim
aaQuitKey = "P"

# If you want to main slightly upwards towards the head
headshot_mode = True

# Displays the Corrections per second in the terminal
cpsDisplay = True

# Set to True if you want to get the visuals
visuals = False

# Smarter selection of people
centerOfScreen = True

# ONNX ONLY - Choose 1 of the 3 below
# 1 - CPU
# 2 - AMD
# 3 - NVIDIA
onnxChoice = 3
