from time import sleep
from picamera import PiCamera
import numpy as np
import cv2
from tensorflow import keras
from sense_hat import SenseHat

# Set up Camera
cam = PiCamera()
cam.resolution = (100, 100)

# Set up LED panel
s = SenseHat()
s.clear()
s.set_pixel(0, 0, (0, 0, 255))
s.set_rotation(r=90) #0 (hdmi side), 90, 180, 270
red = (255, 0, 0)

# Load model
model = keras.models.load_model(r"/home/pi/CPE4903/digit_recog.h5")

output = np.empty((112*128*3,), dtype=np.uint8)

for i in range(4):
    # Start camera
    cam.start_preview()
    sleep(1)

    # Save picture to array
    cam.capture(output, 'rgb')
    img = output.reshape((112, 128, 3))
    img = img[:100, :100, :]

    cam.stop_preview()

    # Edit image
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.resize(gray_img, (28, 28))
    gray_img = cv2.bitwise_not(gray_img)

    # Recognize digit
    X_img = gray_img.reshape(1, 28, 28, 1)/255
    mpred = model.predict(X_img)
    smpl_pred = np.argmax(mpred, axis=-1)
    
    # Find confidence
    temp = mpred[0, int(smpl_pred)]*100
    conf = str(int(temp))
    conf = conf + '%'

    # Display digit prediction and confidence
    num = str(smpl_pred)
    s.show_letter(num, text_colour=red)
    sleep(2)
    s.show_letter(conf, text_colour=red)
    s.clear(
