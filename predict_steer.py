import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread
from skimage.transform import resize
from keras.models import load_model

test_df = pd.read_csv('test_df.csv')
model = load_model('saved_models/best_weights_relu_conv_tanh_fc_tanh_rad.h5')

def smoother(pred_degree, smoothed_angle):
    diff = pred_degree - smoothed_angle
    return pow(abs(diff), 2.0/3.0) * (diff / abs(diff))

data_dir = 'image_data/test/'
smoothed_angle = 0

steer_img = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = steer_img.shape

test_vals = test_df[['image_id', 'angle_deg']].values

idx = 0

while(cv2.waitKey(10) != ord('q')):

    try:
        full_image = cv2.imread(f"{data_dir}{test_vals[idx][0]}")
        cropped_image = resize(full_image[-150:], (66, 200))
        pred_degree = np.rad2deg(model.predict(cropped_image[np.newaxis, ])).ravel()[0]
        print(f"index: {idx}, actual: {test_vals[idx][1]}, pred: {pred_degree : .3f}, error: {test_vals[idx][1]-pred_degree : .3f}")
        cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
        #make smooth angle transitions by turning the steering wheel
        #based on the difference of the current angle
        #and the predicted angle
        smoothed_angle += 0.2 * smoother(pred_degree, smoothed_angle)
        M = cv2.getRotationMatrix2D((cols/2,rows/2), -smoothed_angle, 1)
        dst = cv2.warpAffine(steer_img, M, (cols, rows))
        cv2.imshow("steering wheel", dst)
        idx += 1

    except:
        break
  
cv2.destroyAllWindows()
