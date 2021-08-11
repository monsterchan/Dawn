# import  detect_image
from detect_Json import  Detect
import cv2

predict = Detect()
file_path = f'./inference/outJson/1740871411021303566.jpg'

image ,res = predict.detect(file_path,True)
print(image)
cv2.imshow('1',image)
if cv2.waitKey(200) == ord('q'):  # q to quit
    raise StopIteration
print(res)