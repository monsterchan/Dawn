import base64
import json
import os
from io import BytesIO
import  numpy as np
import cv2
import requests
from PIL import Image
import  random

print(os.getcwd())
image_path = './inference/outJson/zidane.jpg'

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


##当图片存在时，以二进制形式打开图片
if os.path.isfile(image_path):
    print("图片存在！")
    file = open(image_path, mode='rb')
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3",
        "Connection": "keep-alive",
        "Host": "36kr.com/newsflashes",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:55.0) Gecko/20100101 Firefox/55.0"
    }
    files = {'file': ('zidane', file, 'image/jpg')}

    # url = 'http://124.114.22.159:20051/api'
    url = 'http://127.0.0.1:5000/api'
    # res = requests.get(url)
    print("整理好数据！",headers,files)
    re = requests.post(url,headers,files=files)
    print("发送请求！")
    if re.status_code == 200:
        data =  json.loads(re.content)
        imgName = data['imageName']
        boxList = data['resJson']
        clsList = data['resCls']

        #收到boxList，绘制
        for eachbox in boxList:
            plot_one_box(eachbox, img, color=None, label=None, line_thickness=None)
        #整理数据格式
        #存储数据



        # image = Image.fromarray(np.uint8(np.array(data['ImageList'])))
        #当前图片通道RGB
        # r,g,b = image.split()
        # img = Image.merge("RGB",[b,g,r])
        # label = data['message']
        # image = base64.decodebytes(data['ImageBytes'])
        print("接收到数据！")
        # print(label)
        # print(img)
        # im = Image.open(image.encode('ascii'))
        # bytes_stream = BytesIO(re.content)
        # roiimg = Image.open(bytes_stream)
        # img.show()
else:
    print("图片路径有问题")