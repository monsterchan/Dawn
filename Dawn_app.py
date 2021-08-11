import base64
import io
import json
import time
import cv2
from PIL import Image

from flask import Flask, render_template, request, Response, jsonify

from detect_Json import Detect

predict = Detect()

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')




# def get_byte_image(image):
#     img = Image.fromarray(image)
#     print('img',img)
#     img_byte_arr = io.BytesIO()
#     img.save(img_byte_arr, format='PNG')
#     print("img_byte_arr",type(img_byte_arr))
#     print("img_byte_arr",img_byte_arr)
#     print("img_byte_arr.getvalue()",img_byte_arr.getvalue())
#     print("base64.encodebytes(img_byte_arr.getvalue())",base64.encodebytes(img_byte_arr.getvalue()))
#     encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode()
#     print("encoded_img",encoded_img)
#
#     # encoded_img = base64.b64encode(img_byte_arr.getvalue())
#     return encoded_img
#
#
# def get_byte_image(image):
#     cl_image = image[:, :, (2, 1, 0)]
#     print("cl_image",cl_image)
#     image_data_bytes = cv2.imencode(".jpg", cl_image)[1].tostring()
#     image_data_str = image_data_bytes.encode("ISO-8859-1")
#     print("image_data_str",image_data_str)
#     return image_data_str

# def get_byte_image(image):
#     cl_image = image[:, :, (2, 1, 0)]
#     print("cl_image",cl_image)
#     image_data_bytes = cv2.imencode(".jpg", cl_image)[1].tostring()
#     image_data_str = image_data_bytes.encode("ISO-8859-1")
#     print("image_data_str",image_data_str)
#     return image_data_str

@app.route('/api',methods=['POST'])
def detectImage():
        print("接收到检测图片要求！")
        if request.method == 'POST':
            # data = request.get_data()  #读的是二进制,接受一个参数
            print("服务器接到post请求！")
            print(request.files)
            # 接收图片
            image_file = request.files['file']
            print("接收图片",image_file)

            # 获取图片名
            file_name = image_file.filename
            print("获取图片名",file_name)

            # 文件保存目录
            file_path = f'./inference/images/{file_name}.jpg'

            print("文件保存目录", file_path)

            if image_file:
                print("保存文件",file_path)
                image_file.save(file_path)
            print("已经存储完图片")
            #检测图片
            resJson , resCls = predict.detect(file_path,True)
            time.sleep(5)
            response = {'imageName':file_name,'resJson': resJson, 'resCls': resCls}
            return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=5001)
