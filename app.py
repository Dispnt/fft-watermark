from flask import Flask, request, redirect, url_for, session, render_template, jsonify
import cv2
import numpy as np
import random
import math
import os
from datetime import timedelta

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


def encode(im_path, mark_path, res_path):
    # im = cv2.imdecode(np.fromfile(im_path, dtype=np.uint8), -1)
    im = cv2.imread(im_path) / 255
    # mark = cv2.imdecode(np.fromfile(mark_path, dtype=np.uint8), -1)
    mark = cv2.imread(mark_path) / 255
    im_height, im_width, im_channel = np.shape(im)
    mark_height, mark_width = mark.shape[0], mark.shape[1]
    im_f = np.fft.fftshift(np.fft.fft2(im))

    x, y = list(range(math.floor(im_height / 2))), list(range(im_width))
    random.seed(im_height + im_width)
    random.shuffle(x)
    random.shuffle(y)
    tmp = np.zeros(im.shape)
    for i in range(math.floor(im_height / 2)):
        for j in range(im_width):
            if x[i] < mark_height and y[j] < mark_width:
                tmp[i][j] = mark[x[i]][y[j]]
                tmp[im_height - i - 1][im_width - j - 1] = tmp[i][j]  # 对称
    res_f = im_f + tmp * 6  # 混杂
    res = np.fft.ifftshift(res_f)  # 逆变换
    # res = np.real(res)
    res = np.abs(np.fft.ifft2(res)) * 255  # 回乘
    # cv2.imencode('.jpg', res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tofile(res_path)
    cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def decode(ori_path, im_path, res_path):
    # ori = cv2.imdecode(np.fromfile(ori_path, dtype=np.uint8), -1)
    ori = cv2.imread(ori_path) / 255
    # im = cv2.imdecode(np.fromfile(im_path, dtype=np.uint8), -1)
    im = cv2.imread(im_path) / 255
    im_height, im_width, im_channel = np.shape(ori)
    # ori_f = np.fft.fft2(ori)
    ori_f = np.fft.fftshift(np.fft.fft2(ori))
    # im_f = np.fft.fft2(im)
    im_f = np.fft.fftshift(np.fft.fft2(im))
    mark = np.abs((im_f - ori_f) / 2)
    res = np.zeros(ori.shape)

    x, y = list(range(math.floor(im_height / 2))), list(range(im_width))  # 获取随机种子
    random.seed(im_height + im_width)
    random.shuffle(x)
    random.shuffle(y)
    for i in range(math.floor(im_height / 2)):
        for j in range(im_width):
            res[x[i]][y[j]] = mark[i][j] * 255
            res[im_height - i - 1][im_width - j - 1] = res[i][j]
    # cv2.imencode('.jpg', res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tofile(res_path)
    cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


basepath = os.path.dirname(__file__)
imgpath = 'static\\images'  # 'static/images'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/encode', methods=['POST', 'GET'])
def encoderoute():
    if request.method == 'POST':
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return render_template('index.html')
        upload_path = os.path.join(basepath, imgpath, 'encode.png')
        f.save(upload_path)
        f.close()
        # opencv
        encode(upload_path, os.path.join(basepath, imgpath, 'watermark.png'),
               os.path.join(basepath, imgpath, 'encoded.png'))
        return render_template('encoded.html')
    return render_template('index.html')


@app.route('/decode', methods=['POST', 'GET'])
def decoderoute():
    if request.method == 'POST':
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return render_template('index.html')
        upload_path = os.path.join(basepath, imgpath, 'original.png')
        f.save(upload_path)
        f.close()
        decode(os.path.join(basepath, imgpath, 'encode.png'), upload_path,
               os.path.join(basepath, imgpath, 'decoded.png'))
        return render_template('decoded.html')

    return render_template('index.html')


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
