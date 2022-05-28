#%%
from urllib import response
from flask import Flask, render_template, request, flash, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pencilsketch import *
import random
from skimage import io, exposure
from scipy import ndimage

app = Flask(__name__)
app.config.from_object('config')
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH']

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
path_upload_file='static/upload/'
file_name_filtered = 'filteredPic.jpg'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload1', methods=['POST'])
def upload_image_filter_one():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        ft1 = filter_one(file.filename)
        return render_template('index.html', filename=ft1)
    else:
        return redirect(request.url)

@app.route('/upload2', methods=['POST'])
def upload_image_filter_two():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_image filename: ' + file.filename)
        ft2 = filter_two(file.filename)
        return render_template('index.html', filename=ft2)
    else:
        return redirect(request.url)

@app.route('/upload3', methods=['POST'])
def upload_image_filter_three():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_image filename: ' + file.filename)
        ft3 = filter_three(file.filename)
        return render_template('index.html', filename=ft3)
    else:
        return redirect(request.url)

@app.route('/upload4', methods=['POST'])
def upload_image_filter_four():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        ft4 = filter_four(file.filename)
        return render_template('index.html', filename=ft4)
    else:
        return redirect(request.url)

@app.route('/upload5', methods=['POST'])
def upload_image_filter_five():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        ft5 = filter_five(file.filename)
        return render_template('index.html', filename=ft5)
    else:
        return redirect(request.url)

@app.route('/download', methods=['POST'])
def download_image():
    return send_file(path_upload_file+file_name_filtered, as_attachment=True)

@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='upload/' + file_name_filtered), code=301)

@app.route('/my-link')
def my_link():
    print("I got clicked!")
    return 'Click.'

def filter_one(uploaded_img): #old fashion
    print('filter_one uploaded_img: ', uploaded_img)
    # reading source file
    img = cv2.imread(path_upload_file+uploaded_img)
    print('filter_one img: ', img)
    # converting the image into gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.medianBlur(img, 1)

    # applying adaptive threshold to use it as a mask
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    color = cv2.bilateralFilter(img2, 9, 200, 200)

    # cartoonize
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    cv2.imwrite(path_upload_file+file_name_filtered, cartoon)
    print('filter_one cartoon: ', cartoon)
    return file_name_filtered

def filter_two(uploaded_img): #sketch pencil 3
    print('filter_two uploaded_img: ', uploaded_img)
    # reading source file
    img = io.imread('static/upload/'+uploaded_img)

    pencil_tex = 'static/pencils/pencil3.jpg'
    print('filter_five pencil_tex: ', pencil_tex)
    im_pen = gen_pencil_drawing(img, kernel_size=8, stroke_width=1, num_of_directions=8,
                                           smooth_kernel="gauss",
                                           gradient_method=1, rgb=True, w_group=2, pencil_texture_path=pencil_tex,
                                           stroke_darkness=2, tone_darkness=1.5)

    im_pen = exposure.rescale_intensity(im_pen, in_range=(0, 1))
    io.imsave(path_upload_file+file_name_filtered, im_pen)
    return file_name_filtered

def filter_three(uploaded_img): #oil
    print('filter_three uploaded_img: ', uploaded_img)
    #Parameter you can choose from 
    brush_width=3 #The size of the brush
    gradient='scharr' # The type of the artstyle you want to choose
    result=[]

    # reading source file
    img = cv2.imread('static/upload/'+uploaded_img)
    print('filter_three img: ', img)

    r = 2 * int(img.shape[0] / 50) + 1
    Gx, Gy = get_gradient(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (r, r), gradient)
    
    Gh = np.sqrt(np.sqrt(np.square(Gx) + np.square(Gy)))    # Length of the ellipse
    Ga = (np.arctan2(Gy, Gx) / np.pi) * 180 + 90            # Angle of the ellipse

    canvas = cv2.medianBlur(img, 11)

    order = draw_order(img.shape[0], img.shape[1], scale=brush_width*2)
    colors = np.array(img, dtype=np.float)

    for i, (y, x) in enumerate(order):
        length = int(round(brush_width + brush_width * Gh[y, x]))
        color = colors[y,x]
        cv2.ellipse(canvas, (x, y), (length, brush_width), Ga[y, x], 0, 360, color, -1, cv2.LINE_AA)

    result.append(canvas)

    cv2.imwrite(path_upload_file+file_name_filtered, result[0])
    print('filter_three cartoon: ', result[0])
    return file_name_filtered

def filter_four(uploaded_img): #pencil sketch by opencv
    print('filter_two uploaded_img: ', uploaded_img)
    
    # reading source file
    img = cv2.imread(path_upload_file+uploaded_img)

    # converting the image into gray-scale
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    invert_img = cv2.bitwise_not(grey_img)

    blur_img = cv2.GaussianBlur(invert_img, (111, 111), 0)
    invblur_img = cv2.bitwise_not(blur_img)

    # cartoonize sketch
    sketch_img = cv2.divide(grey_img, invblur_img, scale=256.0)

    cv2.imwrite(path_upload_file+file_name_filtered, sketch_img)
    print('filter_two sketch_img: ', sketch_img)
    return file_name_filtered

def filter_five(uploaded_img): #sketch pencil 0
    print('filter_two uploaded_img: ', uploaded_img)
    # reading source file
    img = io.imread('static/upload/'+uploaded_img)

    pencil_tex = 'static/pencils/pencil0.jpg' 
    print('filter_five pencil_tex: ', pencil_tex)
    im_pen = gen_pencil_drawing(img, kernel_size=8, stroke_width=1, num_of_directions=8,
                                           smooth_kernel="gauss",
                                           gradient_method=1, rgb=True, w_group=2, pencil_texture_path=pencil_tex,
                                           stroke_darkness=2, tone_darkness=1.5)
    im_pen = exposure.rescale_intensity(im_pen, in_range=(0, 1))
    io.imsave(path_upload_file+file_name_filtered, im_pen)
    return file_name_filtered
 
def draw_order(h, w, scale):
    order = []
    for i in range(0, h, scale):
        for j in range(0, w, scale):
            y = random.randint(-scale // 2, scale // 2) + i
            x = random.randint(-scale // 2, scale // 2) + j
            order.append((y % h, x % w))
    return order

def get_gradient(img_o, ksize, gtype):
    if gtype == 'scharr':
        X = cv2.Scharr(img_o, cv2.CV_32F, 1, 0) / 50.0
        Y = cv2.Scharr(img_o, cv2.CV_32F, 0, 1) / 50.0
    elif gtype == 'prewitt':
        X, Y = prewitt(img_o)
    elif gtype == 'sobel':
        X = cv2.Sobel(img_o,cv2.CV_32F,1,0,ksize=5)  / 50.0
        Y = cv2.Sobel(img_o,cv2.CV_32F,0,1,ksize=5)  / 50.0
    elif gtype == 'roberts':
        X, Y = roberts(img_o)
    else:
        print('Not suppported type!')
        exit()

    # Blur the Gradient to smooth the edge
    X = cv2.GaussianBlur(X, ksize, 0)
    Y = cv2.GaussianBlur(Y, ksize, 0)
    return X, Y
    
def prewitt(img):
    img_gaussian = cv2.GaussianBlur(img,(3,3),0)
    kernelx = np.array( [[1, 1, 1],[0, 0, 0],[-1, -1, -1]] )
    kernely = np.array( [[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]] )
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
    return img_prewittx // 15.36, img_prewitty // 15.36

def roberts(img):
    roberts_cross_v = np.array( [[ 0, 0, 0 ],
                                 [ 0, 1, 0 ],
                                 [ 0, 0,-1 ]] )
    roberts_cross_h = np.array( [[ 0, 0, 0 ],
                                 [ 0, 0, 1 ],
                                 [ 0,-1, 0 ]] )
    vertical = ndimage.convolve( img, roberts_cross_v )
    horizontal = ndimage.convolve( img, roberts_cross_h )
    return vertical // 50.0, horizontal // 50.0    


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 9000)))

# %%
