from rembg import remove
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
import json

app = Flask(__name__)
model = load_model('b64_e50_lr001.h5')
target_img = os.path.join(os.getcwd() , 'static/images')

@app.template_filter('enumerate')
def jinja2_enumerate(iterable, start=0):
    return enumerate(iterable, start)

@app.route('/')
def index_view():
    return render_template('index.html')

# Melakukan pengecekan ekstensi file
ALLOWED_EXT = set(['jpg' , 'jpeg'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

@app.route('/rembg',methods=['GET','POST'])
def rembg():
    if request.method == 'POST':
        rembg = request.form['rembg']
        file = request.files['file']
        if file and allowed_file(file.filename):
            # save uploaded file to desired folder
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)

            # Load uploaded file
            img = load_img(file_path, target_size=(100, 100))
            if rembg == "1":
              remBG = remove(img)
              img = remBG.convert("RGB")
            output_path = os.path.join('static/rmbg', filename)
            img.save(output_path)

            return render_template('rembg.html', user_image=output_path, ori_image=file_path, rembg=rembg)
        else:
            return "Unable to read the file. Please check file extension"

@app.route('/rescale',methods=['GET','POST'])
def rescale():
    if request.method == 'POST':
        file = request.form['user_image']
        img = load_img(file, target_size=(100, 100))
        x = image.img_to_array(img)
        x_original = np.array(image.img_to_array(img))
        x_original = np.expand_dims(x_original, axis=0)
        x = np.array(x) / 255.0
        x = np.expand_dims(x, axis=0)
        x_str = json.dumps(x.tolist())

        return render_template('rescale.html', img_array=x_str, user_image=file, matrix=x[0], ori_matrix=x_original[0])

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file_path = request.form['file_path']
        file_name = file_path.split("\\")

        serialized_array =request.form['img_array']
        img = json.loads(serialized_array)
        img = np.array(img)
        class_prediction=model.predict(img)

        hasil_softmax = [
          "{:.10f}".format(float(class_prediction[0][0])),
          "{:.10f}".format(float(class_prediction[0][1])),
          "{:.10f}".format(float(class_prediction[0][2]))
        ]

        pct = np.max(class_prediction)
        classes_x=np.argmax(class_prediction,axis=1)

        pct = np.max(class_prediction)
        classes_x=np.argmax(class_prediction,axis=1)
        if classes_x == 0:
          coffee = "Light Roast"
        elif classes_x == 1:
          coffee = "Medium Roast"
        elif classes_x == 2:
          coffee = "Dark Roast"
              
        return render_template('predict.html', coffee = coffee, prob=pct, user_image = os.path.join('static/rmbg', file_name[-1]), softmax=hasil_softmax)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)