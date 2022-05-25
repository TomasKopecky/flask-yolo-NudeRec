import io
import os
from io import BytesIO
import json
from PIL import Image
import base64
import subprocess
import requests

import torch
from flask import Flask, jsonify, url_for, render_template, request, redirect

app = Flask(__name__)

RESULT_FOLDER = os.path.join('./static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

print(os.getcwd())

URL = "https://github.com/ronadlisko/flask-yolo-NudeRec/raw/main/model/model.pt"
response = requests.get(URL)
open("model.pt", "wb").write(response.content)

model = torch.hub.load('ultralytics/yolov5','custom', path='model.pt', force_reload=True).autoshape() # 'yolov5s', pretrained=True)#.autoshape()  # for PIL/cv2/np inputs and NMS
model.eval()
#subprocess.run(["python", "print('asfasdfasdf')"])

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images

# Inference
    results = model(imgs, size=640)  # includes NMS
    #print(results)
    return results

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return

        img_bytes = file.read()

        #print("PATH", app.config['RESULT_FOLDER'])

        results = get_prediction(img_bytes)
        #print(results)
        results.imgs # array of original images (as np array) passed to model for inference
        results.render()  # updates results.imgs with boxes and labels
        results.print()
        print(results.pandas().xyxy[0])
        
        for img in results.imgs:
            #buffered = BytesIO()
            img_base64 = Image.fromarray(img)
            img_base64.save('./static/result.jpg')
            #buff = new Buffer(data, 'base64');
            #fs.writeFileSync('stack-abuse-logo-out.png', buff);
            #print(base64.b64encode(buffered.getvalue()).decode('utf-8'))  # base64 encoded image with results
        

        #results.save()  # save as results1.jpg, results2.jpg... etc.
        #os.rename("./saldfj.jpg", "./static/results0.jpg")

        

        #full_filename = os.path.join(app.config['RESULT_FOLDER'], './results0.jpg')
        full_filename = os.path.join(app.config['RESULT_FOLDER'], './result.jpg')
        return render_template('result.html',result_image=full_filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
