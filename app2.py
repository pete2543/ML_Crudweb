# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
app = Flask(__name__)

dic = {0: 'งูกะปะ กลุ่มงูมีพิษ', 1: 'งูกินทากหัวโหนก กลุ่มงูไม่มีพิษ', 2: 'งูก้นขบหางแดง กลุ่มงูไม่มีพิษ', 3: 'งูคออ่อนปากจงอย กลุ่มงูมีพิษ', 4: 'งูงวงช้าง กลุ่มงูไม่มีพิษ', 5: 'งูชายธงหลังดำ กลุ่มงูมีพิษ', 6: 'งูดินบ้าน กลุ่มงูไม่มีพิษ', 7: 'งูทับสมิงคลา กลุ่มงูมีพิษ', 8: 'งูทางมะพร้าว กลุ่มงูไม่มีพิษ', 9: 'งูปลิง กลุ่มงูพิษอ่อน', 10: 'งูปล้องฉนวน กลุ่มงูไม่มีพิษ', 11: 'งูปล้องทอง กลุ่มงูพิษอ่อน',   12: 'งูคออ่อนปากจงอย กลุ่มงูมีพิษ', 13: 'งูปากกว้างน้ำเค็ม กลุ่มงูพิษอ่อน', 14: 'งูปี่แก้ว กลุ่มงูไม่มีพิษ', 15: 'งูพริกท้องแดง กลุ่มงูมีพิษ',
       16: 'งูพริกสีน้ำตาล กลุ่มงูมีพิษ', 17: 'งูพังกา กลุ่มงูมีพิษ', 18: 'งูม่านทอง กลุ่มงูพิษอ่อน', 19: 'งูลายสอ กลุ่มงูพิษอ่อน', 20: 'งูลายสาบ กลุ่มงูพิษอ่อน', 21: 'งูลายสาบตาโต กลุ่มงูไม่มีพิษ', 22: 'งูสมิงทะเล กลุ่มงูมีพิษ', 23: 'งูสามเหลี่ยม กลุ่มงูมีพิษ', 24: 'งูสายม่าน กลุ่มงูไม่มีพิษ', 25: 'งูสิงตาโต กลุ่มงูไม่มีพิษ', 26: 'งูหมอก กลุ่มงูพิษอ่อน', 27: 'งูหลาม กลุ่มงูไม่มีพิษ', 28: 'งูเขียว กลุ่มงูพิษอ่อน', 29: 'งูเขียวกาบหมาก กลุ่มงูไม่มีพิษ', 30: 'งูเขียวปากจิ้งจก กลุ่มงูพิษอ่อน', 31: 'งูเขียวพระอินทร์ กลุ่มงูพิษอ่อน', 32: 'งูเขียวหางเทา กลุ่มงูมีพิษ', 33: 'งูเขียวหางไหม้ตาโต กลุ่มงูมีพิษ', 34: 'งูเขียวหางไหม้ท้องเขียวใต้ กลุ่มงูมีพิษ', 35: 'งูเขียวหางไหม้ท้องเหลือง กลุ่มงูมีพิษ', 36: 'งูเหลือม กลุ่มงูไม่มีพิษ', 37: 'งูเห่า กลุ่มงูมีพิษ', 38: 'งูแมวเซา กลุ่มงูมีพิษ', 39: 'งูแม่ตะงาวรังนก กลุ่มงูพิษอ่อน', 40: 'งูแสงอาทิตย์ กลุ่มงูไม่มีพิษ', 41: 'งูแส้หางม้า กลุ่มงูพิษอ่อน', 42: 'จงอาง กลุ่มงูมีพิษ'}


model = load_model('viper.h5')

model.make_predict_function()


def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i)
    i = i.reshape(1, 224, 224, 3)
    predict_x = model.predict(i)
    classes_x = np.argmax(predict_x, axis=1)
    return dic[classes_x[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/about")
def about_page():
    return "Please subscribe  Artificial Intelligence Hub..!!!"


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        p = predict_label(img_path)

    return render_template("index.html", prediction=p, img_path=img_path, prediction2=p.split(" "))


if __name__ == '__main__':
    #app.debug = True
    app.run(debug=True)
