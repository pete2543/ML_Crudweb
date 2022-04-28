from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

dic = {0: 'งูกะปะ', 1: 'งูกินทากหัวโหนก', 2: 'งูก้นขบหางแดง', 3: 'งูคออ่อนปากจงอย', 4: 'งูงวงช้าง', 5: 'งูชายธงหลังดำ', 6: 'งูดินบ้าน', 7: 'งูทับสมิงคลา', 8: 'งูทางมะพร้าว', 9: 'งูปลิง', 10: 'งูปล้องฉนวน', 11: 'งูปล้องทอง',   12: 'งูคออ่อนปากจงอย', 13: 'งูปากกว้างน้ำเค็ม', 14: 'งูปี่แก้ว', 15: 'งูพริกท้องแดง',
       16: 'งูพริกสีน้ำตาล', 17: 'งูพังกา', 18: 'งูม่านทอง', 19: 'งูลายสอ', 20: 'งูลายสาบ', 21: 'งูลายสาบตาโต', 22: 'งูสมิงทะเล', 23: 'งูสามเหลี่ยม', 24: 'งูสายม่าน', 25: 'งูสิงตาโต', 26: 'งูหมอก', 27: 'งูหลาม', 28: 'งูเขียว', 29: 'งูเขียวกาบหมาก', 30: 'งูเขียวปากจิ้งจก', 31: 'งูเขียวพระอินทร์', 32: 'งูเขียวหางเทา', 33: 'งูเขียวหางไหม้ตาโต', 34: 'งูเขียวหางไหม้ท้องเขียวใต้', 35: 'XXXXXXXXXXXXX', 36: 'งูเหลือม', 37: 'งูเห่า', 38: 'งูแมวเซา', 39: 'งูแม่ตะงาวรังนก', 40: 'งูแสงอาทิตย์', 41: 'งูแส้หางม้า', 42: 'จงอาง'}


model = load_model('viper.h5')

model.make_predict_function()


def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i)/255.0
    i = i.reshape(1, 224, 224, 3)
    p = model.predict_classes(i)
    return dic[p[0]]


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

    return render_template("index.html", prediction=p, img_path=img_path)


if __name__ == '__main__':
    #app.debug = True
    app.run(debug=True)
