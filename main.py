from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import PIL
import numpy as np
import cv2
import os
import sqlite3







app = Flask(__name__)


model = load_model('Model_2021_11_23_12_00_29.h5')





def predict_label(img_path):
	i = image.load_img(img_path, target_size=(28,28))
	i = image.img_to_array(i)
	i = i/255
	i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
	i = np.expand_dims(i, 0)
	a = model.predict(i)
	a = a.reshape(-1)
	a = a.tolist()
	b = a.copy()
	a.sort(reverse=True)
	p = b.index(a[0])
	return p


# routes
@app.route("/", methods=['GET', 'POST'])
def home():
	return render_template("index.html")

@app.route("/submit", methods = ['POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path ="static/" + img.filename
		img.save(img_path)
		db = sqlite3.connect("mnist.db")
		c = db.cursor()
		c.execute('create table if not exists image(img_file text,out int)')
		c.execute('select * from image')
		d = c.fetchall()
		if len(d)==0:
			p = predict_label(img_path)
			c.execute('insert into image values(?,?)', (img_path, p))
			db.commit()
			return render_template("index.html", prediction=p, img_path=img_path)
		else:
			for i in d:
				if i[0] == img_path:
					img_path,p = i
					db.commit()
					return render_template("index.html", prediction=p, img_path=img_path,res='Image Predicted Before')

		p = predict_label(img_path)
		c.execute('insert into image values(?,?)',(img_path,p))
		db.commit()
		db.close()
		return render_template("index.html", prediction=p, img_path=img_path)















if __name__ =='__main__':
	#app.debug = True
	app.run (port = 5000,debug=True)