import numpy as np
import keras.models
import sys
import base64
from keras.preprocessing import image
import numpy as np
import keras.models
from keras.models import model_from_json
import tensorflow as tf
sys.path.append(os.path.abspath("./model_saved"))


def getmodel():
    json_file = open('model_saved.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
	#load weights into new model
    loaded_model.load_weights("model_saved.h5")
    print("Loaded Model from disk")

	#compile and evaluate loaded model
    loaded_model.compile(loss ='binary_crossentropy',optimizer ='rmsprop',metrics =['accuracy'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	#graph = tf.get_default_graph()
    return loaded_model

def read_image(filename):
	img = keras.preprocessing.image.load_img(filename, target_size=(224,224))
	img_array = keras.preprocessing.image.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0)
	return img_array

@app.route("/predict",methods=['POST'])
def predict():
    model=getmodel()
    if request.method =='POST':
        file =request.files['photo']
        if file :
            filename=file.filename
            file.save(filename)
            img_array= read_image(filename)
            #prediction
            predictions = model.predict(img_array)
            score = predictions[0]
            if score==0:
                return render_template('blousedisp.html')
            else:
                return render_template('kurtidisp.html')
    render_template("predict.html")