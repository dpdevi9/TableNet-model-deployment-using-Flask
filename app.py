from flask import Flask, jsonify, request, redirect, render_template, url_for
import flask
import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K



app = Flask(__name__)
# app configs
app.config['IMAGE_UPLOADS'] = "im_uploads/"

def getTableNet():

    tf.keras.backend.clear_session()

    class table_mask(Layer):

        def __init__(self):
            super().__init__()
            self.conv_7 = Conv2D(kernel_size=(1,1), filters=128, kernel_regularizer=tf.keras.regularizers.l2(0.002))
            self.upsample_pool4 = UpSampling2D(size=(2, 2), interpolation='bilinear')
            self.upsample_pool3 = UpSampling2D(size=(2, 2), interpolation='bilinear')
            self.upsample_final = Conv2DTranspose(filters=2, kernel_size=3, strides=2, padding='same', activation='softmax')

        def call(self, input, pool3, pool4):
            
            x = self.conv_7(input)
            x = self.upsample_pool4(x)
            x = Concatenate()([x, pool4])
        
            x = self.upsample_pool3(x)
            x = Concatenate()([x, pool3])
            
            x = UpSampling2D((2,2))(x)
            x = UpSampling2D((2,2))(x)

            x = self.upsample_final(x)

        

            return x

    class col_mask(Layer):
        
        def __init__(self):
            super().__init__()
            self.conv_7 = Conv2D(kernel_size=(1,1), filters=128, kernel_regularizer=tf.keras.regularizers.l2(0.004), kernel_initializer='he_normal',)
            self.drop = Dropout(0.8)
            self.conv_8 = Conv2D(kernel_size=(1,1), filters=128, kernel_regularizer=tf.keras.regularizers.l2(0.004), kernel_initializer='he_normal',)
            self.upsample_pool4 = UpSampling2D(size=(2, 2), interpolation='bilinear')
            self.upsample_pool3 = UpSampling2D(size=(2, 2), interpolation='bilinear')
            self.upsample_final = Conv2DTranspose(filters=2, kernel_size=3, strides=2, padding='same', activation='softmax')

        def call(self, input, pool3, pool4):
            
            x = self.conv_7(input)
            x = self.drop(x)
            x = self.conv_8(x)

            x = self.upsample_pool4(x)
            x = Concatenate()([x, pool4])
        
            x = self.upsample_pool3(x)
            x = Concatenate()([x, pool3])
            
            x = UpSampling2D((2,2))(x)
            x = UpSampling2D((2,2))(x)

            x = self.upsample_final(x)

            

            return x



    input_shape = (1024, 1024, 3)
    input_ = Input(shape=input_shape)

    vgg19_ = VGG19(
        include_top=False,
        weights="imagenet",
        input_tensor=input_,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )

    for layer in vgg19_.layers:
        layer.trainable = False

    pool3 = vgg19_.get_layer('block3_pool').output
    pool4 = vgg19_.get_layer('block4_pool').output

    conv_1_1_1 = Conv2D(filters=128, kernel_size=(1, 1), activation='relu', name="block6_conv1", kernel_regularizer=tf.keras.regularizers.l2(0.004))(vgg19_.output)
    conv_1_1_1_drop = Dropout(0.8)(conv_1_1_1)

    conv_1_1_2 = Conv2D(filters=128, kernel_size=(1, 1), activation='relu', name="block6_conv2", kernel_regularizer=tf.keras.regularizers.l2(0.004))(conv_1_1_1_drop)
    conv_1_1_2_drop = Dropout(0.8)(conv_1_1_2)

    table_mask = table_mask()(conv_1_1_2_drop, pool3, pool4)
    col_mask = col_mask()(conv_1_1_2_drop, pool3, pool4)

    model = Model(input_, [table_mask, col_mask])

    model.load_weights('model/table_net.h5')

    return model



@app.route("/")
def hello():
    return "hello"


## upload image page
@app.route('/upload')
def upload():
    return flask.render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['filename']
    image.save(os.path.join(app.config['IMAGE_UPLOADS'], image.filename))
    image_path = "im_uploads/" + image.filename

    model = getTableNet()

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image,(1024, 1024), cv2.INTER_NEAREST)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    table_mask_pred, col_mask_pred = model.predict(image)

    table_mask_pred = tf.argmax(table_mask_pred, axis=-1)
    table_mask_pred = table_mask_pred[..., tf.newaxis][0]
    print("mask shape {}".format(table_mask_pred.shape))

    col_mask_pred = tf.argmax(col_mask_pred, axis=-1)
    col_mask_pred = col_mask_pred[..., tf.newaxis][0]

    im=tf.keras.preprocessing.image.array_to_img(image[0])
    im.save("output/" + 'image.png')

    im=tf.keras.preprocessing.image.array_to_img(table_mask_pred)
    im.save("output/" + 'table_mask_pred.png')

    im=tf.keras.preprocessing.image.array_to_img(col_mask_pred)
    im.save("output/" + 'col_mask_pred.png')

    img_org = Image.open("output/" + './image.png')
    table_mask = Image.open("output/" + './table_mask_pred.png')
    col_mask = Image.open("output/" + './col_mask_pred.png')

    # convert images
    img_mask = table_mask.convert('L')

    # grayscale
    # add alpha channel
    img_org.putalpha(img_mask)

    # save as png which keeps alpha channel
    img_org.save("output/"+'output.png')

    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    text = pytesseract.image_to_string(Image.open("output/"+'./output.png'), lang='eng' )

    return flask.render_template("result.html", text=text)


if __name__ == "__main__":
    app.run(debug=True)