

# import numpy as np
# import cv2
#
# if __name__ == "__main__":
#     # faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#     # faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#     # eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#     img = cv2.imread('data/abba.png')
#     # cv2.imshow()
#     # print(img)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # # print (gray)
#     cv2.imshow('img', gray)
#     cv2.waitKey(0)
#     # # faces = face_cascade.detectMultiScale(gray, .1, 10)
#     # faces = faceCascade.detectMultiScale(
#     #     gray,
#     #     scaleFactor=1.1,
#     #     minNeighbors=5,
#     #     minSize=(30, 30),
#     #     # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
#     # )
#     # # for (x,y,w,h) in faces:
#     # #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     # #     roi_gray = gray[y:y+h, x:x+w]
#     # #     roi_color = img[y:y+h, x:x+w]
#     # #     # eyes = eye_cascade.detectMultiScale(roi_gray)
#     # #     # for (ex,ey,ew,eh) in eyes:
#     # #     #     cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#     # # cv2.imshow('img',img)
#     # # cv2.waitKey(0)
#     # # cv2.destroyAllWindows()
#     # print('hello')

from keras.models import Model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
# import cv2
# from keras_applications import inception_v3
# import numpy as np

def vgg_face(weights_path=None):
    img = Input(shape=(3, 224, 224))

    pad1_1 = ZeroPadding2D(padding=(1, 1))(img)
    conv1_1 = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(pad1_1)
    pad1_2 = ZeroPadding2D(padding=(1, 1))(conv1_1)
    conv1_2 = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(pad1_2)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2)

    pad2_1 = ZeroPadding2D((1, 1))(pool1)
    conv2_1 = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(pad2_1)
    pad2_2 = ZeroPadding2D((1, 1))(conv2_1)
    conv2_2 = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(pad2_2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2_2)

    pad3_1 = ZeroPadding2D((1, 1))(pool2)
    conv3_1 = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(pad3_1)
    pad3_2 = ZeroPadding2D((1, 1))(conv3_1)
    conv3_2 = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(pad3_2)
    pad3_3 = ZeroPadding2D((1, 1))(conv3_2)
    conv3_3 = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(pad3_3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3_3)

    pad4_1 = ZeroPadding2D((1, 1))(pool3)
    conv4_1 = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(pad4_1)
    pad4_2 = ZeroPadding2D((1, 1))(conv4_1)
    conv4_2 = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(pad4_2)
    pad4_3 = ZeroPadding2D((1, 1))(conv4_2)
    conv4_3 = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(pad4_3)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4_3)

    pad5_1 = ZeroPadding2D((1, 1))(pool4)
    conv5_1 = Convolution2D(512, 3, 3, activation='relu', name='conv5_1')(pad5_1)
    pad5_2 = ZeroPadding2D((1, 1))(conv5_1)
    conv5_2 = Convolution2D(512, 3, 3, activation='relu', name='conv5_2')(pad5_2)
    pad5_3 = ZeroPadding2D((1, 1))(conv5_2)
    conv5_3 = Convolution2D(512, 3, 3, activation='relu', name='conv5_3')(pad5_3)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5_3)

    flat = Flatten()(pool5)
    fc6 = Dense(4096, activation='relu', name='fc6')(flat)
    fc6_drop = Dropout(0.5)(fc6)
    fc7 = Dense(4096, activation='relu', name='fc7')(fc6_drop)
    fc7_drop = Dropout(0.5)(fc7)
    out = Dense(2622, activation='softmax', name='fc8')(fc7_drop)

    model = Model(input=img, output=out)

    if weights_path:
        model.load_weights(weights_path)

    return model


def detect_faces(f_cascade, colored_img, scaleFactor=1.1):
    # just making a copy of image passed, so that passed image is not changed
    img_copy = colored_img.copy()

    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)

    # go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img_copy


if __name__ == "__main__":
    import cv2
    img = cv2.imread('data/abba.png')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    haar_face_cascade = cv2.CascadeClassifier('venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
    faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    # print the number of faces found
    print('Faces found: ', len(faces))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('gray', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # from keras.applications import vgg16, inception_v3, resnet50, mobilenet
    # # Load the VGG model
    # # vgg_model = vgg16.VGG16(weights='imagenet')
    # # print (dir(inception_v3))
    # model = inception_v3.InceptionV3()
    # # im = Image.open('data/GoT-920x564.jpg')
    # # # im = cv2.imread('data/abba.png')
    # #
    # # im = im.resize((224, 224))
    # # im = np.array(im).astype(np.float32)
    # # #    im[:,:,0] -= 129.1863
    # # #    im[:,:,1] -= 104.7624
    # # #    im[:,:,2] -= 93.5940
    # # im = im.transpose((2,0,1))
    # # im = np.expand_dims(im, axis=0)
    # #
    # # # Test pretrained model
    # # model = vgg_face('vgg-face-keras-fc.h5')
    # # out = model.predict(im)
    # # print(out[0][0])
    #
    # from keras.preprocessing.image import load_img
    # from keras.preprocessing.image import img_to_array
    # from keras.applications.imagenet_utils import decode_predictions
    # import matplotlib.pyplot as plt
    # # % matplotlib
    # # inline
    #
    # filename = 'data/GoT-920x564.jpg'
    # # load an image in PIL format
    # original = load_img(filename, target_size=(224, 224))
    # print('PIL image size', original.size)
    # plt.imshow(original)
    # plt.show()
    #
    # # convert the PIL image to a numpy array
    # # IN PIL - image is in (width, height, channel)
    # # In Numpy - image is in (height, width, channel)
    # numpy_image = img_to_array(original)
    # plt.imshow(np.uint8(numpy_image))
    # plt.show()
    # print('numpy array size', numpy_image.shape)
    #
    # # Convert the image / images into batch format
    # # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    # image_batch = np.expand_dims(numpy_image, axis=0)
    # print('image batch size', image_batch.shape)
    # plt.imshow(np.uint8(image_batch[0]))


