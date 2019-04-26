from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
import numpy as np
from keras.preprocessing import image
##from keras.utils.np_utils import probas_to_classes

model=Sequential()
model.add(Convolution2D(32, (3,3), input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.8))

model.add(Dense(2))
model.add(Activation('sigmoid'))

train_datagen=ImageDataGenerator(rescale=1./255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
'training',
target_size=(64,64),
classes=['apple','orange'],
batch_size=32,
class_mode='categorical',
shuffle=True)

validation_generator=test_datagen.flow_from_directory(
'test',
target_size=(64, 64),
classes=['apple','orange'],
batch_size=32,
class_mode='categorical',
shuffle=True)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#early_stopping=EarlyStopping(monitor='val_loss', patience=2)
model.fit_generator(train_generator, steps_per_epoch=28, validation_data=validation_generator,validation_steps=304/32, nb_epoch=50)

json_string=model.to_json()
open(r'\mnistcnn_arc.json','w').write(json_string)
model.save_weights(r'\mnistcnn_weights.h5')
score=model.evaluate_generator(validation_generator, 160/32)

print('Test score:', score[0])
print('Test accuracy:', score[1])
import cv2
cam=cv2.VideoCapture(0)
cv2.namedWindow("test", cv2.WINDOW_NORMAL)
img_counter = 0
def test1(img_path):
        img = image.load_img(img_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        y_proba = model.predict(x)
        print(y_proba)
        y_classes = model.predict_classes(x)
        print(y_classes)
#        if y_proba[0][0] == 1:
#                
#                print("apple")
#
#        else:
#                print("orange")
#        print(y_proba[0][0])

while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
                break
        k = cv2.waitKey(1)

        if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
        elif k%256 == 32:
                # SPACE pressed
                img_path = ("fruit.png")
                cv2.imwrite(img_path, frame)
                print("{} written!".format(img_path))
test1("fruit.png")
cam.release()
cv2.destroyAllWindows()



if(y_proba[0]>y_proba[1]):
                print('Apple')
elif(y_proba[1]>y_proba[0]):
                print('Orange')
else :
                print('Nothing')
y_classes = probas_to_classes(y_proba)
print(train_generator.class_indices)

