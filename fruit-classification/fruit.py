#Bulinding the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model      
#Installing the CNN
classifier_an = Sequential()

#Step 1- Convolution
classifier_an.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation =
                    'relu'))
    
#step 2- Pooling
classifier_an.add(MaxPooling2D(pool_size = (2,2))) 

#step 3 -flattening
classifier_an.add(Flatten())      

#step 4- full connection
classifier_an.add(Dense(activation= 'relu', units=128))
classifier_an.add(Dense( activation = 'sigmoid',units=1))

#compiling CNN
classifier_an.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#part 2 fitting CNN to images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('fruits\Training',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('fruits\Test',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier_an.fit_generator(training_set,
                    steps_per_epoch=8000/32,
                    epochs=100,
                    validation_data=test_set,
                    validation_steps=2000)
classifier_an.save('classifier_an.h5') 
#Taking image as input using webcam
import cv2
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0
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
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
cam.release()
cv2.destroyAllWindows()

#testing the model
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(img_name, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier_an.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'apple'
    print(prediction)
else:
    prediction = 'orange' 
    print(prediction)
print(result[0][0])

classifier_an.save('classifier_an.h5') 
new_model = load_model('classifier_an.h5')
