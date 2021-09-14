import numpy as np
import os
import matplotlib.pyplot as plt
from imutils import paths
from sklearn import preprocessing

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Input, AveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.backend import shape
from tensorflow.python.keras.models import Model
from tensorflow.python.training.saving import checkpoint_options

dataset = 'C:/Users/hoya9/Desktop/Mask_detector/data'
imagePaths=list(paths.list_images(dataset))

data = []
labels = []

for i in imagePaths:
    label=i.split(os.path.sep)[-2]
    labels.append(label)
    image=load_img(i, target_size=(224,224))
    image=img_to_array(image)
    image=preprocess_input(image)
    data.append(image)

data = np.array(data, dtype='float32')
labels = np.array(labels)

lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42, stratify=labels)

aug = ImageDataGenerator(rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))

head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7,7))(head_model)
head_model = Flatten(name='Flatten_layer')(head_model)
head_model = Dense(128, activation='relu')(head_model)
head_model = Dropout(.5)(head_model)
head_model = Dense(3, activation='softmax')(head_model)

model = Model(inputs=base_model.input, outputs=head_model)

for layers in base_model.layers:
    layers.trainable = False

model.summary()

Epochs = 300
BS = 30

adam = Adam(learning_rate=(0.01))

model_callbacks = [
    EarlyStopping(monitor='loss', patience=30, mode='min', min_delta=0.0001),
]

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    aug.flow(X_train, y_train, batch_size=BS),
    steps_per_epoch=len(X_train) // BS,
    validation_data=(X_test, y_test),
    validation_steps=len(X_test) // 10,
    epochs=Epochs,
    callbacks=model_callbacks
)
model.save('mobilenet_v2.model')

plt.subplot(121)
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')

plt.subplot(122)
plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()