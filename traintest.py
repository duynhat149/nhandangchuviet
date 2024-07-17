import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import ImageFile

# Thiết lập cho việc đọc ảnh bị cắt ngắn
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Đường dẫn tới dữ liệu ảnh
image_path = 'data/data'
models_path = 'models/saved_model.hdf5'
rgb = False
imageSize = 224

gestures = {
    'kh': '0', 'mo': '1', 'ha': '2', 'th': '3', 'bo': '4',
    'na': '5', 'sa': '6', 'ba': '7', 'ta': '8', 'ch': '9',
    'AA': 'A', 'BB': 'B', 'CC': 'C', 'DD': 'D', 'EE': 'E', 'FF': 'F',
    'GG': 'G', 'HH': 'H', 'II': 'I', 'JJ': 'J', 'KK': 'K', 'LL': 'L',
    'MM': 'M', 'NN': 'N', 'OO': 'O', 'PP': 'P', 'QQ': 'Q', 'RR': 'R',
    'SS': 'S', 'TT': 'T', 'UU': 'U', 'VV': 'V', 'WW': 'W', 'XX': 'X',
    'YY': 'Y', 'ZZ': 'Z', 'aa': 'a', 'bb': 'b', 'cc': 'c', 'dd': 'd',
    'ee': 'e', 'ff': 'f', 'gg': 'g', 'hh': 'h', 'ii': 'i', 'jj': 'j',
    'kk': 'k', 'll': 'l', 'mm': 'm', 'nn': 'n', 'oo': 'o', 'pp': 'p',
    'qq': 'q', 'rr': 'r', 'ss': 's', 'tt': 't', 'uu': 'u', 'vv': 'v',
    'ww': 'w', 'xx': 'x', 'yy': 'y', 'zz': 'z'
}

gestures_map = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15,
    'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21,
    'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27,
    'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33,
    'Y': 34, 'Z': 35, 'a': 36, 'b': 37, 'c': 38, 'd': 39,
    'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45,
    'k': 46, 'l': 47, 'm': 48, 'n': 49, 'o': 50, 'p': 51,
    'q': 52, 'r': 53, 's': 54, 't': 55, 'u': 56, 'v': 57,
    'w': 58, 'x': 59, 'y': 60, 'z': 61
}

gesture_names = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F',
    16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L',
    22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X',
    34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c', 39: 'd',
    40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
    46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p',
    52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v',
    58: 'w', 59: 'x', 60: 'y', 61: 'z'
}


# Hàm xử lý ảnh resize về 224x224 và chuyển về numpy array
def process_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_img = img[y:y + h, x:x + w]
        resized_img = cv2.resize(cropped_img, (imageSize, imageSize))
        return resized_img
    else:
        return None


# Xử lý dữ liệu đầu vào
def process_data(X_data, y_data):
    X_data = np.array(X_data, dtype='float32') / 255.0
    y_data = np.array(y_data)
    y_data = to_categorical(y_data)
    return X_data, y_data


# Hàm duyệt thư mục ảnh dùng để train
def walk_file_tree(image_path):
    X_data = []
    y_data = []
    for directory, subdirectories, files in os.walk(image_path):
        for file in files:
            if not file.startswith('.'):
                path = os.path.join(directory, file)
                gesture_name = gestures[file[7:9]]
                y_data.append(gestures_map[gesture_name])
                X_data.append(process_image(path))
            else:
                continue
    X_data, y_data = process_data(X_data, y_data)
    return X_data, y_data


# Load dữ liệu vào X và Y
X_data, y_data = walk_file_tree(image_path)

# Phân chia dữ liệu train và test theo tỷ lệ 80/20
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2,
                                                    random_state=12, stratify=y_data)
# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

# Đặt các checkpoint để lưu lại model tốt nhất
model_checkpoint = ModelCheckpoint(filepath=models_path, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10,
                               verbose=1, mode='auto',
                               restore_best_weights=True)

# Khởi tạo model
model1 = VGG16(weights='imagenet', include_top=False, input_shape=(imageSize,
                                                                   imageSize, 3))
optimizer1 = optimizers.Adam()
base_model = model1

# Thêm các lớp bên trênS
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
x = Dense(128, activation='relu', name='fc3')(x)
x = Dense(128, activation='relu', name='fc4')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', name='fc5')(x)

predictions = Dense(62, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Đóng băng các lớp dưới, chỉ train lớp bên trên mình thêm vào
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test),
                    verbose=1,
                    callbacks=[early_stopping, model_checkpoint])

# Lưu model đã train ra file
model.save('models/mymodel.h5')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
