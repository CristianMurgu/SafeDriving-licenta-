import cv2 as cv
import os
import graphviz


from tensorflow.python.keras.utils.version_utils import training

#from dataset_creator_down import to_gray

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #chill down the optimization statement )))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN / Intel math kernel

import tensorflow as tf

from tensorflow.keras import layers, models, regularizers, Input
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
import numpy as np

class MinAccFromEpchToEpch(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.monitor = 'accuracy'
        self.wait = 0
        self.stopped_epoch = 0
        self.mini = 0.001
        self.patience = 1
        self.prev_mini = -np.Inf

    def on_epoch_end(self, epoch, info = None):
        if info is not None:
            curr = info.get(self.monitor)
            if curr - self.prev_mini <self.mini:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
            else:
                self.wait = 0
            self.prev_mini = curr
        else:
            return
#print(tf.__version__)
def create_model():
    if not os.path.exists('eye_models/eye_closed_or_opened_classifier.keras'):
        photo_h = 90
        photo_w = 90
        model = models.Sequential([
            Input(shape=(photo_w, photo_h, 1)),

            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')

        ])
        total_files = 133103 + 41653
        weight_false_files = 1 / 41653 * (total_files / 2.0)
        weight_true_files = 1 / 133103 * (total_files / 2.0)
        class_weights = {0 : weight_false_files, 1 : weight_true_files}

        learnin_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = 0.01,
            decay_steps = 9999,
            decay_rate = 0.9
        )

        optimizer = Adam(learning_rate=0.0001)#SGD(learning_rate = learnin_rate_schedule)

        callbacks = [
            MinAccFromEpchToEpch(),
            #EarlyStopping(monitor='val_accuracy', patience=15, mode='max', restore_best_weights=True, min_delta=0.001),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
            #ModelCheckpoint('co_best_eye.keras', save_best_only=True, monitor='val_accuracy', save_weights_only=True)
        ]

        model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
        #model.summary()


        train_dataset = create_dataset('eye_data/closed_open/train', True)
        validation_dataset = create_dataset('eye_data/closed_open/val')
        test_dataset = create_dataset('eye_data/closed_open/test')

        epochs_count = 50#enough, 12 acc: 0.8989, 13 acc: 0.9007

        stats = model.fit(train_dataset, epochs = epochs_count, validation_data = validation_dataset, callbacks = callbacks, class_weight = class_weights)


        stats.history.keys()

        test_model = model.evaluate(test_dataset)

        print(f"test data: acc {test_model[1]}, prec: {test_model[2]}, recc: {test_model[3]}")
        os.makedirs('eye_models', exist_ok = True)
        model.save('eye_models/eye_closed_or_opened_classifier.keras')



def to_grayscale(img, label):
    img = tf.image.rgb_to_grayscale(img)
    return img / 255.0, label

def create_dataset(data_folder, isTraining = False):
    photo_h = 90
    photo_w = 90
    batch_size = 64

    #train_folder = 'eye_data/train'
    #test_folder = 'eye_data/test'

    dataset = tf.keras.utils.image_dataset_from_directory(data_folder, image_size = (photo_h, photo_w),
                                                             batch_size = batch_size, label_mode = 'binary',
                                                          shuffle = isTraining)

    dataset = dataset.map(to_grayscale) #normalize the data

    if isTraining:
        augmentation = tf.keras.Sequential([layers.RandomRotation(0.1), layers.RandomZoom(0.1),
                                            layers.RandomFlip('horizontal_and_vertical'), layers.RandomTranslation(0.25, 0.25)])
        dataset = dataset.map(lambda img, result: (augmentation(img, training = True), result))

    return dataset.prefetch(tf.data.AUTOTUNE)

create_model()


def get_model_path():
    return 'eye_models/eye_closed_or_opened_classifier.keras'

co_model = None
co_predict = None

def load_model_onetime():
    global co_model, co_predict
    if co_model is None:
        co_model = tf.keras.models.load_model(get_model_path())
        @tf.function(reduce_retracing = True)

        def aux_co_predict(x):
            return co_model(x)

        co_predict = aux_co_predict

        warmup_input = np.zeros((1, 90, 90, 1), dtype = np.float32)
        _ = co_predict(warmup_input)


def is_eye_opened(photo):
    load_model_onetime()

    photo = cv.resize(photo, (90, 90))
    photo = photo.astype(np.float32) / 255.0

    if len(photo.shape) == 2:
        photo = np.expand_dims(photo, axis=-1)

    photo_tens = tf.convert_to_tensor(photo[np.newaxis, ...])

    prediction = co_predict(photo_tens)

    print(f"prediction: {prediction[0][0]}")
    if prediction[0] < 0.7:
        return False
    else:
        return True

def ploting():
    model = models.load_model('eye_models/eye_closed_or_opened_classifier.keras')

    import visualkeras
    from PIL import ImageFont

    font = ImageFont.load_default()
    visualkeras.layered_view(
        model,
        legend=True,
        to_file='model_architecture.png',
        font=font,
        scale_xy=1,
        scale_z=1,
        max_z=100,
        max_xy=100
    )