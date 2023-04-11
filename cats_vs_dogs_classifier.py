import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import multiprocessing
from tensorflow import keras

class desired_accuracy(tf.keras.callbacks.Callback):
    def __init__(self, accuracy):
        self.accuracy = accuracy

    def on_epoch_end(self, epoch, logs={}):
        print(f"Accuracy: {logs.get('accuracy')}")
        if(logs.get('accuracy') >= self.accuracy):
            print(f"\nAchieved {self.accuracy} , terminating training...")
            self.model.stop_training = True

def create_model():
    model = keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', 
                               input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def read_tfrecord(example):
    features = {
        "image": tf.io.FixedLenFeature((), tf.string, ""),
        "label": tf.io.FixedLenFeature((), tf.int64, -1),
    }
    example = tf.io.parse_single_example(example, features)
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [300, 300])
    return image, example['label']

def main():
    print("[!] Configure training settings")
    desired_accuracy_input = (float(input("Desired accuracy: ")) / 100)
    accuracy_setting = desired_accuracy(
        accuracy = desired_accuracy_input
    )
    training_dataset = tfds.load('cats_vs_dogs', split='train', as_supervised=True)
    print("[!] Dataset loaded")
    print(training_dataset)
    #e.g. /$HOME/$USER/tensorflow_datasets/cats_vs_dogs/4.0.0/cats_vs_dogs-train.tfrecord*
    file_pattern = input('Enter file pattern: ')
    files = tf.data.Dataset.list_files(file_pattern)
    train_dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=4, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    cores = multiprocessing.cpu_count()
    print(cores)
    training_dataset = train_dataset.map(read_tfrecord, num_parallel_calls=cores)
    training_dataset = training_dataset.shuffle(1024).batch(32)
    training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    model = create_model()
    model.summary()
    model.fit(training_dataset, epochs=10, callbacks=[accuracy_setting], verbose=1)
    model.save('cats_vs_dogs.h5')

if __name__ == "__main__":
    main()