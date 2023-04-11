import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds  
import tensorflow_addons as tfa 
import keras.utils as image
from tensorflow import keras
import os


def main():
    #loads the cast_vs_dogs.h5 model
    model = keras.models.load_model('cats_vs_dogs.h5')  

    directory = input('Specify path to test images: $ ')
    for filename in os.scandir(directory):
        if filename.is_file():
            image_path = filename.path
            img = image.load_img(image_path, target_size=(300,300))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            image_tensor = np.vstack([x])
            classes = model.predict(image_tensor)
            if classes[0]>0.5:
                print(f"The image file {image_path} is an image of a human.")
            else:
                print(f"The image file {image_path} is an image of a cat.")

if __name__ == "__main__":
    main()