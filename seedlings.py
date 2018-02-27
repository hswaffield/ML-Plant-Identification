from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# To enable little bonuses:
from time import time
from math import floor

# testing requirements:
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
import csv

# from keras import backend as K
# K.set_image_dim_ordering('tf')

# Do this down the line to see if accuracies get better.
# Can make this data set way bigger, or not

# eventually expend the training dataset, because otherwise data is being waisted... initially I eliminated some for a
# CV set

IMAGE_DIM = 128
BATCH_SIZE = 50
NUM_CLASS = 12

# Files to scan:
TEST_FILES = [f for f in listdir('test') if isfile(join('test', f)) and f.endswith(".png")]

# Output classes:
CLASSES = 12
CLASSES_DICT = {0: 'Black-grass', 1: 'Charlock', 2: 'Cleavers', 3: 'Common Chickweed', 4: 'Common wheat', 5: 'Fat Hen', 6: 'Loose Silky-bent', 7: 'Maize', 8: 'Scentless Mayweed', 9: 'Shepherds Purse', 10: 'Small-flowered Cranesbill', 11: 'Sugar beet'}


def outfile(dataname, extension):
    """ generates a filename for saving a model with a timehash """
    signature = floor(time())
    return '%s_model_%i.%s' % (dataname, signature, extension)

# Thanks Hans:
def get_image(path):
    with Image.open(path) as image:
        image = square_image(image)
        image = image.resize((IMAGE_DIM, IMAGE_DIM))
        image = np.array(image) / 255

        # Convert gray-scale images to RGB
        if len(image.shape) == 2:
            image = np.dstack((image, image, image))

        return image.astype('float32')

def square_image(image):
    # TODO: Change to crop from the center
    smallest_dimension = min(image.size)
    return image.crop((0, 0, smallest_dimension, smallest_dimension))

def testing_procedure(model):
    print("testing model now...")
    submission_name = outfile("competition_submission", "csv")
    with open(submission_name, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['file', 'species'])

        for f in TEST_FILES:
            image = get_image(str('test/' + f))
            pred = model.predict(np.expand_dims(image, axis=0))
            pred = pred[0]
            pred_index = 0
            max = 0
            for i in range(CLASSES):
                if pred[i] > max:
                    max = pred[i]
                    pred_index = i
            result = [f, CLASSES_DICT[pred_index]]
            print(result)
            csv_writer.writerow(result)

def main():

        # dataaugmentation tools... lots of params to tweak.
        # datagen = ImageDataGenerator(
        #         rotation_range=40,
        #         width_shift_range=0.2,
        #         height_shift_range=0.2,
        #         rescale=1./255,
        #         shear_range=0.2,
        #         zoom_range=0.2,
        #         horizontal_flip=True,
        #         fill_mode='nearest')

        train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1./255)

        # image = image.resize((IMAGE_DIM, IMAGE_DIM))

        model = Sequential()
        model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(IMAGE_DIM, IMAGE_DIM, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # the model so far outputs 3D feature maps (height, width, features)

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        # consider a not dumb image size here:
        model.add(Dense(IMAGE_DIM))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASS))
        model.add(Activation('softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


        # read in the input data:

        train_generator = train_datagen.flow_from_directory(
                'train',  # this is the target directory
                target_size=(IMAGE_DIM, IMAGE_DIM),  # all images will be resized to 150x150
                batch_size=BATCH_SIZE,
                class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

        # make a cv set? - validation generator
        validation_generator = test_datagen.flow_from_directory(
                'validation',
                target_size=(IMAGE_DIM, IMAGE_DIM),
                batch_size=BATCH_SIZE,
                class_mode='categorical')

        model.fit_generator(
                train_generator,
                steps_per_epoch=2000 // BATCH_SIZE,
                epochs=14,
                validation_data=validation_generator,
                validation_steps=800 // BATCH_SIZE)

        model_name = outfile("seedling_cnn_", "h5")

        model.save(model_name)  # always save your weights after training or during training


        # Then you might as well run the tests... right?
        # the following runs the tests:
        testing_procedure(model)


if __name__ == '__main__':
    main()
