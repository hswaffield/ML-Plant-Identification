from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import sys

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
# in first few iterations, we're only training on: 3830 images belonging to 12 classes.
# 63 - 73 % accuracy was success range, pre Titan XP use
# getting into upper val acc 80s... at around 76 iterations... then crashes down to 81%... but lower part looks
# 87405

#4750 now...

# TODO: ideas to tweak:
# actually feeding in all the data... X
# completely different cnn architecture...
# data augmentation ... more of it... at test time, so you can then average the samples...
# epoch count
# Image_dim / reading in / smart reading in
# test-time data augmentation??
#https://towardsdatascience.com/transfer-learning-using-keras-d804b2e04ef8


# best ones so far, in descending order:
# xception_seedling_cnn__model_1520027537.h5
# xception_seedling_cnn__model_1520015393.h5

# some people suggest using size = 299...
TEST_SET_SIZE = 4750
IMAGE_DIM = 400

# IMAGE_DIM = 299
# ^^ for use with vanilla xception
# BATCH_SIZE = 100
BATCH_SIZE = 5
NUM_CLASS = 12
NUM_EPOCHS = 100
CURRENT_TRAIN_SET = 'train'

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
        # image = square_image(image)
        image = image.resize((IMAGE_DIM, IMAGE_DIM))
        image = np.array(image) / 255

        # Convert gray-scale images to RGB
        if len(image.shape) == 2:
            image = np.dstack((image, image, image))

        return image.astype('float32')

# def square_image(image):
#     # TODO: Change to crop from the center
#     smallest_dimension = min(image.size)
#     return image.crop((0, 0, smallest_dimension, smallest_dimension))

def testing_procedure(model):
    print("testing model now...")
    submission_name = outfile("competition_submission", "csv")
    # num_output = outfile(("nums_predictions", "csv"))
    print("will write to: " + submission_name)
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
            csv_writer.writerow(result)

    # not clean to do same computations again:
    # TODO: write a new script to wrangle these files:
    # print("will write to: " + num_output)
    # with open(num_output, 'w') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow(['file', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
    #
    #     for f in TEST_FILES:
    #         image = get_image(str('test/' + f))
    #         pred = model.predict(np.expand_dims(image, axis=0))
    #         pred = pred.tolist()
    #         result = pred.insert(0, f)
    #         csv_writer.writerow(result)


def make_xception():
    return Xception(weights=None, classes=12, input_shape=(IMAGE_DIM, IMAGE_DIM, 3))

def make_inception_res_net(img_dim):
    model = InceptionV3(weights="imagenet", include_top=False, input_shape=(img_dim, img_dim, 3))

    # could also toggle to inception...

    # model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(img_dim, img_dim, 3))

    # Consider preventing the convolutions from training, if dataset is super small:
    # This would essentially make the new model a linear classifier on features extracted by the InceptionResNetV2
    # for layer in model.layers:
    #     layer.trainable = False

    # alternatively, only freeze a few (because these higher level features are definitely relevant):
    # for layer in model.layers[:5]:
    #     layer.trainable = False

    # other keras model api
    # Adding custom Layers
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(CLASSES, activation="softmax")(x)

    # creating the final model
    model_final = Model(input=model.input, output=predictions)

    return model_final


def main():
    global NUM_EPOCHS
    # dataaugmentation tools... lots of params to tweak.
    # less shift, vertical flips ok...
    train_datagen = ImageDataGenerator(
            rotation_range=90,
            width_shift_range=0.3,
            height_shift_range=0.3,
            rescale=1./255,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')

    # train_datagen = ImageDataGenerator(
    #         rescale=1./255,
    #         shear_range=0.2,
    #         zoom_range=0.2,
    #         horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)


    if len(sys.argv) > 1 and isfile(sys.argv[1]):
        print("loading and further training provided model.")
        model = load_model(sys.argv[1])

    else:
        # can try to use Xception model:

        print("no model provided, making a new one.")

        # model = make_xception()
        # might as well make the images bigger, right ?
        # model = make_inception_res_net(IMAGE_DIM)

        model = make_xception()


        # TODO: try transfer learning! using adifferent model, that is already trained

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()

    if len(sys.argv) > 2 and sys.argv[2] == "skip":
        testing_procedure(model)
        sys.exit()

    elif len(sys.argv) > 2 and sys.argv[2] != "skip":
        NUM_EPOCHS = int(sys.argv[2])

    # read in the input data:

    train_generator = train_datagen.flow_from_directory(
            CURRENT_TRAIN_SET,  # this is the target directory
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
            steps_per_epoch=TEST_SET_SIZE // BATCH_SIZE,
            epochs=NUM_EPOCHS,
            validation_data=validation_generator,
            validation_steps=800 // BATCH_SIZE)

    model_name = outfile("xception_seedling_cnn_", "h5")
    print("saving newest model to: " + model_name)
    model.save(model_name)  # always save your weights after training or during training


    # Then you might as well run the tests... right?
    # the following runs the tests:
    testing_procedure(model)


if __name__ == '__main__':
    main()
