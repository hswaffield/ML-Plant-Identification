from keras.models import load_model
from PIL import Image
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('tf')
# ...weird... ^^ I don't think this is actually necessary
import csv

from os import listdir
from os.path import isfile, join


TEST_FILES = [f for f in listdir('test') if isfile(join('test', f)) and f.endswith(".png")]

IMAGE_DIM = 150
# needs to match the size of the trained model input...

# toggle this to get new models...
pre_trained_file = 'first_try.h5'

# first try shape:
# IMAGE_DIM = 150
# BATCH_SIZE = 70
# NUM_CLASS = 12

model = load_model(pre_trained_file)

print(model.summary())

# Output classes:
classes = 12
classes_dict = {0: 'Black-grass', 1: 'Charlock', 2: 'Cleavers', 3: 'Common Chickweed', 4: 'Common wheat', 5: 'Fat Hen', 6: 'Loose Silky-bent', 7: 'Maize', 8: 'Scentless Mayweed', 9:'Shepherds Purse', 10:'Small-flowered Cranesbill', 11: 'Sugar beet'}

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


# where the predictions are actually generated:
with open('submissions.csv','w') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['file', 'species'])

    for f in TEST_FILES:
        # print(f)
        image = get_image(str('test/'+f))
        pred = model.predict(np.expand_dims(image, axis=0))

        pred = pred[0]

        pred_index = 0
        max = 0
        for i in range(classes):
            if pred[i] > max:
                max = pred[i]
                pred_index = i
        result = [f, classes_dict[pred_index]]
        print(result)
        csv_writer.writerow(result)





