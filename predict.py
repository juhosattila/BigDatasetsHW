from modules.DataLoader import PascalDataLoader
from modules.NeuralNetworks import *
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions as imagenet_utils_decode_predictions
from keras.applications.inception_v3 import preprocess_input
import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument('--image-path', action="store",
                    dest="image_path", default=None)
parser.add_argument('--model', action="store",
                    dest="model_name", default=None)
parser.add_argument('--weights', action="store",
                    dest="weights", default=None)

args = parser.parse_args()

img_path = args.image_path
if img_path is None:
    print("Image path must be specified.")
    exit()
print("Image path: " + img_path)

model_name = args.model_name  # InceptionV3 will be used if None
weights = args.weights
if model_name is not None and weights is None:
    print("Weights must be specified if model is specified")
    exit()

# Every input parameter has been validated

if model_name is None:
    print("InceptionV3 will be used as model with weights=imagenet")
    model = InceptionV3(include_top=True, weights='imagenet')
    decoder_function = imagenet_utils_decode_predictions
else:
    print(model_name + " will be used as model with weights=" + weights)
    ModelClass = getattr(importlib.import_module("modules.NeuralNetworks"), model_name)
    model = ModelClass(output_target_size=PascalDataLoader.NUMBER_OF_CLASSES)
    model.load(weights)
    pascal_data_loader = PascalDataLoader(minidataset=True)
    decoder_function = pascal_data_loader.decode_predictions

# model, decoder_function has been set accordingly to the command line parameters

img = image.load_img(img_path, target_size=InceptionNeuralNetwork.IMG_TARGET_SIZE)
input = image.img_to_array(img)
input = np.expand_dims(input, axis=0)
input = preprocess_input(input)

predictions = model.predict(input)
text_predictions = decoder_function(predictions)
print(text_predictions)
