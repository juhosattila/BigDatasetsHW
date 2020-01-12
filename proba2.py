from keras.layers import GlobalAveragePooling2D, Dense

from modules.DataLoader import PascalDataLoader
from modules.NeuralNetworks import InceptionNeuralNetwork

from modules.NeuralNetworks import InceptionNeuralNetwork1

bath_size = 32

train_iterator, valid_iterator, test_iterator = PascalDataLoader(minidataset=False).get_train_valid_test_iterators(
    img_target_size=InceptionNeuralNetwork.IMG_TARGET_SIZE, batch_size=bath_size
)

nn = InceptionNeuralNetwork1(output_target_size=PascalDataLoader.NUMBER_OF_CLASSES)

nn.summary()
nn.fit_generator(train_iterator, valid_iterator)
nn.evaluate_generator(test_iterator)