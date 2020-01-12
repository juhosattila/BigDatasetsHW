from keras.layers import GlobalAveragePooling2D, Dense

from DataLoader import PascalDataLoader
from NeuralNetworks import InceptionNeuralNetwork

from NeuralNetworks import InceptioNeuralNetwork1

bath_size = 32

train_iterator, valid_iterator, test_iterator = PascalDataLoader(minidataset=True).get_train_valid_test_iterators(
    img_target_size=InceptionNeuralNetwork.IMG_TARGET_SIZE, batch_size=bath_size
)


def supplement1(base_output):
    x = base_output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    return x


nn = InceptionNeuralNetwork(output_target_size=20,
                            supplement_model=supplement1)

nn.summary()
nn.fit_generator(train_iterator, valid_iterator)
nn.evaluate_generator(test_iterator)

nn1 = InceptioNeuralNetwork1(out_target_size=20)