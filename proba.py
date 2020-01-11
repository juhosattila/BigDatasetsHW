from DataLoader import PascalDataLoader

from NeuralNetworks import InceptionNeuralNetwork

bath_size = 32

train, valid, test = PascalDataLoader(minidataset=True).get_train_valid_test_iterators(
    img_target_size=InceptionNeuralNetwork.IMG_TARGET_SIZE, batch_size=bath_size
)



