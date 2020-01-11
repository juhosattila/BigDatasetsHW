from DataLoader import PascalDataLoader
from NeuralNetworks import InceptionNeuralNetwork

bath_size = 32

train_iterator, valid_iterator, test_iterator = PascalDataLoader(minidataset=True).get_train_valid_test_iterators(
    img_target_size=InceptionNeuralNetwork.IMG_TARGET_SIZE, batch_size=bath_size
)

nn = InceptionNeuralNetwork(output_target_size=20)
nn.summary()
nn.fit_generator(train_iterator, valid_iterator)
nn.evaluate_generator(test_iterator)
