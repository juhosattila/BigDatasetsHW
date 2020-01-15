from keras.layers import GlobalAveragePooling2D, Dense

from modules.DataLoader import PascalDataLoader
from modules.NeuralNetworks import InceptionNeuralNetwork

from modules.NeuralNetworks import InceptionNeuralNetwork1

bath_size = 32

train_iterator, valid_iterator, test_iterator = PascalDataLoader(minidataset=True).get_train_valid_test_iterators(
    img_target_size=InceptionNeuralNetwork.IMG_TARGET_SIZE, batch_size=bath_size
)
print('\nImages loaded and processed.\n'
      )
nn = InceptionNeuralNetwork1(output_target_size=PascalDataLoader.NUMBER_OF_CLASSES)
print('\nTransformed InceptionV3 loaded.\n')

#nn.summary()
print('\nStarting learing.\n')
nn.fit_generator(train_iterator, valid_iterator)

print('\nSaving modell\n')
nn.save('model365')

print('\nEvaluation')
nn.evaluate_generator(test_iterator)