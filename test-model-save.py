from modules.NeuralNetworks import InceptionNeuralNetwork2
from modules.DataLoader import PascalDataLoader

nn = InceptionNeuralNetwork2(output_target_size=PascalDataLoader.NUMBER_OF_CLASSES)
nn.save("test-model-save")
