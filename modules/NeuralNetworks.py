import os
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import Adam, SGD
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

from abc import ABC, abstractmethod
import json

from modules.Metrics import *

models_folder = 'BigDatasetsHW/Models/'


def get_model_file_path(model_name):
    return os.path.join(models_folder, model_name + '.h5')
def get_model_decoding_path(model_name):
    return os.path.join(models_folder, model_name + '.json')

class AbstractNeuralNetwork(ABC):

    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def fit_generator(self, train_generator_iterator, validation_generator_iterator):
        pass

    @abstractmethod
    def evaluate_generator(self, test_generator_iterator):
        pass

    @abstractmethod
    def save(self, model_name):
        pass

    @abstractmethod
    def load(self, model_name):
        pass

    @abstractmethod
    def predict(self, x):
        pass


class InceptionNeuralNetwork(AbstractNeuralNetwork):
    IMG_TARGET_SIZE = (299, 299)

    def __init__(self, output_target_size, supplement_model):
        # Let us download the InceptionV3 network without the top part.
        self.base_model = InceptionV3(weights='imagenet', include_top=False)
        predictions = Dense(output_target_size, activation='sigmoid') \
            (supplement_model(self.base_model.output))
        self.model = Model(inputs=self.base_model.input, outputs=predictions)

        self.model_file_name = 'model.hdf5'
        self.metrics = ['categorical_accuracy']
        self.__compile(optimizer=Adam(lr=0.25))

        self.class_indices = None

    def __compile(self, optimizer):
        self.model.compile(optimizer=optimizer,
                           metrics=self.metrics,
                           loss='binary_crossentropy')

    def __load_model(self, model_name=None):
        if model_name is None:
            model_name = self.model_file_name
        custom_objects = {
            'recall_m': recall_m,
            'precision_m': precision_m,
            'f1_m': f1_m}

        self.model = load_model(model_name, custom_objects=custom_objects)

    def save(self, model_name):
        self.model.save(get_model_file_path(model_name))

        class_file_name = get_model_decoding_path(model_name)
        with open(class_file_name, 'w') as f:
            json.dump(self.class_indices, f)

    def load(self, model_name):
        self.__load_model(get_model_file_path(model_name))

        class_file_name = get_model_decoding_path(model_name)
        with open(class_file_name, 'r') as f:
            self.class_indices = json.load(f)

    def predict(self, x):
        return self.model.predict(x)

    def summary(self):
        print(self.model.summary())

    def fit_generator(self, train_generator_iterator, validation_generator_iterator):
        # Freeze the base model layers and train only the newly added layers.
        for layer in self.base_model.layers:
            layer.trainable = False

        self.__compile(optimizer=Adam(lr=0.25))

        # We are going to use early stopping and model saving-reloading mechanism.
        checkpointer = ModelCheckpoint(filepath=self.model_file_name, save_best_only=True, verbose=1)
        earlystopping = EarlyStopping(patience=2, verbose=1)

        # After all, train the upper layers of the  network.
        self.model.fit_generator(train_generator_iterator,
                                 epochs=1,
                                 validation_data=validation_generator_iterator,
                                 shuffle=True,
                                 callbacks=[checkpointer, earlystopping])
        # Reload the model.
        self.__load_model()

        # Now, unfreeze the upper part of the convolutional layers.
        # As, many things, this idea was also adopted from the corresponding seminar.
        for layer in self.model.layers[:172]:
            layer.trainable = False
        for layer in self.model.layers[172:]:
            layer.trainable = True

        self.__compile(optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True))

        self.model.fit_generator(train_generator_iterator,
                                 epochs=1,
                                 validation_data=validation_generator_iterator,
                                 shuffle=True,
                                 callbacks=[checkpointer, earlystopping])

        self.__load_model()

        # Retrieve class indices.
        self.class_indices = train_generator_iterator.class_indices

    def evaluate_generator(self, test_generator_iterator):
        # Evaluate the model on the test set.
        print(self.model.metrics_names)
        print(self.model.evaluate_generator(test_generator_iterator))

    def set_metrics(self, *args):
        self.metrics.clear()
        self.metrics.extend(args)

    def decode_predictions(self, predictions, threshold=0.5):
        class_indices = self.class_indices
        full_result = []
        for single_prediction in predictions:
            result = []
            for label, prediction in zip(class_indices, single_prediction):
                if prediction >= threshold:
                    result.append((label, prediction))
            result = sorted(result, key=lambda tup: tup[1], reverse=True)
            full_result.append(result)
        return full_result


class InceptionNeuralNetwork1(InceptionNeuralNetwork):
    def __init__(self, output_target_size):
        super().__init__(output_target_size=output_target_size,
                         supplement_model=self.__supplement_model)
        self.set_metrics('categorical_accuracy', recall_m, precision_m, f1_m)

    def __supplement_model(self, base_output):
        x = base_output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        return x


class InceptionNeuralNetwork2(InceptionNeuralNetwork):
    def __init__(self, output_target_size):
        super().__init__(output_target_size=output_target_size,
                         supplement_model=self.__supplement_model)

    def __supplement_model(self, base_output):
        x = base_output
        x = GlobalAveragePooling2D()(x)
        x = Dense(20, activation='relu')(x)
        return x
