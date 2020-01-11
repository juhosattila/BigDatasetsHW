from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import Adam, SGD
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

from keras.metrics import top_k_categorical_accuracy

def top_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

class InceptionNeuralNetwork:
    IMG_TARGET_SIZE = (299, 299)

    def __init__(self, output_target_size):
        # Let us download the InceptionV3 network without the top part.
        base_model = InceptionV3(weights='imagenet', include_top=False)
        x = base_model.output
        # We supplement it with a few layers that have proven to be useful.
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(output_target_size, activation='sigmoid')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze the base model layers and train only the newly added layers.
        for layer in base_model.layers:
            layer.trainable = False

        # We use Adam, which is not all that sensitive to the learning rate.
        self.model.compile(optimizer=Adam(lr=0.05),
                           #metrics=[(lambda y_true, y_pred: top_k_categorical_accuracy(y_true, y_pred, k=2))],
                           metrics=[top_categorical_accuracy],
                           loss='binary_crossentropy')

    def summary(self):
        print(self.model.summary())

    def fit_generator(self, train_generator_iterator, validation_generator_iterator):
        # We are going to use early stopping and model saving-reloading mechanism.
        checkpointer = ModelCheckpoint(filepath='weights.hdf5', save_best_only=True, verbose=1)
        earlystopping = EarlyStopping(patience=2, verbose=1)

        # After all, train the upper layers of the  network.
        self.model.fit_generator(train_generator_iterator,
                                 epochs=5,
                                 validation_data=validation_generator_iterator,
                                 shuffle=True,
                                 callbacks=[checkpointer, earlystopping])
        # Reload the model.
        self.model = load_model('weights.hdf5')

        # Now, unfreeze the upper part of the convolutional layers.
        # As, many things, this idea was also adopted from the corresponding seminar.
        for layer in self.model.layers[:172]:
            layer.trainable = False
        for layer in self.model.layers[172:]:
            layer.trainable = True

        self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True),
                           # metrics=[(lambda y_true, y_pred: top_k_categorical_accuracy(y_true, y_pred, k=2))],
                           metrics=[top_categorical_accuracy],
                           loss='binary_crossentropy')

        self.model.fit_generator(train_generator_iterator,
                                 epochs=5,
                                 validation_data=validation_generator_iterator,
                                 shuffle=True,
                                 callbacks=[checkpointer, earlystopping])

        self.model = load_model('weights.hdf5')

    def evaluate_generator(self, test_generator_iterator):
        # Evaluate the model on the test set.
        print(self.model.metrics_names)
        print(self.model.evaluate_generator(test_generator_iterator))
