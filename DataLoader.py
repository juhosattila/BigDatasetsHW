import urllib.request as ure
import os
import tarfile
from xml.dom import minidom
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

class PascalDataLoader:
    def __init__(self, minidataset: bool):
        if minidataset:
            self.dataset_url = 'http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar'
            self.inner_path = 'VOCdevkit/VOC2007'
        else:
            self.dataset_url = 'http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar'
            self.inner_path = 'VOCdevkit/VOC2012'

        self.base_dir = './Pascal_dir'
        self.tarfile_name = 'Pascal_dir.tar'

        self.dataframe_prepared = False
        self.df = None

    def __load_tarfile(self):
        # Check if data is already downloaded by checking if inner path is present.
        if os.path.exists(os.path.join(self.base_dir, self.inner_path)):
            return

        self.dataframe_prepared = False
        self.df = None

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        tarfile_path = os.path.join(self.base_dir, self.tarfile_name)
        ure.urlretrieve(self.dataset_url, filename=tarfile_path)

        tar_obj = tarfile.open(tarfile_path)
        tar_obj.extractall(path=self.base_dir)
        tar_obj.close()

    def __prepare_dataframe(self):
        if self.dataframe_prepared:
            return

        annotations_dir = os.path.join(self.base_dir, self.inner_path, 'Annotations')
        annotations_files = os.listdir(annotations_dir)

        df_cols = ['filename', 'classes']
        df = pd.DataFrame(columns=df_cols)
        for file in annotations_files:
            file_dom = minidom.parse(os.path.join(annotations_dir, file))
            file_name = file_dom.getElementsByTagName('filename')[0].childNodes[0].data
            labels = set()
            for obj in file_dom.getElementsByTagName('object'):
                labels.add(obj.childNodes[1].childNodes[0].data)
            df = df.append(pd.Series([file_name, list(labels)], index=df_cols),
                           ignore_index=True)

        self.df = df
        self.dataframe_prepared = True
        return

    def load(self):
        self.__load_tarfile()
        self.__prepare_dataframe()

        image_dir = os.path.join(self.base_dir, self.inner_path, 'JPEGImages')
        return image_dir, self.df

    def get_train_test_dataframes(self, test_split=0.1):
        image_dir, df = self.load()
        train_df, test_df = train_test_split(df, test_size=test_split, shuffle=True)

        return image_dir, train_df, test_df

    def get_train_valid_test_iterators(self, img_target_size, batch_size, test_split=0.1):
        image_dir, train_df, test_df = self.get_train_test_dataframes(test_split=test_split)

        iterator_parameters = {'directory': image_dir,
                               'x_col': 'filename',
                               'y_col': 'classes',
                               'target_size': img_target_size,
                               'batch_size': batch_size}

        # The validation and train data generator applies some geometrical transformation to expand the
        # number of training samples.
        train_valid_datagen = ImageDataGenerator(
            rotation_range=20,
            brightness_range=[-0.1, 0.1],
            rescale=1./255,
            shear_range=0.2,  zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.1)

        train_iterator = train_valid_datagen.flow_from_dataframe(
            train_df, **iterator_parameters, subset='training')
        valid_iterator = train_valid_datagen.flow_from_dataframe(
            train_df, **iterator_parameters, subset='validation')

        # The test data generator won't apply transformations apart from
        # rescaling.
        test_datagen = ImageDataGenerator(rescale=1./255)

        test_iterator = test_datagen.flow_from_dataframe(
            test_df, **iterator_parameters)

        return train_iterator, valid_iterator, test_iterator
