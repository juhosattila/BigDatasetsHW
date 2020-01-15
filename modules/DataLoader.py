import urllib.request as ure
import os
import tarfile
from xml.dom import minidom
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

class PascalDataLoader:
    NUMBER_OF_CLASSES = 20

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
        self.data_to_single_class('cat')
        self.dataframe_prepared = True
        return

    def load(self):
        self.__load_tarfile()
        self.__prepare_dataframe()

        image_dir = os.path.join(self.base_dir, self.inner_path, 'JPEGImages')
        return image_dir, self.df

    def get_train_valid_test_dataframes(self, valid_split=0.1, test_split=0.1):
        image_dir, df = self.load()
        train_df, valid_test_df = train_test_split(df, test_size=valid_split + test_split, shuffle=True)
        valid_df, test_df = train_test_split(valid_test_df,
                                             test_size=(test_split / (valid_split + test_split)), shuffle=True)

        return image_dir, train_df, valid_df, test_df

    def get_train_valid_test_iterators(self, img_target_size, batch_size, valid_split=0.1, test_split=0.1):
        image_dir, train_df, valid_df, test_df = self.get_train_valid_test_dataframes(
            valid_split=valid_split, test_split=test_split)

        iterator_parameters = {'directory': image_dir,
                               'x_col': 'filename',
                               'y_col': 'classes',
                               'target_size': img_target_size,
                               'batch_size': batch_size}

        # The validation and train data generator applies some geometrical transformation to expand the
        # number of training samples.
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            shear_range=0.2,  zoom_range=0.2,
            horizontal_flip=True)
        train_iterator = train_datagen.flow_from_dataframe(
            train_df, **iterator_parameters)

        # The valid and test data generator won't apply transformations apart from preprocessing
        valid_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        valid_iterator = valid_test_datagen.flow_from_dataframe(
            valid_df, **iterator_parameters)
        test_iterator = valid_test_datagen.flow_from_dataframe(
            test_df, **iterator_parameters)

        return train_iterator, valid_iterator, test_iterator

    def __class_indices(self):
        # TODO is there a better way?
        train_iterator, valid_iterator, test_iterator = self.get_train_valid_test_iterators((299, 299), 32)
        return test_iterator.class_indices

    def decode_predictions(self, predictions, threshold=0.5):
        class_indices = self.__class_indices()
        result = []
        single_prediction = predictions[0]  # TODO extend the function to handle multiple predictions
        for label, prediction in zip(class_indices, single_prediction):
            if prediction >= threshold:
                result.append((label, prediction))
        result = sorted(result, key=lambda tup: tup[1], reverse=True)
        return result
    
    #Images in different classes
    def statistics_classes(self):

        df = self.df

        # merge df classes to one array
        merged_list = []
        for l in df['classes']:
          merged_list += l

        unique, counts = np.unique(merged_list, return_counts=True)

        no_of_different_classes = len(counts)
        image_classes = []
        for i in range(0, no_of_different_classes):
          image_classes.append(unique[i])

        #plotting (classes - no. of images in class) 
        label_loc_x = np.arange(len(image_classes)) # label location on the bar

        plt.bar(label_loc_x, counts, width=0.8)
        plt.xticks(label_loc_x, image_classes, rotation='vertical')
        plt.xlabel('Classes')
        plt.ylabel('No. of images')
        plt.title("No. of images in one class")
        plt.show()
        return

    # Different labels on the image
    def statistics_labels(self):

        df = self.df

        label_count_list = []
        for l in df['classes']:
          label_count_list.append(len(l))

        no_of_labels_in_image, label_counts = np.unique(label_count_list, return_counts=True)

        #plotting (no of labels in image - no. of images) 
        label_loc_x = np.arange(len(no_of_labels_in_image)) # label location on the bar

        plt.bar(label_loc_x, label_counts, width=0.8)
        plt.xticks(label_loc_x, no_of_labels_in_image)
        plt.xlabel('No. of labels in image')
        plt.ylabel('No. of images')
        plt.title("Label - image")

        for a,b in enumerate(label_counts):
          plt.text(a - len(str(b))/16, b + 20, str(b), color='green')
          #plt.text(rect.get_width() + rect.get_width()/2.0, b + 0.5, str(b))

        plt.show()
        return
    
    #Clean data (no person)
    def clean_data(self):
        df = self.df
        #DELETE ALL PERSON LABELS

        for index, row in df.iterrows():
          if('person' in row['classes']):
            row['classes'].remove('person')
        
        #DELETE ALL IMAGES WITH NO LABELS
        df2 = df.copy()

        for index, row in df2.iterrows():
          l = row['classes']
          if not l:
            df = df.drop(index)
        
        self.df = df
        return

    def data_to_single_class(self, single_class):
        for index, row in self.df.iterrows():
            if single_class in row['classes']:
                row['classes'] = [single_class]
            else:
                row['classes'] = []

