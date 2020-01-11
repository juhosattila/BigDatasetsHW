import urllib.request as ure
import os
import tarfile
from xml.dom import minidom
import pandas as pd

class PascalDataLoader():
    def __init__(self, minidataset: bool):
        if minidataset:
            self.dataset_url = 'http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar'
            self.inner_path = 'VOCdevkit/VOC2007'
        else:
            self.dataset_url = 'http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar'
            self.inner_path = 'VOCdevkit/VOC2012'

        self.base_dir = './Pascal_dir'
        self.tarfile_name = 'Pascal_dir.tar'

    def __load_tarfile(self):
        # Check if data is already deleted by checking if inner path is present.
        if os.path.exists( os.path.join(self.dataset_url, self.inner_path) ):
            return

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        tarfile_path = os.path.join(self.base_dir, self.tarfile_name)
        ure.urlretrieve(self.dataset_url, filename=tarfile_path)

        tar_obj = tarfile.open(tarfile_path)
        tar_obj.extractall(path=self.base_dir)
        tar_obj.close()

    def load(self):
        self.__load_tarfile()

        image_dir = os.path.join(self.base_dir, self.inner_path, 'JPEGImages')

        annotations_dir = os.path.join(self.base_dir, self.inner_path, 'Annotations')
        annotations_files = os.listdir(annotations_dir)

        df = pd.DataFrame(columns=['filename', 'classes'])
        for file in annotations_files:
            file_dom = minidom.parse(os.path.join(annotations_dir, file))
            file_name = file_dom.getElementsByTagName('filename')[0]
            labels = []
            for obj in file_dom.getElementsByTagName('object'):
                labels.append(obj.childNodes[1].childNodes[0].data)
            df.append([file_name, labels], ignore_index=True)

        return df