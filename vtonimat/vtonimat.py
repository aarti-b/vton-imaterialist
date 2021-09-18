#import gzip
import os
import struct
from array import array
import random
import subprocess
import sys

_allowed_modes = (
    # integer values in {0..255}
    'standard',
)
_allowed_return_types = (
    'lists',
    'numpy',
)

np=None
cv2=None

#check packages installed otherwise install them

def _import_numpy():
    global np
    if np is None:
        try:
            import numpy as _np
        except ImportError as e:
            print(e)
        try:
            print('[INFO] Installing numpy package .....')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install',
            'numpy'])
        except:
            print('Please check Python version [env version ....]')
        np = _np
    else:
        pass
    return np

def _import_cv2():
    global cv2
    if cv2 is None:
        try:
            import cv2 as _cv2
        except ImportError as e:
            print(e)
        try:
            print('[INFO] Installing opencv package .....')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install',
            'opencv-python'])
        except:
            print('Please check Python version')
        cv2=_cv2
    else:
        pass
    return cv2

class SegData(object):

    def __init__(self, path='./convert_imaterialist', mode='standard', return_type='lists', gz=False, dataset='vton'):
        self.path = path

        assert mode in _allowed_modes, \
            "selected mode '{}' not in {}".format(mode,_allowed_modes)

        self._mode = mode

        assert return_type in _allowed_return_types, \
            "selected return_type '{}' not in {}".format(
                return_type,
                _allowed_return_types
            )

        self._return_type = return_type

        self.test_image_filename, self.test_label_filename, self.train_image_filename, self.train_label_filename=self.select_segdata(dataset)
        
        self.gz = gz

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

    def select_segdata(self, dataset):
        '''
        Select one of the binary segmentation dataset
        Available datasets:
            - vton     #TODO
            - imaterialist_top
            - imaterialist_bottom       #TODO
            - augmented vton+imaterialist_top     #TODO

        '''
        #template = 'bsegmentation-{0}-{1}-{2}-id3-ubyte'
        template = '{0}-{1}-{2}-idx3-ubyte'
        test_image_filename = template.format(dataset, 'test', 'images')
        test_label_filename = template.format(dataset, 'test', 'labels')

        train_image_filename = template.format(dataset, 'train', 'images')
        train_label_filename = template.format(dataset, 'train', 'labels')


        return test_image_filename, test_label_filename, train_image_filename, train_label_filename

    @property # read only because set only once, via constructor
    def mode(self):
        return self._mode

    @property # read only because set only once, via constructor
    def return_type(self):
        return self._return_type

    def load_testing(self):
        ims, labels = self.load(os.path.join(self.path, self.test_image_filename),
                                os.path.join(self.path, self.test_label_filename))

        self.test_images = self.process_images(ims)
        self.test_labels = self.process_labels(labels)

        return self.test_images, self.test_labels

    def load_training(self):
        ims, labels = self.load(os.path.join(self.path, self.train_image_filename),
                                os.path.join(self.path, self.train_label_filename))

        self.train_images = self.process_images(ims)
        self.train_labels = self.process_labels(labels)

        return self.train_images, self.train_labels

    def load_training_in_batches(self, batch_size):
        list_of_image_batch=[]
        list_of_label_batch=[]
        if type(batch_size) is not int:
            raise ValueError('batch_size must be a int number')
        batch_start_point = 0
        dataset_size=self._get_dataset_size(os.path.join(self.path, self.test_image_filename),
                           os.path.join(self.path, self.test_label_filename))
    
        num_of_batches = dataset_size//batch_size 
        count_batches = 0
        print('number of batches', num_of_batches)
        while(count_batches < num_of_batches):
            try:
                ims, labels = self.load(
                    os.path.join(self.path, self.test_image_filename),
                    os.path.join(self.path, self.test_label_filename),
                    batch=[batch_start_point, batch_size])

                self.train_images = self.process_images(ims)
                self.train_labels = self.process_labels(labels)

                count_batches += 1
                batch_start_point += batch_size
                list_of_image_batch.append(ims)
                list_of_label_batch.append(labels)
            except:
                print('Not Enough Examples')
                return list_of_image_batch, list_of_label_batch
        
        return list_of_image_batch, list_of_label_batch

    def _get_dataset_size(self, images_filepath, labels_filepath):

        with open(labels_filepath, 'rb') as file:
            magic, lb_size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))

        with open(images_filepath, 'rb') as file:
            magic, im_size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))

        if lb_size != im_size:
            raise ValueError('image size is not equal to label size')
        return lb_size

    def process_images(self, images):
        if self.return_type is 'lists':
            return self.process_images_to_lists_or_numpy(images, 'lists')
        elif self.return_type is 'numpy':
            return self.process_images_to_lists_or_numpy(images, 'numpy')
        else:
            print('Unknown return type')

    def process_labels(self, labels):
        if self.return_type is 'lists':
            return self.process_labels_to_lists_or_numpy(labels, 'lists')
        elif self.return_type is 'numpy':
            return self.process_labels_to_lists_or_numpy(labels, 'numpy')
        else:
            print('Unknown return type')

    def process_images_to_lists_or_numpy(self,images, ret_type):
        if self.mode == 'standard' and ret_type=='lists':
            pass # no processing, return them standard

        #TODO different processings
        elif self.mode == 'standard' and ret_type=='numpy':
                pass # no processing, return them standard

        else:
            print('not in category')

        return images

    def process_labels_to_lists_or_numpy(self,labels, ret_type):
        if self.mode == 'standard' and ret_type=='numpy':
            pass # no processing, return them standard

        #TODO different processings
        elif self.mode == 'standard' and ret_type=='numpy':
                pass # no processing, return them standard

        else:
            print('not in category')

        return labels

    def load(self, path_image, path_label, batch=None):
        _np = _import_numpy()
        _cv2 = _import_cv2()

        images=[]
        labels=[]
        if batch is not None:
            if type(batch) is not list or len(batch) is not 2:
                raise ValueError('batch should be a 1-D list'
                                 '(start_point, batch_size)')

        with open(path_label, 'rb') as file:
            print(path_label)
            magic, size,rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            label_data = array("B", file.read())

        with open(path_image, 'rb') as file:
            print(path_image)
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())


        if batch is not None:
            image_data = image_data[batch[0] * rows * cols * 3:\
                                    (batch[0] + batch[1]) * rows * cols * 3]
            label_data = label_data[batch[0] * rows * cols :\
                                    (batch[0] + batch[1]) * rows * cols ]

            size = batch[1]

        for i in range(size):
             labels.append([0] * rows * cols )

        for i in range(size):
            lbl = _np.array(label_data[i * rows * cols :(i + 1) * rows * cols ])
            lbl = lbl.reshape(256, 192)
            labels[i][:] = lbl

        for i in range(size):
            images.append([0] * rows * cols * 3)
        for i in range(size):
            img = image_data[i * rows * cols * 3:(i + 1) * rows * cols * 3]
            img=_np.array(img)
            img = img.reshape(256, 192,3)
            images[i][:] = img

        return np.array(images), np.array(labels)
