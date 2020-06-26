import os
import cv2
import time
import h5py
from PIL import Image
import numpy as np
import random

'''
You should modify the file path here
'''
read_path_test = "/Users/zihaoliu/Desktop/Graduate_UW/SYDE660/CNN/data/test"
save_path_test = "/Users/zihaoliu/Desktop/Graduate_UW/SYDE660/CNN/data/test_resize"

read_path_train = "/Users/zihaoliu/Desktop/Graduate_UW/SYDE660/CNN/data/train"
save_path_train = "/Users/zihaoliu/Desktop/Graduate_UW/SYDE660/CNN/data/train_resize"

txt_train_file = "/Users/zihaoliu/Desktop/Graduate_UW/SYDE660/CNN/data/train_split_v3.txt"
txt_test_file = "/Users/zihaoliu/Desktop/Graduate_UW/SYDE660/CNN/data/test_split_v3.txt"


def resize_image(read_path, save_path, resulution):
    '''
    Resize the images under read_path into save_path
    :param read_path: the directory storing COVIDx training or testing dataset
    :param save_path: the directory storing resized training or testing dataset
    :return: None
    '''
    tic = time.time()
    for filename in os.listdir(read_path):
        print(filename)
        read_image_path = read_path + '/' + filename
        image = cv2.imread(read_image_path)
        res = cv2.resize(image, (resulution, resulution), interpolation=cv2.INTER_AREA)
        save_image_path = save_path + '/' + filename
        cv2.imwrite(save_image_path, res)
    toc = time.time()
    print("Resize time is %.2f." % (toc - tic))


def get_class_info(txt_file, image_path):
    with open(txt_file) as file_obj:
        lines = file_obj.readlines()
        image_number = len(lines)
        image_name_list = os.listdir(image_path)
        dict_pneumonia = {}
        dict_normal = {}
        dict_covid = {}
        for index, line in enumerate(lines):
            line_split = line.split()
            image_name = line_split[1]
            image_label = line_split[2]
            if image_name in image_name_list:
                if image_label == 'pneumonia':
                    dict_pneumonia[index] = image_name
                elif image_label == 'normal':
                    dict_normal[index] = image_name
                elif image_label == 'COVID-19':
                    dict_covid[index] = image_name
    print("The number of pneumonia is %d" % len(dict_pneumonia))
    print("The number of pneumonia is %d" % len(dict_normal))
    print("The number of pneumonia is %d" % len(dict_covid))
    return dict_pneumonia, dict_normal, dict_covid


def random_get_small_batch_class_arr(dicts, amount_list):
    dict_pneumonia, dict_normal, dict_covid = dicts
    amount_pneumonia, amount_normal, amount_covid = amount_list
    random_pneumonia = random.sample(dict_pneumonia.keys(), amount_pneumonia)
    random_normal = random.sample(dict_normal.keys(), amount_normal)
    random_covid = random.sample(dict_covid.keys(), amount_covid)
    return random_pneumonia, random_normal, random_covid


def get_np_image_label(dicts, random_lists, image_path):
    tic = time.time()
    dict_pneumonia, dict_normal, dict_covid = dicts
    random_pneumonia, random_normal, random_covid = random_lists
    image_number = len(random_pneumonia) + len(random_normal) + len(random_covid)
    np_array = np.zeros((image_number, 480, 480, 3))
    np_label = np.zeros((image_number, 1))
    for index, key in enumerate(random_pneumonia):
        image = Image.open(image_path + '/' + dict_pneumonia[key])
        np_array_temp = np.array(image)
        np_array[index] = np_array_temp
        np_label[:len(random_pneumonia)] = 0
        print('Pneumonia', ' ', dict_pneumonia[key])
    for index, key in enumerate(random_normal):
        image = Image.open(image_path + '/' + dict_normal[key])
        np_array_temp = np.array(image)
        np_array[len(random_pneumonia) + index] = np_array_temp
        np_label[len(random_pneumonia):len(random_pneumonia) + len(random_normal)] = 1
        print('Normal', ' ', dict_normal[key])
    for index, key in enumerate(random_covid):
        image = Image.open(image_path + '/' + dict_covid[key])
        np_array_temp = np.array(image)
        np_array[len(random_pneumonia) + len(random_normal) + index] = np_array_temp
        np_label[len(random_pneumonia) + len(random_normal):] = 2
        print('COVID-19', ' ', dict_covid[key])
    toc = time.time()
    print("Numpy generating time is %.2f." % (toc - tic))
    return np_array.astype(int), np_label.astype(int)

def get_new_dataset(txt_file, image_path, amount_list):
    dicts = get_class_info(txt_file, image_path)
    random_lists = random_get_small_batch_class_arr(dicts, amount_list)
    np_sets = get_np_image_label(dicts, random_lists, image_path)
    return np_sets

def store_as_h5(np_array, np_label, output_path):
    if not os.path.exists(output_path + '.h5'):
        tic = time.time()
        h5f = h5py.File(output_path + '.h5', 'w')
        h5f.create_dataset('data', data=np_array, chunks=True, dtype='int')
        h5f.create_dataset('label', data=np_label, chunks=True, dtype='i')
        # with h5py.File(output_path + '.h5', 'w') as h5:
        #     h5['data'] = np_array
        #     h5['label'] = np_label
        h5f.close()
        toc = time.time()
        print('.h5 file generating time is %.2f.' % (toc - tic))

def load_covidx_dataset(amount_list_train = [200, 200, 200], amount_list_test = [20, 20, 20]):
    '''
    You can modify the batch size of training and testing sets
    :param amount_list_train: 1st element is the number of pneumonia. 2nd is normal. 3rd is COVID-19
    :param amount_list_test: same as above
    :return: a shape(sum(amount_list_train) * 480 * 480 * 3) train dataset, a (sum(amount_list_train) * 1) train label
             a shape(sum(amount_list_test) * 480 * 480 * 3) test dataset, a (sum(amount_list_test) * 1) test label
    '''
    # amount_list_train = [200, 200, 200]
    # amount_list_test = [20, 20, 20]
    train_X, train_y = get_new_dataset(txt_train_file, save_path_train, amount_list_train)
    print("%d samples in training set generation done" % (sum(amount_list_train)))
    test_X, test_y = get_new_dataset(txt_test_file, save_path_test, amount_list_test)
    print("%d samples in testing set generation done" % (sum(amount_list_test)))
    classes = np.array((0, 1, 2))
    return train_X, train_y, test_X, test_y, classes

def convert_to_one_hot_matrix(Y, C):
    '''
    Convert the (n * 1) shape label to a matrix
    :param Y: the input (n * 1) matrix
    :param C: the number of class
    :return: a (C * n)shape matrix
    '''
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
