import numpy as np
import pandas as pd
import os
import keras
import random
from keras.preprocessing import image
from augmentation.rotate_augmentation import rotate_image
from augmentation.shuffle_points import shuffle_points
from augmentation.bias_augmentation import image_bias
from augmentation.brightness_augmentation import brightness
from augmentation.noise_augmentation import noise


class my_DataGenerator(keras.utils.Sequence):
    def __init__(self, data_dict, img_path, batch_size=32, dim=(512, 512, 3), n_channels = 1, shuffle = True, aug = True):
        self.data_dict = data_dict
        self.dim = dim
        self.batch_size = batch_size
        self.list_data = list(self.data_dict.keys())
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.img_path = img_path
        self.aug = aug


    def __getitem__(self, index):
        "Returns current batch of the data"
        list_data_temp = self.list_data[index*self.batch_size:(index + 1) * self.batch_size]
        X, Y = self.__data_generation(list_data_temp)
        return X, Y


    def __len__(self):
        return int(np.floor(len(self.list_data) / self.batch_size))

    
    def on_epoch_end(self):
        "Shuffle all dataset on the epoch end:)"
        self.list_data = list(self.data_dict.keys())
        if self.shuffle == True:
            np.random.shuffle(self.list_data)

    def __data_generation(self, list_data_temp):
        X = np.empty((self.batch_size, *self.dim))

        Y = np.empty((self.batch_size, 8), dtype=int)
        # TODO: нормальная обработка изображения
        for i, name in enumerate(list_data_temp):
            img = image.load_img(self.img_path + name, target_size=(512, 512))
            img_array = image.img_to_array(img)

            if self.aug == True:
                choice = random.randrange(0, 13)
                #brightness_coeff = random.randrange(18, 22) / 18
                brightness_coeff = random.uniform(1.009, 1.2)
                #darkness_coeff = random.randrange(15, 40) / 40
                darkness_coeff = random.uniform(0.375, 0.99)

                if (choice == 0):
                    img_array, Y[i] = rotate_image(img_array, random.randrange(-30, 30), self.data_dict[name]) #гуд
                    img_array = img_array.astype('float32') / 255
                    X[i,] = img_array
                elif (choice == 1):
                    img_array = img_array.astype('float32') / 255
                    X[i,] = img_array
                    Y[i] = shuffle_points(self.data_dict[name]) #гуд
                elif (choice == 2): 
                    img_array, Y[i] = rotate_image(img_array, random.randrange(-20, 20), self.data_dict[name])
                    img_array = img_array.astype('float32') / 255
                    X[i,] = img_array
                    Y[i] = shuffle_points(Y[i])
                elif (choice == 3):
                    img_array = img_array.astype('float32') / 255
                    X[i,] = img_array
                    Y[i] = self.data_dict[name]
                elif (choice == 4):
                    X[i,], Y[i] = image_bias(img_array, random.randrange(-40, 40), random.randrange(-30, 30), self.data_dict[name])
                    X[i,] = X[i,].astype('float32') / 255
                elif (choice == 5):
                    img_array = brightness(img_array, brightness_coeff)
                    X[i,] = img_array.astype('float32') / 255
                    Y[i] = self.data_dict[name]
                elif (choice == 6):
                    img_array = brightness(img_array, darkness_coeff)
                    X[i,] = img_array.astype('float32') / 255
                    Y[i] = self.data_dict[name]
                elif (choice == 7):
                    img_array, Y[i] = rotate_image(img_array, random.randrange(-20, 20), self.data_dict[name])
                    X[i,], Y[i] = image_bias(img_array, random.randrange(-40, 40), random.randrange(-30, 30), Y[i])
                    X[i,] = X[i,].astype('float32') / 255
                #--------------------------------------------------------------------------------------------
                if (choice == 8):
                    img_array, Y[i] = rotate_image(img_array, random.randrange(-30, 30), self.data_dict[name])
                    if (random.randrange(0, 2)):
                        img_array = brightness(img_array, brightness_coeff)
                    else:
                        img_array = brightness(img_array, darkness_coeff)
                    img_array = img_array.astype('float32') / 255
                    X[i,] = img_array
                elif (choice == 9):
                    if (random.randrange(0, 2)):
                        img_array = brightness(img_array, brightness_coeff)
                    else:
                        img_array = brightness(img_array, darkness_coeff)
                    img_array = img_array.astype('float32') / 255
                    X[i,] = img_array
                    Y[i] = shuffle_points(self.data_dict[name])
                elif (choice == 10): 
                    img_array, Y[i] = rotate_image(img_array, random.randrange(-20, 20), self.data_dict[name])
                    if (random.randrange(0, 2)):
                        img_array = brightness(img_array, brightness_coeff)
                    else:
                        img_array = brightness(img_array, darkness_coeff)
                    img_array = img_array.astype('float32') / 255
                    X[i,] = img_array
                    Y[i] = shuffle_points(Y[i])
                elif (choice == 11):
                    if (random.randrange(0, 2)):
                        img_array = brightness(img_array, brightness_coeff)
                    else:
                        img_array = brightness(img_array, darkness_coeff)
                    img_array = img_array.astype('float32') / 255
                    X[i,] = img_array
                    Y[i] = self.data_dict[name]
                elif (choice == 12):
                    if (random.randrange(0, 2)):
                        img_array = brightness(img_array, brightness_coeff)
                    else:
                        img_array = brightness(img_array, darkness_coeff)
                    X[i,], Y[i] = image_bias(img_array, random.randrange(-40, 40), random.randrange(-30, 30), self.data_dict[name])
                    X[i,] = X[i,].astype('float32') / 255
                #elif (choice == 13):
                #    img_array = noise(img_array, random.randrange(10, 60))
                #    X[i,] = img_array.astype('float32') / 255
                #    Y[i] = self.data_dict[name]
                #--------------------------------------------------------------------------------------------
            else:
                img_array = img_array.astype('float32') / 255
                X[i,] = img_array
                Y[i] = self.data_dict[name]

        return X, Y


def create_dict_from_folder(path_to_folder, path_to_csv):
    'Сreates a dictionary that contains only the images in the folder'
    heads_data = pd.read_csv(path_to_csv)
    images = os.listdir(path_to_folder)

    heads = {}

    for key, value in heads_data.iterrows():
        name = value["#filename"]
        if name in heads:
            point = value["region_shape_attributes"]
            point = eval(point)
            heads[name].append(point["cx"])
            heads[name].append(point["cy"])
        else:
            heads[name] = []
            point = value["region_shape_attributes"]
            if point != '{}':
                point = eval(point)
                heads[name].append(point["cx"])
                heads[name].append(point["cy"])

    keys_to_delete = []
    for key in heads:
        if len(heads[key]) != 8 or key not in images:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del heads[key]

    return heads


if __name__ == "__main__":
    pass