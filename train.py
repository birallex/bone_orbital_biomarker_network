import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from models import build_model_8
from generator import my_DataGenerator
from generator import create_dict_from_folder

def plot_mae(history):
    plt.plot(range(1, len(history) + 1), history)
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    diagrams = os.listdir('learning_statistics/')
    diagrams_counter = int(len(diagrams) / 2)
    new_diagram_name = 'MAE_history_' + str(diagrams_counter + 1) + '.png'
    #plt.show()
    plt.savefig('learning_statistics/' + new_diagram_name)

def plot_val_mae(history):
    plt.plot(range(1, len(history) + 1), history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    diagrams = os.listdir('learning_statistics/')
    diagrams_counter = int(len(diagrams) / 2)
    new_diagram_name = 'VAL_MAE_history_' + str(diagrams_counter + 1) + '.png'
    #plt.show()
    plt.savefig('learning_statistics/' + new_diagram_name)


if __name__ == "__main__":
    train_img_path = "datasets/train/"
    valid_img_path = "datasets/valid/"
    path_to_csv = "datasets/bmp.csv"

    train_dict = create_dict_from_folder(train_img_path, path_to_csv)
    valid_dict = create_dict_from_folder(valid_img_path, path_to_csv)

    train_generator = my_DataGenerator(train_dict, train_img_path, batch_size=16)
    valid_generator = my_DataGenerator(valid_dict, valid_img_path, batch_size=16, aug=False)

    model = build_model_8()
    model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

    checkpoint = ModelCheckpoint("weights/points32.h5", monitor='val_mae', verbose=1, save_best_only=True,
                                  save_weights_only=False, mode='max')

    callbacks_list = [checkpoint]

    history = model.fit_generator(generator = train_generator, validation_data = valid_generator,
     use_multiprocessing=True, callbacks=callbacks_list, epochs = 50, verbose=1)

    print(history.history)
    val_mae_history = history.history['val_mae']
    mae_history = history.history['mae']

    plot_mae(mae_history)
    plot_val_mae(val_mae_history)

    print("Training completed!")