from models import *
import numpy as np
import pandas as pd
import sys
import os
from keras.preprocessing import image
import json


def test_csv_output(way_to_weights):
    test_img_path = "datasets/test/"
    dim=(512, 512, 3)
    names = os.listdir(test_img_path)
    count_of_heads = len(names)

    X = np.empty((count_of_heads, *dim))
    print(X.shape)
    for i, name in enumerate(names):
        img = image.load_img(test_img_path + name, target_size=(512, 512))
        img_array = image.img_to_array(img)
        img_array = img_array.astype('float32') / 255
        X[i,] = img_array

    model = build_model_8()
    model.load_weights(way_to_weights)

    predictions = model.predict(X)
    print(predictions)
    print(type(predictions))

    size = []
    f = open('output/result.csv', 'w')
    f.write('#filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes\n')
    my_range = [x for x in range(8)]
    for i in range(0, len(X)):
            
        img = image.array_to_img(X[i])
        img.save('output/img/{}.png'.format(i))
            
        statinfo = os.stat('output/img/{}.png'.format(i))
        size.append(statinfo.st_size)
            
        predict = np.around(predictions[i])
        predict = predict.astype('int32')
            
        for g in my_range[0:8:2]:      
            fuck = '{}.png,{},'.format(i, statinfo.st_size) \
                + '{},4,' + '{},'.format(int(g/2) )+'"{"' + "\"name\"\":\"\"point\"\",\"\"cx\"\":{},\"\"cy\"\":{}".format(predict[g], predict[g+1]) +"}\",{}\n"
            f.write(fuck)

    f.close()

def test_json_output(test_img_path, way_to_weights):
    dim=(512, 512, 3)
    names = os.listdir(test_img_path)
    count_of_heads = len(names)

    X = np.empty((count_of_heads, *dim))

    for i, name in enumerate(names):
        img = image.load_img(test_img_path + name, target_size=(512, 512))
        img_array = image.img_to_array(img)
        img_array = img_array.astype('float32') / 255
        X[i,] = img_array

    model = build_model_8()
    model.load_weights(way_to_weights)

    predictions = model.predict(X)
    
    json_dict = dict()
    my_range = [x for x in range(8)]
    for i, name in enumerate(names):        
        statinfo = os.stat(test_img_path + '{}'.format(name))
        predict = np.around(predictions[i])
        predict = predict.astype('int32')

        img_dict = dict()
        img_dict["fileref"] = ""
        img_dict["size"] = statinfo.st_size
        img_dict["filename"] = name
        img_dict["base64_img_data"] = ""
        img_dict["file_attributes"] = {}

        points_dict = dict()
        for g in my_range[0:8:2]:
            current_point = dict()
            coordinates = dict()
            coordinates["name"] = "point"
            coordinates["cx"] = int(predict[g])
            coordinates["cy"] = int(predict[g+1])       
            current_point["shape_attributes"] = coordinates
            current_point["region_attributes"] = {}
            points_dict[str(int(g/2))] = current_point
        img_dict["regions"] = points_dict
        json_dict[name+str(statinfo.st_size)] = img_dict
    
    with open("output/output.json", "w") as output_file:
        json.dump(json_dict, output_file, indent=4)

if __name__ == "__main__":
    #way_to_images = "/your/way/to/images/"
    #way_to_weights = "weights/points30.h5"
    #test_json_output(way_to_images, way_to_weights)
    way_to_weights = "weights/test30.h5"
    test_csv_output(way_to_weights)

    