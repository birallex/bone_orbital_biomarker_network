from generator import *

train_img_path = "datasets/train/"
valid_img_path = "datasets/valid/"
path_to_csv = "datasets/bmp.csv"
train_dict = create_dict_from_folder(train_img_path, path_to_csv)
valid_dict = create_dict_from_folder(valid_img_path, path_to_csv)

train_generator = my_DataGenerator(train_dict, train_img_path, batch_size=32)
valid_generator = my_DataGenerator(valid_dict, valid_img_path, aug=False)

for X, Y in train_generator:
    print(X, Y)
    size = []
    f = open('output/testGeneratorOutput.csv', 'w')
    f.write('#filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes\n')
    my_range = [x for x in range(8)]
    for i in range(0, len(X)):
        
        img = image.array_to_img(X[i])
        img.save('output/img/{}.png'.format(i))
        
        statinfo = os.stat('output/img/{}.png'.format(i))
        size.append(statinfo.st_size)
        
        predict = np.around(Y[i])
        predict = predict.astype('int32')
        
        for g in my_range[0:8:2]:      
            point = '{}.png,{},'.format(i, statinfo.st_size) + '{},4,' + '{},'.format(int(g/2) ) \
                +'"{"' + "\"name\"\":\"\"point\"\",\"\"cx\"\":{},\"\"cy\"\":{}".format(predict[g], predict[g+1]) +"}\",{}\n"
            f.write(point)

    f.close()
    break