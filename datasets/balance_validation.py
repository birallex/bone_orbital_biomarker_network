from file import *
import random

train_way = "train"
train = Folder(train_way)

valid_way = "valid"
valid = Folder(valid_way)

names = train.get_sorted_files()
count_of_files = len(names)
print("Totlal images: " + str(count_of_files))

amount_of_validation_data = 0.06 #%

amount_of_validation_data *= count_of_files
print("Total validation images: " + str(amount_of_validation_data))

random.shuffle(names)

files_to_move = names[:int(amount_of_validation_data)]

for target in files_to_move:
    source = train_way + '/' + target
    destination =  valid_way + '/' + target
    shutil.move(source, destination)
    #print(source)