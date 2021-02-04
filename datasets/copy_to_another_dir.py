import file
import os

way = '/для теста нейронки/'
test = file.Folder("datasets/test")
test.clean_folder()

for folder in os.listdir(way):
    f = file.Folder(way + folder)
    test.copy_with_new_names(f, test.count_of_files())
