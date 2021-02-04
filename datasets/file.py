import logging
import os
import shutil
import zipfile


class Folder:

    def __init__(self, way):
        self.logger = logging.getLogger("file_logger")
        self.way = way
        self.logger.info("Folder ({}) was open".format(self.way))

    def get_sorted_files(self):
        self.logger.info("File names in folder({}) was found  ".format(self.way))
        return sorted(os.listdir(self.way))

    def copy_with_new_names(self, external_folder, order):
        counter = order
        for name in external_folder.get_sorted_files():
            self.logger.info("Copy ({}) was coped".format(name))
            *other_name, extension = name.split(".") 
            shutil.copyfile(external_folder.way + '/' + name, self.way + '/' + str(counter) + '.' + extension)
            counter += 1

    def copy_with_given_names(self, folder, names):
        file_names = folder.get_sorted_files()
        for name in file_names:
            shutil.copyfile(folder.way + '/' + name, self.way + '/' + names[name])

    def count_of_files(self):
        names = self.get_sorted_files()
        counter = len(names)
        self.logger.info("Amount of files: {}".format(counter))
        return counter

    def delete_difference(self, external_folder):
        list_of_right_names = external_folder.get_sorted_files()
        for name in self.get_sorted_files():
            if name not in list_of_right_names:
                os.remove(self.way + "/" + name)
                self.logger.info("File {} was deleted!".format(self.way + "/" + name))

    def delete_match(self, external_folder):
        list_of_names = external_folder.get_sorted_files()
        for name in self.get_sorted_files():
            if name in list_of_names:
                os.remove(self.way + "/" + name)
                self.logger.info("File {} was deleted!".format(self.way + "/" + name))

    def unzip_to_folder(self, way_to_zip):
        with zipfile.ZipFile(way_to_zip, 'r') as archive:
            archive.extractall(self.way + "/")

    def clean_folder(self):
        names = os.listdir(self.way)
        for name  in names:
            os.remove(self.way + "/" + name)