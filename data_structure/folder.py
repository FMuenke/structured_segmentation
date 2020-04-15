import os


class Folder:
    def __init__(self, path_to_folder):
        self.path_to_folder = path_to_folder

    def __str__(self):
        return self.path()

    def check_n_make_dir(self, clean=False):
        """
        checks if a directory exits and maks one if necessary
        :param tar_dir:
        :param clean: if True all files in folder will be deleted
        :return:
        """
        if not os.path.isdir(self.path_to_folder):
            os.mkdir(self.path_to_folder)

        if clean:
            for f in os.listdir(self.path_to_folder):
                if not os.path.isdir(os.path.join(self.path_to_folder, f)):
                    os.remove(os.path.join(self.path_to_folder, f))
                else:
                    fold = Folder(os.path.join(self.path_to_folder, f))
                    fold.check_n_make_dir(clean=True)

    def path(self):
        return self.path_to_folder

    def exists(self):
        return os.path.isdir(self.path_to_folder)
