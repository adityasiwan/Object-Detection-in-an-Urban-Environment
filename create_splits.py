import argparse
import glob
import os
import random
import shutil
import numpy as np
from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    cwd = os.getcwd()
    dir_list = ["train", "test", "val"]

    source = data_dir

    files = os.listdir(source)
    # Shuffle the files before split
    random.shuffle(files)
    num = len(files)

    def create_dir(folder_name):
        """
        creates an empty directory with folder_name
        """
        dir_path = "{}/{}".format(cwd, folder_name)
        os.makedirs(dir_path, exist_ok=True)

    def split(start, end, dest_dir):
        """
        a function to split the files in the desired ratio
        args:
            - start, end: split the files from start:end
            - dir: the destination directory where the splitted files will be moved
        """
        dest_dir = cwd + '/' + dest_dir
        for f in files[start:end]:
            shutil.move(source + '/' + f, dest_dir + '/' + f)

    for x in dir_list:
        create_dir(x)

    # For Train
    start = 0
    end = start + int(0.8 * num)
    split(start, end, dir_list[0])

    # For Test
    start = end
    end = start + int(0.1 * num)
    split(start, end, dir_list[1])

    # For Validation
    start = end
    end = num
    split(start, end, dir_list[2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)
    logger.info('Split completed')
