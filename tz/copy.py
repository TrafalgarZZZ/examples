from shutil import copyfile
from shutil import copytree
import uuid
import os
import tqdm

source = "/home/tz/Pictures/unix-cloud-1920x1080.jpg"

destination_dir = "/media/tz/TOSHIBA EXT/dataset/class"

def batch_copy(num):
    for i in range(num):
        basename = str(uuid.uuid1()) + ".jpg"
        filename = os.path.join(destination_dir, basename)
        copyfile(source, filename)

def copy_dir(num):
    for i in tqdm.trange(num):
        from_dir = destination_dir
        to_dir = destination_dir + str(i)
        # print("Copying %s to %s..." % (source, to_dir))
        copytree(from_dir, to_dir)

if __name__ == '__main__':
    batch_copy(5)
    copy_dir(100)
