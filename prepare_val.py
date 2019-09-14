import os
from tqdm import tqdm

data = open('val/val_annotations.txt', 'r')
data = [x.strip().split('\t')[:2] for x in data.readlines()]


for item in tqdm(data):
    try:
        new_dir_path = os.path.join('val', item[1])
        os.mkdir(new_dir_path)
    except FileExistsError:
        pass
    src = os.path.join('val', 'images', item[0])
    dest = os.path.join('val', item[1], item[0])
    os.rename(src, dest)

print('Done')
print('Remove images folder')
os.rmdir('val/images')
