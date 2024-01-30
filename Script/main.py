import os
import sys
import urllib.request

DATA_URL = ['http://www.image-net.org/image/downsample/Imagenet64_train_part1.zip',
            'http://www.image-net.org/image/downsample/Imagenet64_train_part2.zip']

dataset_dir = "E:/DDPM/Dataset"

filepath = os.path.join(dataset_dir, 'train_data_batch_1')    # filepath = 'imagenet/train/train_batch_1'
if not os.path.exists(filepath):
    file_paths = []

    def _download_progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    for i, url in enumerate(DATA_URL):
        filepath, _ = urllib.request.urlretrieve(url=url, filename=os.path.join(dataset_dir, url.split('/')[-1]), reporthook=_download_progress)
        file_paths.append(filepath)

    print("Download finished.")
