from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SingleImageDataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(SingleImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        import os
        self.lq_folder = os.path.dirname(opt['dataroot_lq'])
        self.lq_file = opt['dataroot_lq']
        # self.lq_folder = opt['dataroot_lq']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = paths_from_lmdb(self.lq_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.lq_folder, line.rstrip().split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load lq image
        # lq_path = self.paths[index]
        lq_path = self.lq_file  # no index, just single image file
        img_bytes = self.file_client.get(lq_path, 'lq')

        ############################################################################################
        # 240814, 240903 written by SChoi
        import os
        import re
        import numpy as np
        # Get the file extension
        _, extension = os.path.splitext(lq_path)
        if extension == '.raw':
            filename = os.path.basename(lq_path)
            # Extract width and height using regular
            match = re.search(r'_(\d+)X(\d+)', filename)
            # match = re.search(r"w(\d+)_h(\d+)", filename)
            if match:
                width = int(match.group(1))
                height = int(match.group(2))
            else:
                raise ValueError("Could not extract dimensions from filename")
            #
            # # Read binary file
            # with open(lq_path, 'rb') as file:
            #     binary_data = file.read()
            #
            # # Convert binary data to numpy array
            # image_array = np.frombuffer(binary_data, dtype=np.float32)

            # Read the raw file into a NumPy array
            image_array = np.fromfile(lq_path, dtype='uint16')
            image_array = image_array.astype(np.float32)
            # min-max norm
            image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
            # Reshape array to 2D image
            img_gray = image_array.reshape((height, width))

            # Ensure the input is 2D and float32
            if img_gray.ndim != 2 or img_gray.dtype != np.float32:
                raise ValueError("Input must be a 2D array with float32 dtype")

            # Create a 3D array by stacking the grayscale image 3 times
            img_lq = np.stack((img_gray,) * 3, axis=-1)
        else:
            img_lq = imfrombytes(img_bytes, float32=True)
        ############################################################################################

        # img_lq = imfrombytes(img_bytes, float32=True)

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
        return {'lq': img_lq, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths)
