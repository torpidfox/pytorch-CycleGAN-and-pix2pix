import h5py
from PIL import Image

from data.image_folder import make_dataset

"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image


class HdfPngDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--hdf_dataset',
                            type=str,
                            default='A',
                            choices=['A', 'B'],
                            help='index of the dataset in hdf format')
        
        
        parser.set_defaults(input_nc=1, output_nc=1, preprocess='crop', crop_size=64)  # specify dataset-specific default values
        
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        print(opt.dataroot)
        self.dataroot = opt.dataroot
        self.hdf_label = opt.hdf_dataset
        self.png_label = 'B' if opt.hdf_dataset == 'A' else 'A'
        hdf_file = h5py.File(opt.dataroot + f'/train{opt.hdf_dataset}.hdf', 'r')['volumes']
        
        self.hdf_dataset = hdf_file.get('raw')[()]
        self.png_image_paths = sorted(make_dataset(opt.dataroot + f'/train{self.png_label}', opt.max_dataset_size))
        self.png_size = len(self.png_image_paths)
        
        self.transform_hdf = get_transform(self.opt, grayscale=True, noise=self.opt.aug_noise)
        self.transform_png = get_transform(self.opt, scale=opt.aug_scale, grayscale=True)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        png_path = self.png_image_paths [index % len(self.png_image_paths)]
        hdf_idx = index % self.hdf_dataset.shape[0]
        
        png_image = Image.open(png_path).convert('L')
        hdf_image = Image.fromarray(self.hdf_dataset[hdf_idx])
        
        png = self.transform_png(png_image)
        hdf = self.transform_hdf(hdf_image)
        
        return {self.hdf_label: hdf , self.png_label: png,
                f'{self.hdf_label}_paths': self.dataroot + f'/train{self.hdf_label}.hdf',
                f'{self.png_label}_paths': png_path}

    def __len__(self):
        """Return the total number of images."""
        return max(len(self.png_image_paths), self.hdf_dataset.shape[0])
