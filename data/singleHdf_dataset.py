import h5py
from PIL import Image

from data.image_folder import make_dataset
from data.base_dataset import BaseDataset, get_transform

class SingleHdfDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

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
    
        parser.set_defaults(input_nc=1, output_nc=1, preprocess='crop',
                            crop_size=64)  # specify dataset-specific default values
    
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        hdf_file = h5py.File(opt.dataroot + f'/train{opt.hdf_dataset}.hdf', 'r')['volumes']
        self.hdf_dataset = hdf_file.get('raw')[()]
        self.dataroot = opt.dataroot
        self.hdf_label = opt.hdf_dataset
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        hdf_idx = index % self.hdf_dataset.shape[0]
        hdf_image = Image.fromarray(self.hdf_dataset[hdf_idx])
        hdf = self.transform(hdf_image)
        
        return {'A': hdf, 'A_paths': self.dataroot + f'/trainA_{hdf_idx}.png'}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.hdf_dataset.shape[0]