import h5py
from PIL import Image

from data.image_folder import make_dataset
from data.base_dataset import BaseDataset, get_transform

class SingleHdfDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        hdf_file = h5py.File(opt.dataroot + f'/train{opt.hdf_dataset}.hdf', 'r')['volumes']
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
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
        
        return {'A': hdf, 'A_paths': self.dataroot + f'/trainA+_{hdf_idx}.png'}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.hdf_dataset.shape[0]