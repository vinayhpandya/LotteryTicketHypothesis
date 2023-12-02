from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Lambda, ToTensor


class MNISTDataset(Dataset):
    """MNIST Dataset
    Feature images which consitsts of 28*28
    will be flattened into a single vector

    Parameters:
    ----
    root: str
        Place where all the images will be downloaded
    train: bool
        Determine if the batch is for training or testing and apply transforms
        accordingly
    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, root, train=True, download=True) -> None:
        super().__init__()
        transforms = Compose(
            [
                ToTensor(),
                Lambda(lambda x: x.ravel()),
            ]
        )
        self.tv_dataset = MNIST(root, train=train, download=True, transform=transforms)

    def __len__(self):
        """
        Generate the length of the dataset
        """
        return len(self.tv_dataset)

    def __getitem__(self, index):
        """Get Specific item from a dataset

        Args:
            index (int): Index of item

        Returns:
            x: Torch.tensor
                Flattened tensor of shape (784,)
            y: Torch.tensor
                label/digit for that particular item
        """
        return self.tv_dataset[index]
