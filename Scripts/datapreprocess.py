import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, ConcatDataset, Dataset
from typing import Tuple, List, Dict
import random

class empty_dataset(Dataset):

    def __init__(self):
        pass

    def __len__(self):
        return 0
    
    def __getitem__(self, idx):
        return IndexError("This dataset is empty!")
    

class DataPreprocessor():

    """Handles dataset loading, augmentation, and subset creation.
    
    Attributes:
        train_dir (str): Path to training data directory
        test_dir (str): Path to test data directory
        augmentation_presets (Dict[str, transforms.Compose]): Available augmentation strategies
    """

    def __init__(self, train_dir, test_dir):
        """
        Initialize with dataset paths.
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.augmentation_presets = self._get_augmentation_transforms()

    def _get_augmentation_transforms(self) -> Dict[str, transforms.Compose]:

        """
        Define all augmentation strategies.
        """

        return {
            "No_Augmentation" : transforms.Compose([
                                transforms.Resize((256,256)),
                                transforms.RandomResizedCrop((224,224), scale = (0.1, 1)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                            ]),
            "Gaussian_Blur" :   transforms.Compose([
                                transforms.RandomApply([
                                    transforms.GaussianBlur(kernel_size= 3, sigma= (0.1, 0.3))
                            ], p= 0.5),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomResizedCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                            ])
        }

    def Create_Datasets(self,
                        train_augmentation: str= "No_Augmentation", 
                        test_augmentation : str= "No_Augmentation") -> Tuple[Dataset, Dataset]:
        """
        Create training and test datasets with specified augmentations.
        
        Args:
            train_augmentation: Augmentation strategy for training data
            test_augmentation: Augmentation strategy for test data
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        train_data = datasets.ImageFolder(root= self.train_dir,
                                          transform= self.augmentation_presets[train_augmentation],
                                          target_transform= None)
        test_data = datasets.ImageFolder(root= self.test_dir,
                                         transform= self.augmentation_presets[test_augmentation])
        return train_data, test_data
    
    def create_subset(self, 
                      dataset : Dataset, 
                      num_of_datasets : int, 
                      size_of_datasets : int) -> List[Subset]:

        """
        This will create n no. of subsets of the given data

        Args:
            dataset: The original dataset.
            num_subsets: Number of subsets to create.
            subset_size: Number of samples in each subset.
        
        Returns:
            A list of Subset objects.
        """
        subsets = []
        for i in range(num_of_datasets):
            start_idx = i * int(size_of_datasets / 2) 
            end_idx = start_idx + int(size_of_datasets / 2)

            start_idx_2 = start_idx + int(len(dataset) / 2)
            end_idx_2 = end_idx + int(len(dataset) / 2)

            subset_indices_1 = list(range(start_idx, end_idx))
            subset_indices_2 = list(range(start_idx_2, end_idx_2))
            subset_indices = subset_indices_1 + subset_indices_2
            random.shuffle(subset_indices)
            
            subsets.append(Subset(dataset, subset_indices))

        return subsets
    
    def Dataloader_subsets(self, 
                           train_subset_data : List[Subset], 
                           test_subset_data : List[Subset], 
                           batch_size : int) -> Tuple[DataLoader, DataLoader]:

        train_dataloader_subsets = [DataLoader(subset, batch_size, shuffle= True) for subset in train_subset_data]
        test_dataloader_subsets = [DataLoader(subset, batch_size) for subset in test_subset_data]

        return train_dataloader_subsets, test_dataloader_subsets
    
    def amount_of_data(self, 
                       dataloader_subsets: torch.utils.data.DataLoader, 
                       multiple : int , 
                       batch_size : int) -> torch.utils.data.DataLoader:

        total_subsets = len(dataloader_subsets)

        concatdataset = empty_dataset()
        
        if multiple > total_subsets:
            raise ValueError(
                "multiple greater than the number of subsets"
            )
        else:
            for i in range(multiple):
                dataloader_dataset = dataloader_subsets[i].dataset
                concatdataset = ConcatDataset([concatdataset, dataloader_dataset])
            dataloaders = DataLoader(concatdataset, batch_size= batch_size, shuffle= True)
            
        return dataloaders
    
    def Build_Dataloaders(self, 
                          train_augmentation : str = "No_Augmentation", 
                          test_augmentation : str = "No_Augmentation", 
                          num_subsets: int = 40, 
                          batch_size : int = 50, 
                          percentage_data: int = 5) -> Tuple[DataLoader, DataLoader]:
        
        """
        Main pipeline for creating data loaders.
        
        Args:
            train_augmentation: Training data augmentation preset
            test_augmentation: Test data augmentation preset
            num_subsets: Total subsets to create
            batch_size: Samples per batch
            percentage_data: Percentage of total data to use
            
        Returns:
            Tuple of (train_loader, test_loader)
        """

        train_data, test_data = self.Create_Datasets(train_augmentation= train_augmentation,
                                                    test_augmentation= test_augmentation)
        
        
        train_subset_size = int(len(train_data) / num_subsets)
        test_subset_size = int(len(test_data) / num_subsets)

        subsetdata_use = int(percentage_data * len(train_data) / (100 * train_subset_size))

        train_subsets = self.create_subset(train_data, num_subsets, train_subset_size)
        test_subsets = self.create_subset(test_data, num_subsets, test_subset_size)

        train_dataloader_subsets, test_dataloader_subsets = self.Dataloader_subsets(train_subsets, test_subsets, batch_size)

        built_train_dataloader = self.amount_of_data(train_dataloader_subsets, subsetdata_use, batch_size)
        built_test_dataloader = self.amount_of_data(test_dataloader_subsets, subsetdata_use, batch_size)

        return built_train_dataloader, built_test_dataloader
    
