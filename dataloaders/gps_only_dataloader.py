import numpy as np
import torch
import pytorch_lightning as pl
import os
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.functional as TF


def read_files_to_tensor_train(directory, augmentation=False):
    
    final_tensors = []
    
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith('.npy'): 
                file_path = os.path.join(directory, file_name)
                data = np.load(file_path)
                
                tensor_layer_1 = torch.tensor(data[:, :, 0], dtype=torch.float32)  
                tensor_layer_target = torch.tensor(data[:, :, -1], dtype=torch.float32)  
                    
                if augmentation == True : 
                  
                    tensor_layer_1 = tensor_layer_1.unsqueeze(0)
                    tensor_layer_target = tensor_layer_target.unsqueeze(0)
        
                    for i in range(4) :
                        TF.rotate(tensor_layer_1, angle = i*90)
                        TF.rotate(tensor_layer_target, angle = i*90)
                        tensor_layer_1 = tensor_layer_1.unsqueeze(-1)
                        tensor_layer_target = tensor_layer_target.unsqueeze(-1)
                        final_tensors.append((tensor_layer_1, tensor_layer_target))
                        tensor_layer_1 = tensor_layer_1.squeeze(-1)
                        tensor_layer_target = tensor_layer_target.squeeze(-1)
                    
                    TF.hflip(tensor_layer_1)
                    TF.hflip(tensor_layer_target)    
                    
                    for i in range(4) :
                        TF.rotate(tensor_layer_1, angle = i*90)
                        TF.rotate(tensor_layer_target, angle = i*90)
                        tensor_layer_1 = tensor_layer_1.unsqueeze(-1)
                        tensor_layer_target = tensor_layer_target.unsqueeze(-1)
                        final_tensors.append((tensor_layer_1, tensor_layer_target))
                        tensor_layer_1 = tensor_layer_1.squeeze(-1)
                        tensor_layer_target = tensor_layer_target.squeeze(-1)
                            
                else : 
        
                    tensor_layer_1 = tensor_layer_1.unsqueeze(0)
                    tensor_layer_1 = tensor_layer_1.unsqueeze(-1)
                    tensor_layer_target = tensor_layer_target.unsqueeze(0)
                    tensor_layer_target = tensor_layer_target.unsqueeze(-1)
                    final_tensors.append((tensor_layer_1, tensor_layer_target))
    return final_tensors


class ODBADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):        
        input_data, target_data = self.data[idx]
        return {'input': input_data, 'target': target_data}
    
class ODBADataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, test_split=0.05, num_workers=0, augmentation=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_split = test_split
        self.num_workers = num_workers
        self.augmentation = augmentation

        self.dataset = None
        self.train_val_set = None
        self.test_set = None
        self.train_subset = None
        self.val_subset = None
        self.test_subset = None

    def setup(self, stage=None):
        self.dataset = read_files_to_tensor_train(self.data_dir, augmentation=self.augmentation)
        test_size = int(self.test_split * len(self.dataset))
        train_val_size = len(self.dataset) - test_size

        indices = torch.randperm(len(self.dataset))
        train_val_data = Subset(self.dataset, indices[:train_val_size])
        test_data = Subset(self.dataset, indices[train_val_size:])

        self.train_val_set = ODBADataset(train_val_data)
        self.test_set = ODBADataset(test_data)

        val_size = len(self.train_val_set) // 5  # Example: 20% for validation
        train_size = len(self.train_val_set) - val_size

        self.train_subset, self.val_subset = torch.utils.data.random_split(self.train_val_set, [train_size, val_size])
        self.test_subset = self.test_set

    def train_dataloader(self):
        return DataLoader(self.train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_subset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_subset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    @property
    def num_train_samples(self):
        return len(self.train_subset) if self.train_subset is not None else 0

    @property
    def num_val_samples(self):
        return len(self.val_subset) if self.val_subset is not None else 0

    @property
    def num_test_samples(self):
        return len(self.test_subset) if self.test_subset is not None else 0
    
    
def read_files_to_tensor_test(directory):                        
    final_tensors = []
    metadata = []

    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith('.npy'):         
                file_path = os.path.join(directory, file_name)
                data = np.load(file_path)
                
                tensor_layer_1 = torch.tensor(data[:, :, 0], dtype=torch.float32)  
                tensor_layer_target = torch.tensor(data[:, :, -1], dtype=torch.float32)   
                    
            
                tensor_layer_1 = tensor_layer_1.unsqueeze(0)
                tensor_layer_1 = tensor_layer_1.unsqueeze(-1)
                tensor_layer_target = tensor_layer_target.unsqueeze(0)
                tensor_layer_target = tensor_layer_target.unsqueeze(-1)
        
                final_tensors.append((tensor_layer_1, tensor_layer_target))
                file_id = file_name.replace("_input.npy", "")
                metadata.append(file_id)
            
    return final_tensors, metadata


class ODBATestDataset(Dataset):
    def __init__(self, data, metadata=None):
        self.data = data
        self.metadata = metadata if metadata is not None else []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Unpack input-target pair from data
        input_data, target_data = self.data[idx]
        if self.metadata:
            return {'input': input_data, 'target': target_data, 'metadata': self.metadata[idx]}
        else:
            return {'input': input_data, 'target': target_data}

    
class ODBATestDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = None

    def setup(self, stage=None):
        final_tensors, metadata = read_files_to_tensor_test(self.data_dir)  # Unpack the returned tuple
        self.dataset = ODBATestDataset(final_tensors, metadata)  # Pass both data and metadata


    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    @property
    def num_train_samples(self):
        return 0
    
    @property
    def num_val_samples(self):
        return 0
    
    @property
    def num_test_samples(self):
        if self.dataset is not None:
            return len(self.dataset)
        else:
            return 0    