from torch.utils.data import Dataset, DataLoader
from semantic_kitti_api.utils.voxel import unpack, KITTILabelProcessor
from semantic_kitti_pytorch.data.datasets import SemanticKITTI_Completion, VOXEL_DIMS

import numpy as np
import torch


class MultiScaleDataset(Dataset):
    """Dataset wrapper that adds multi-scale functionality to SemanticKITTI_Completion"""

    def __init__(self, base_dataset):
        self.dataset = base_dataset
        self.label_processor = KITTILabelProcessor()
        self.phase = base_dataset.phase
        self.filenames = base_dataset.filenames

        self.multi_scales = [1, 2, 4, 8]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get base data from SemanticKITTI_Completion
        base_data = self.dataset[idx]

        if self.phase == 'test':
            return base_data

        # Initialize multi-scale data dictionary with explicit structure
        multi_scale_data = {
            'label_1_2': None,
            'invalid_1_2': None,
            'label_1_4': None,
            'invalid_1_4': None,
            'label_1_8': None,
            'invalid_1_8': None,
        }

        # Process each scale (skip 1:1 as it's in base_data)
        for scale in self.multi_scales[1:]:
            try:
                suffix = f'_1_{scale}'
                voxel_dims_scaled = tuple(int(dim / scale)
                                          for dim in VOXEL_DIMS)

                # Process labels
                label_path = f'{self.filenames[idx]}.label{suffix}'
                multi_scale_data[f'label{suffix}'] = self._load_and_process_label(
                    label_path, voxel_dims_scaled)

                # Process invalid masks
                invalid_path = f'{self.filenames[idx]}.invalid{suffix}'
                multi_scale_data[f'invalid{suffix}'] = self._load_and_process_invalid(
                    invalid_path, voxel_dims_scaled)

            except (FileNotFoundError, ValueError) as e:
                print(
                    f"Error processing scale 1:{scale} for idx {idx}: {str(e)}")
                continue

        return {**base_data, **multi_scale_data}

    def _load_and_process_label(self, path, dims):
        """Helper method to load and process label data"""
        label = np.fromfile(path, dtype=np.uint16).reshape(dims)
        label = self.label_processor.remap(label, mapping='learning')
        return torch.from_numpy(label.astype(np.int64))

    def _load_and_process_invalid(self, path, dims):
        """Helper method to load and process invalid mask data"""
        invalid = unpack(np.fromfile(path, dtype=np.uint8)).reshape(dims)
        return torch.from_numpy(invalid)


class LMSCNetDataset:
    def __init__(self, root_dir, phase):
        self.dataset = SemanticKITTI_Completion(root_dir, phase)
        self.label_processor = KITTILabelProcessor()
        self.filenames = self.dataset.filenames

        self.multi_scales = [1, 2, 4, 8]
        self.phase = phase

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx) -> dict:
        """Get multi-scale data for a single sample

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
            - occupancy: Point cloud data (original scale)
            - label: Original scale labels (1:1)
            - occluded: Original scale occluded mask
            - label_1_[2/4/8]: Downscaled labels at 1:2, 1:4, and 1:8 scales
            - invalid_1_[2/4/8]: Invalid masks at 1:2, 1:4, and 1:8 scales
        """
        # Get base data from SemanticKITTI_Completion
        base_data = self.dataset[idx]

        if self.phase == 'test':
            return base_data

        # Initialize multi-scale data dictionary
        multi_scale_data = {
            'label_1_2': None,
            'invalid_1_2': None,
            'label_1_4': None,
            'invalid_1_4': None,
            'label_1_8': None,
            'invalid_1_8': None,
        }

        # Process each scale (skip 1:1 as it's in base_data)
        for scale in self.multi_scales[1:]:
            try:
                suffix = f'_1_{scale}'
                VOXEL_DIMS_SCALED = tuple(int(dim / scale)
                                          for dim in VOXEL_DIMS)

                # Process labels
                label_path = f'{self.filenames[idx]}.label{suffix}'
                multi_scale_data[f'label{suffix}'] = self._load_and_process_label(
                    label_path, VOXEL_DIMS_SCALED)

                # Process invalid masks
                invalid_path = f'{self.filenames[idx]}.invalid{suffix}'
                multi_scale_data[f'invalid{suffix}'] = self._load_and_process_invalid(
                    invalid_path, VOXEL_DIMS_SCALED)

            except (FileNotFoundError, ValueError) as e:
                print(
                    f"Error processing scale 1:{scale} for idx {idx}: {str(e)}")
                continue

        return {**base_data, **multi_scale_data}

    def _load_and_process_label(self, path, dims):
        """Helper method to load and process label data"""
        label = np.fromfile(path, dtype=np.uint16).reshape(dims)
        label = self.label_processor.remap(label, mapping='learning')
        return torch.from_numpy(label.astype(np.int64))

    def _load_and_process_invalid(self, path, dims):
        """Helper method to load and process invalid mask data"""
        invalid = unpack(np.fromfile(path, dtype=np.uint8)).reshape(dims)
        return torch.from_numpy(invalid)


class LMSCNetDataLoader:
    def __init__(self, dataset, batch_size, shuffle, num_workers=4):
        """
        Wrapper around torch.utils.data.DataLoader for multi-scale data handling
        Args:
            dataset: LMSCNetDataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data
            num_workers: Number of workers for data loading
        """
        # Create DataLoader with custom collate function
        self.data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader)

    def _collate_fn(self, batch):
        """
        Custom collate function to handle multi-scale data with nested structure
        Args:
            batch: List of dictionaries containing multi-scale data
        Returns:
            Collated batch with nested structure for labels/invalids
        """
        collated_batch = {
            'occupancy': torch.stack([sample['occupancy'] for sample in batch]),
            'occluded': torch.stack([sample['occluded'] for sample in batch]),
            'label': {},
            'invalid': {}
        }

        scales = ['1_1', '1_2', '1_4', '1_8']
        for scale in scales:
            key_suffix = '' if scale == '1_1' else f'_{scale}'
            label_key = f'label{key_suffix}'
            invalid_key = f'invalid{key_suffix}'

            if label_key.strip('_') in batch[0]:
                try:
                    collated_batch['label'][scale] = torch.stack([
                        sample[label_key] for sample in batch
                    ])
                    collated_batch['invalid'][scale] = torch.stack([
                        sample[invalid_key] for sample in batch
                    ])
                except:
                    continue

        return collated_batch


if __name__ == "__main__":

    dataset = LMSCNetDataset(
        root_dir="/data/semanticKITTI/dataset/", phase="train")

    # Test dataset class first.
    print(len(dataset))
    print(dataset[0].keys())
    print(dataset[0]['occupancy'].shape)
    print(dataset[0]['label'].shape)
    print(dataset[0]['invalid'].shape)
    print(dataset[0]['occluded'].shape)
    print(dataset[0]['label_1_2'].shape)
    print(dataset[0]['invalid_1_2'].shape)
    print(dataset[0]['label_1_4'].shape)
    print(dataset[0]['invalid_1_4'].shape)
    print(dataset[0]['label_1_8'].shape)
    print(dataset[0]['invalid_1_8'].shape)
    print()

    # Test dataloader class
    dataloader = LMSCNetDataLoader(dataset, batch_size=10, shuffle=True)
    for batch in dataloader:
        print(batch.keys())
        print(batch['occupancy'].shape)
        print(batch['occluded'].shape)
        print()
        print(batch['label']['1_1'].shape)
        print(batch['label']['1_2'].shape)
        print(batch['label']['1_4'].shape)
        print(batch['label']['1_8'].shape)
        print()
        print(batch['invalid']['1_1'].shape)
        print(batch['invalid']['1_2'].shape)
        print(batch['invalid']['1_4'].shape)
        print(batch['invalid']['1_8'].shape)
        break
