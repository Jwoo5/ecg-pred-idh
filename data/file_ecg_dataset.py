import warnings
import os
import sys

import numpy as np
import scipy.io

import torch
from torch.utils.data import Dataset

class FileECGDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        max_sample_size=None,
        min_sample_size=0,
        pad=True,
        label_key=None, #should be a choice of ['idh_a', 'idh_b', 'idh_ab']
    ):
        super().__init__()

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.pad = pad
        self.label_key = label_key

        self.skipped = 0
        self.fnames = []
        self.skipped_indices = set()

        with open(manifest_path, 'r') as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split('\t')
                assert len(items) == 2, line

                self.fnames.append(items[0])
        
        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            warnings.warn("Could not create a pyarrow array. Please install pyarrow for better performance.")
            pass
    
    def crop_to_max_size(self, data, target_size):
        size = len(data)
        diff = size - target_size
        if diff <= 0:
            return data
        
        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return data[start:end]
    
    def collator(self, samples):
        samples = [s for s in samples if s['source'] is not None]
        if len(samples) == 0:
            return {}
        
        sources = [s['source'] for s in samples]
        sizes = [s.size(-1) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)
        
        collated_sources = sources[0].new_zeros((len(sources), len(sources[0]), target_size))
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((source.shape[0], -diff, ), 0.0)], dim=-1
                )
                padding_mask[i, :, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)
        
        input = {'source': collated_sources}
        out = {'label': torch.cat([s['label'] for s in samples])}

        if self.pad:
            input['padding_mask'] = padding_mask
        
        out['net_input'] = input
        return out
    
    def __getitem__(self, index):
        path = os.path.join(self.root_dir, str(self.fnames[index]))

        data = scipy.io.loadmat(path)

        feats = torch.from_numpy(data['feats']).float()

        res = {'source': feats}

        if self.label_key:
            label = torch.from_numpy(data[self.label_key][0])
        else:
            label = torch.from_numpy(data['label'][0])


        res['label'] = label

        return res
    
    def __len__(self):
        return len(self.fnames)