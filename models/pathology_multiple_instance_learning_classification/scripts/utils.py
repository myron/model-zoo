import collections.abc
import csv

import numpy as np
import torch
import torch.distributed as dist

from torch.utils.data.dataloader import default_collate

from monai.config import KeysCollection
from monai.transforms import MapTransform

from monai.inferers import SlidingWindowInferer
from torch.cuda.amp import autocast



class PatchInferer(SlidingWindowInferer):
    def __init__(self, amp=True, max_tiles=44, roi_size=128, **kwargs) -> None:
        super().__init__(roi_size=roi_size, **kwargs)
        self.amp = amp
        self.max_tiles = max_tiles
    
    @torch.no_grad()
    def __call__(self, inputs: torch.Tensor, network: torch.nn.Module) -> torch.Tensor:
        
        network.eval()

        distributed = dist.is_available() and dist.is_initialized()
        model = network if not distributed else network.module
        calc_head = model.calc_head
        max_tiles = self.max_tiles

        data = inputs.as_subclass(torch.Tensor)

        with autocast(enabled=self.amp):

            if max_tiles is not None and data.shape[1] > max_tiles:

                logits = []

                for i in range(int(np.ceil(data.shape[1] / float(max_tiles)))):
                    data_slice = data[:, i * max_tiles : (i + 1) * max_tiles]
                    logits_slice = model(data_slice, no_head=True)
                    logits.append(logits_slice)

                logits = torch.cat(logits, dim=1)
                logits = calc_head(logits)

            else:
                logits = model(data)


        return logits


class LabelEncodeIntegerGraded(MapTransform):
    """
    Convert an integer label to encoded array representation of length num_classes,
    with 1 filled in up to label index, and 0 otherwise. For example for num_classes=5,
    embedding of 2 -> (1,1,0,0,0)

    Args:
        num_classes: the number of classes to convert to encoded format.
        keys: keys of the corresponding items to be transformed. Defaults to ``'label'``.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(self, num_classes: int, keys: KeysCollection = "label", allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.num_classes = num_classes

    def __call__(self, data):

        d = dict(data)
        for key in self.keys:
            label = int(d[key])
            lz = np.zeros(self.num_classes, dtype=np.float32)
            lz[:label] = 1.0
            d[key] = lz

        return d


class LabelDecodeIntegerGraded(MapTransform):

    def __init__(self,  keys: KeysCollection = "pred", allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):

        print('LabelDecodeIntegerGraded keys', data.keys())

        d = dict(data)
        for key in self.keys:
            # d[key] = d[key].sigmoid().sum(1).detach().round()
            d[key] = d[key].sigmoid().sum(0).detach().round() #sum of channel dim

        return d

def label_decode(xall):
    print('label_decode', xall.shape)
    xall = [x.sigmoid().sum(0).detach().round() for x in xall]
    return xall

def list_data_collate(batch: collections.abc.Sequence):
    """
    Combine instances from a list of dicts into a single dict, by stacking them along first dim
    [{'image' : 3xHxW}, {'image' : 3xHxW}, {'image' : 3xHxW}...] - > {'image' : Nx3xHxW}
    followed by the default collate which will form a batch BxNx3xHxW
    """

    for i, item in enumerate(batch):
        data = item[0]
        data["image"] = torch.stack([ix["image"] for ix in item], dim=0)
        batch[i] = data
    return default_collate(batch)


def write_csv_row(filename, row, mode="a"):
    with open(filename, mode, encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

