"""
Utility classes for dataset / dataloader plumbing:
  - SharedEpoch: a multiprocessing-safe integer epoch holder
  - DataInfo: simple container for (dataloader, sampler, shared_epoch)
"""

from dataclasses import dataclass
from multiprocessing import Value

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class SharedEpoch:
    """
    Simple shared integer used to keep track of the current epoch
    across workers / processes (if needed).
    """

    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch: int):
        self.shared_epoch.value = epoch

    def get_value(self) -> int:
        return self.shared_epoch.value


@dataclass
class DataInfo:
    """
    Bundle dataloader + sampler + shared_epoch with a .set_epoch() helper.
    """

    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch: int):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)