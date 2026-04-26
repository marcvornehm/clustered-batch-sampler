import hashlib
import math
import pickle
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Generic, Optional, Protocol, TypeVar, Union

import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, RandomSampler, Sampler, SequentialSampler

Key = TypeVar("Key")


class SizedDataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int, /) -> Any: ...


def _stable_seed_from_key(key: Any) -> int:
    """Create a deterministic integer seed from a cluster key."""
    try:
        payload = pickle.dumps(key)
    except Exception:
        payload = repr(key).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


class ClusteredBatchSampler(Sampler, Generic[Key]):
    r"""Batch sampler that ensures each batch contains samples with the same key.

    Args:
        dataset (SizedDataset): Dataset used for sampling.
        key (Sequence[Key] | Callable[[Any], Key]): Either a sequence of keys
            (one per dataset item) or a callable that returns a key for each
            dataset item.
        batch_size (int): Size of mini-batch.
        shuffle (bool, optional): If `True` (default), sampler will shuffle the
            batches and clusters.
        seed (int, optional): Random seed used to shuffle the sampler if
            `shuffle=True`. This number should be identical across all processes
            in the distributed group. Default: `0`.
        drop_last_samples (bool, optional): If `True`, the sampler will drop the
            last batch of each cluster if its size would be less than
            `batch_size`. Default: `False`.
        drop_last_batches (bool, optional): If `True`, then the sampler will
            drop the tail of the data to make it evenly divisible across the
            number of replicas. If `False`, the sampler will add extra batches
            to make the data evenly divisible across the replicas.
            Default: `False`.
        distributed (bool, optional): Whether to use distributed sampling across
            multiple processes. Default: `False`.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, `world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within `num_replicas`.
            By default, `rank` is retrieved from the current distributed group.
    """

    def __init__(
        self,
        dataset: SizedDataset,
        key: Union[Sequence[Key], Callable[[Any], Key]],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last_samples: bool = False,
        drop_last_batches: bool = False,
        distributed: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, but got "
                f"batch_size={batch_size}"
            )

        if distributed:
            if (num_replicas is None or rank is None) and not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            if (num_replicas is None or rank is None) and not dist.is_initialized():
                raise RuntimeError(
                    "Distributed process group is not initialized. Call "
                    "torch.distributed.init_process_group() before creating "
                    "ClusteredBatchSampler with distributed=True, or pass "
                    "num_replicas and rank explicitly."
                )
            if num_replicas is None:
                num_replicas = dist.get_world_size()
            if rank is None:
                rank = dist.get_rank()
            if num_replicas <= 0 or not isinstance(num_replicas, int):
                raise ValueError(
                    "num_replicas should be a positive integer value, but got "
                    f"num_replicas={num_replicas}"
                )
            if rank >= num_replicas or rank < 0:
                raise ValueError(
                    "Invalid rank "
                    f"{rank}, rank should be in the interval [0, {num_replicas - 1}]"
                )
        else:
            if num_replicas is not None:
                warnings.warn(
                    "`num_replicas` is given but ignored because `distributed` is False"
                )
            if rank is not None:
                warnings.warn(
                    "`rank` is given but ignored because `distributed` is False"
                )
            num_replicas = 1
            rank = 0

        self.dataset = dataset
        self.key = key
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last_samples = drop_last_samples
        self.drop_last_batches = drop_last_batches
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self._clusters: dict[Key, list[int]] = self._create_clusters()
        self._samplers: dict[Key, BatchSampler] = {
            k: self._create_cluster_sampler(v) for k, v in self._clusters.items()
        }

        sum_len_samplers = sum(len(s) for s in self._samplers.values())
        if self.drop_last_batches and sum_len_samplers % self.num_replicas != 0:
            self.num_batches = math.ceil(
                (sum_len_samplers - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_batches = math.ceil(sum_len_samplers / self.num_replicas)
        self.total_size = self.num_batches * self.num_replicas

    def _create_clusters(self) -> dict[Key, list[int]]:
        # create mapping of cluster key to list of dataset indices
        clusters = defaultdict(list)

        if isinstance(self.key, Sequence):
            if len(self.key) != len(self.dataset):
                raise ValueError("Length of key sequence must match length of dataset")
            keys = self.key
        else:
            keys = [self.key(self.dataset[i]) for i in range(len(self.dataset))]

        for i, k in enumerate(keys):
            try:
                clusters[k].append(i)
            except TypeError as exc:
                raise TypeError(
                    "Cluster keys must be hashable so they can be grouped into batches"
                ) from exc

        return clusters

    def _create_cluster_sampler(self, indices: list[int]) -> BatchSampler:
        if len(indices) == 0:
            raise ValueError("Cannot create sampler for empty cluster")

        if self.shuffle:
            # no need to set the generator here, it will be set in __iter__
            sampler = RandomSampler(indices)
        else:
            sampler = SequentialSampler(indices)
        batch_sampler = BatchSampler(
            sampler,
            batch_size=self.batch_size,
            drop_last=self.drop_last_samples,
        )
        return batch_sampler

    def __iter__(self) -> Iterator[list[int]]:
        # list keys of all batches
        batch_keys: list[Key] = []
        for k, sampler in self._samplers.items():
            batch_keys.extend([k] * len(sampler))

        # shuffle deterministically
        if self.shuffle:
            # shuffle batches based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(batch_keys), generator=g).tolist()
            batch_keys = [batch_keys[i] for i in perm]

            # shuffle each cluster based on epoch, seed, and cluster key
            for k, sampler in self._samplers.items():
                samp = sampler.sampler
                if isinstance(samp, RandomSampler):
                    g = torch.Generator()
                    g.manual_seed(self.seed + self.epoch + _stable_seed_from_key(k))
                    samp.generator = g

        # create an iterator for each cluster sampler
        cluster_iters: dict[Key, Iterator[list[int]]] = {
            k: iter(sampler) for k, sampler in self._samplers.items()
        }

        # generate batches
        batches = []
        for k in batch_keys:
            cluster_indices = next(cluster_iters[k])
            dataset_indices = [self._clusters[k][i] for i in cluster_indices]
            batches.append(dataset_indices)

        # make it evenly divisible across replicas
        if not self.drop_last_batches:
            # add extra batches
            padding_size = self.total_size - len(batches)
            if padding_size <= len(batches):
                batches += batches[:padding_size]
            else:
                batches += (batches * math.ceil(padding_size / len(batches)))[
                    :padding_size
                ]
        else:
            # remove extra batches
            batches = batches[: self.total_size]

        # distribute batches to replicas
        batches = batches[self.rank : self.total_size : self.num_replicas]
        assert len(batches) == self.num_batches

        return iter(batches)

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
