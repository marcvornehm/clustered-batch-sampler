import math
import os
import random
import subprocess
import sys
from collections import Counter
from itertools import chain

import pytest
import torch.distributed as dist
from torch.utils.data import Dataset

from clustered_batch_sampler import ClusteredBatchSampler


class SimpleDataset(Dataset):
    def __init__(self, size: int):
        self.data = list(range(size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


@pytest.fixture
def num_samples(request):
    return request.param


@pytest.fixture
def num_clusters(request):
    return request.param


@pytest.fixture
def batch_size(request):
    return request.param


@pytest.fixture
def shuffle(request):
    return request.param


@pytest.fixture
def drop_last(request):
    return request.param


@pytest.fixture
def dataset(num_samples):
    return SimpleDataset(num_samples)


@pytest.fixture
def keys(num_samples, num_clusters):
    random.seed(42)
    return random.choices(range(num_clusters), k=num_samples)


@pytest.mark.parametrize("shuffle", [True, False], indirect=True)
@pytest.mark.parametrize("batch_size", [1, 8], indirect=True)
@pytest.mark.parametrize("num_clusters", [1, 5], indirect=True)
@pytest.mark.parametrize("num_samples", [1, 3, 100], indirect=True)
def test_batches(dataset, num_samples, keys, batch_size, shuffle):
    sampler = ClusteredBatchSampler(
        dataset, keys, batch_size=batch_size, shuffle=shuffle, drop_last_samples=False
    )
    batches = list(sampler)

    assert len(batches) == len(sampler)
    assert len(batches) == sum(
        math.ceil(count / batch_size) for count in Counter(keys).values()
    )
    assert all(len(batch) <= batch_size for batch in batches)
    assert sum(len(batch) for batch in batches) == num_samples
    assert set(chain.from_iterable(batches)) == set(range(num_samples))
    assert all(keys[idx] == keys[batch[0]] for batch in batches for idx in batch)


@pytest.mark.parametrize("shuffle", [True, False], indirect=True)
@pytest.mark.parametrize("batch_size", [1, 8], indirect=True)
@pytest.mark.parametrize("num_clusters", [1, 5], indirect=True)
@pytest.mark.parametrize("num_samples", [1, 3, 100], indirect=True)
def test_batches_drop_last(dataset, num_samples, keys, batch_size, shuffle):
    sampler = ClusteredBatchSampler(
        dataset, keys, batch_size=batch_size, shuffle=shuffle, drop_last_samples=True
    )
    batches = list(sampler)

    assert len(batches) == len(sampler)
    assert len(batches) == sum(
        math.floor(count / batch_size) for count in Counter(keys).values()
    )
    assert all(len(batch) == batch_size for batch in batches)
    assert (
        sum(len(batch) for batch in batches) == len(sampler) * batch_size <= num_samples
    )
    assert set(chain.from_iterable(batches)).issubset(set(range(num_samples)))
    assert all(keys[idx] == keys[batch[0]] for batch in batches for idx in batch)


@pytest.mark.parametrize("num_clusters", [1, 5], indirect=True)
@pytest.mark.parametrize("num_samples", [8, 100], indirect=True)
def test_callable_key(dataset, num_samples, num_clusters):
    def key_fn(item):
        return item % num_clusters

    sampler = ClusteredBatchSampler(dataset, key_fn, batch_size=8, shuffle=False)
    batches = list(sampler)

    assert len(batches) == len(sampler)
    assert len(batches) == sum(
        math.ceil(count / 8)
        for count in Counter(key_fn(i) for i in range(num_samples)).values()
    )
    assert all(len(batch) <= 8 for batch in batches)
    assert sum(len(batch) for batch in batches) == num_samples
    assert set(chain.from_iterable(batches)) == set(range(num_samples))
    assert all(key_fn(idx) == key_fn(batch[0]) for batch in batches for idx in batch)


@pytest.mark.parametrize("drop_last", [True, False], indirect=True)
@pytest.mark.parametrize("num_clusters", [1, 5], indirect=True)
@pytest.mark.parametrize("num_samples", [100], indirect=True)
def test_shuffle_determinism(dataset, keys, drop_last):
    sampler1 = ClusteredBatchSampler(
        dataset, keys, batch_size=8, shuffle=True, drop_last_samples=drop_last
    )
    batches1 = list(sampler1)
    batches2 = list(sampler1)
    assert batches1 == batches2

    sampler2 = ClusteredBatchSampler(
        dataset, keys, batch_size=8, shuffle=True, drop_last_samples=drop_last
    )
    batches3 = list(sampler2)
    assert batches1 == batches3


@pytest.mark.parametrize("drop_last", [True, False], indirect=True)
@pytest.mark.parametrize("num_clusters", [1, 5], indirect=True)
@pytest.mark.parametrize("num_samples", [8, 100], indirect=True)
def test_shuffle_seed(dataset, keys, drop_last):
    sampler1 = ClusteredBatchSampler(
        dataset, keys, batch_size=8, shuffle=True, seed=42, drop_last_samples=drop_last
    )
    sampler2 = ClusteredBatchSampler(
        dataset, keys, batch_size=8, shuffle=True, seed=43, drop_last_samples=drop_last
    )
    batches1 = list(sampler1)
    batches2 = list(sampler2)
    if len(batches1) == 0 and len(batches2) == 0:
        pytest.skip("Batches are the same due to small dataset or few clusters")
    assert batches1 != batches2


@pytest.mark.parametrize("drop_last", [True, False], indirect=True)
@pytest.mark.parametrize("num_clusters", [1, 5], indirect=True)
@pytest.mark.parametrize("num_samples", [8, 100], indirect=True)
def test_shuffle_epoch(dataset, keys, drop_last):
    sampler = ClusteredBatchSampler(
        dataset, keys, batch_size=8, shuffle=True, drop_last_samples=drop_last
    )
    batches1 = list(sampler)
    sampler.set_epoch(1)
    batches2 = list(sampler)
    if len(batches1) == 0 and len(batches2) == 0:
        pytest.skip("Batches are the same due to small dataset or few clusters")
    assert batches1 != batches2


@pytest.mark.parametrize("shuffle", [True, False], indirect=True)
@pytest.mark.parametrize("batch_size", [1, 8], indirect=True)
@pytest.mark.parametrize("num_clusters", [1, 5], indirect=True)
@pytest.mark.parametrize("num_samples", [3, 100], indirect=True)
def test_distributed(dataset, num_samples, keys, batch_size, shuffle):
    world_size = 4
    batches = []
    sampler_lens = []

    for rank in range(world_size):
        sampler = ClusteredBatchSampler(
            dataset,
            keys,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last_batches=False,
            distributed=True,
            rank=rank,
            num_replicas=world_size,
        )
        batches.extend(list(sampler))
        sampler_lens.append(len(sampler))

    assert sum(sampler_lens) == len(batches)
    assert all(len_ == sampler_lens[0] for len_ in sampler_lens)
    assert len(batches) >= sum(
        math.ceil(count / batch_size) for count in Counter(keys).values()
    )
    assert sum(len(batch) for batch in batches) >= num_samples
    assert set(chain.from_iterable(batches)) == set(range(num_samples))
    assert all(keys[idx] == keys[batch[0]] for batch in batches for idx in batch)


@pytest.mark.parametrize("shuffle", [True, False], indirect=True)
@pytest.mark.parametrize("batch_size", [1, 8], indirect=True)
@pytest.mark.parametrize("num_clusters", [1, 5], indirect=True)
@pytest.mark.parametrize("num_samples", [3, 100], indirect=True)
def test_distributed_drop_last(dataset, num_samples, keys, batch_size, shuffle):
    world_size = 4
    batches = []
    sampler_lens = []

    for rank in range(world_size):
        sampler = ClusteredBatchSampler(
            dataset,
            keys,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last_batches=True,
            distributed=True,
            rank=rank,
            num_replicas=world_size,
        )
        batches.extend(list(sampler))
        sampler_lens.append(len(sampler))

    assert sum(sampler_lens) == len(batches)
    assert sum(len(batch) for batch in batches) <= num_samples
    assert set(chain.from_iterable(batches)).issubset(set(range(num_samples)))
    assert all(keys[idx] == keys[batch[0]] for batch in batches for idx in batch)


@pytest.mark.parametrize("num_clusters", [5], indirect=True)
@pytest.mark.parametrize("num_samples", [100], indirect=True)
def test_invalid_batch_size(dataset, keys):
    with pytest.raises(ValueError, match="batch_size should be a positive integer"):
        ClusteredBatchSampler(dataset, keys, batch_size=0)
    with pytest.raises(ValueError, match="batch_size should be a positive integer"):
        ClusteredBatchSampler(dataset, keys, batch_size=1.5)  # type: ignore
    with pytest.raises(ValueError, match="batch_size should be a positive integer"):
        ClusteredBatchSampler(dataset, keys, batch_size=True)


@pytest.mark.parametrize("num_clusters", [5], indirect=True)
@pytest.mark.parametrize("num_samples", [100], indirect=True)
def test_mismatched_key_length(dataset, keys):
    with pytest.raises(
        ValueError, match="Length of key sequence must match length of dataset"
    ):
        ClusteredBatchSampler(dataset, keys[:-1], batch_size=8)
    with pytest.raises(
        ValueError, match="Length of key sequence must match length of dataset"
    ):
        ClusteredBatchSampler(dataset, keys + [0], batch_size=8)


@pytest.mark.parametrize("num_clusters", [5], indirect=True)
@pytest.mark.parametrize("num_samples", [100], indirect=True)
def test_invalid_distributed_args(dataset, keys):
    with pytest.raises(ValueError, match="rank should be in the interval"):
        ClusteredBatchSampler(
            dataset, keys, batch_size=8, distributed=True, num_replicas=4, rank=-1
        )
    with pytest.raises(ValueError, match="rank should be in the interval"):
        ClusteredBatchSampler(
            dataset, keys, batch_size=8, distributed=True, num_replicas=4, rank=4
        )
    with pytest.raises(ValueError, match="num_replicas should be a positive integer"):
        ClusteredBatchSampler(
            dataset, keys, batch_size=8, distributed=True, num_replicas=0, rank=0
        )


@pytest.mark.parametrize("num_clusters", [5], indirect=True)
@pytest.mark.parametrize("num_samples", [100], indirect=True)
def test_non_distributed_warns_about_unused_rank_and_num_replicas(dataset, keys):
    with pytest.warns(UserWarning, match="`num_replicas` is given but ignored"):
        ClusteredBatchSampler(
            dataset, keys, batch_size=8, distributed=False, num_replicas=4
        )

    with pytest.warns(UserWarning, match="`rank` is given but ignored"):
        ClusteredBatchSampler(dataset, keys, batch_size=8, distributed=False, rank=1)


@pytest.mark.parametrize("num_samples", [4], indirect=True)
def test_unhashable_keys_raise_clear_error(dataset):
    keys = [[0], [1], [0], [1]]

    with pytest.raises(TypeError, match="Cluster keys must be hashable"):
        ClusteredBatchSampler(dataset, keys, batch_size=2)


@pytest.mark.parametrize("num_samples", [12], indirect=True)
def test_tuple_keys_work(dataset):
    keys = [(i % 2, i % 3) for i in range(len(dataset))]
    sampler = ClusteredBatchSampler(dataset, keys, batch_size=2, shuffle=True)

    for batch in sampler:
        assert all(keys[idx] == keys[batch[0]] for idx in batch)


@pytest.mark.parametrize("num_clusters", [5], indirect=True)
@pytest.mark.parametrize("num_samples", [100], indirect=True)
def test_distributed_requires_initialized_group_if_rank_and_world_size_missing(
    dataset, keys, monkeypatch
):
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)

    with pytest.raises(
        RuntimeError, match="Distributed process group is not initialized"
    ):
        ClusteredBatchSampler(dataset, keys, batch_size=8, distributed=True)


def test_shuffle_is_stable_across_python_hash_seed():
    code = r'''
import json
from torch.utils.data import Dataset
from clustered_batch_sampler import ClusteredBatchSampler

class D(Dataset):
    def __len__(self):
        return 24

    def __getitem__(self, idx):
        return idx

dataset = D()
keys = [f"cluster_{i % 3}" for i in range(len(dataset))]
sampler = ClusteredBatchSampler(
    dataset,
    keys,
    batch_size=2,
    shuffle=True,
    seed=123,
    distributed=True,
    num_replicas=2,
    rank=0,
)
print(json.dumps(list(sampler)))
'''
    env_a = dict(os.environ)
    env_b = dict(os.environ)
    env_a["PYTHONHASHSEED"] = "1"
    env_b["PYTHONHASHSEED"] = "2"

    proc_a = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env_a,
    )
    proc_b = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env_b,
    )

    assert proc_a.stdout.strip() == proc_b.stdout.strip()


@pytest.mark.parametrize("num_clusters", [5], indirect=True)
@pytest.mark.parametrize("num_samples", [0], indirect=True)
def test_empty_dataset(dataset, keys):
    sampler = ClusteredBatchSampler(dataset, keys, batch_size=8)
    batches = list(sampler)
    assert len(batches) == 0
