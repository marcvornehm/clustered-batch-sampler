# Clustered Batch Sampler

A PyTorch BatchSampler that produces batches from samples clustered by keys. Ensures that each batch contains only samples belonging to the same cluster, which is useful for tasks like training on images with different sizes or when samples need to be processed together based on shared properties.

## Installation

```bash
pip install clustered-batch-sampler
```

## Features

- **Cluster integrity**: Guarantees each batch contains samples from only one cluster
- **Flexible keys**: Use a list of explicit keys or a callable function to define clusters. Keys must be hashable. Using a callable function eagerly evaluates the whole dataset when the sampler is initialized.
- **Shuffling**: Supports deterministic shuffling of both clusters and samples within clusters
- **Distributed training**: Support for distributed sampling, following PyTorch's DistributedSampler
- **Drop options**: Control whether to drop incomplete batches, following PyTorch's BatchSampler

## Usage

### Example Usage

```python
import torch
from torch.utils.data import DataLoader, Dataset

from clustered_batch_sampler import ClusteredBatchSampler


class RandomImageDataset(Dataset):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        self.sizes = [192, 224, 256, 288, 320]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= self.num_samples:
            raise IndexError("Index out of range")
        size = self.sizes[idx % len(self.sizes)]
        image = torch.randn(3, size, size)
        return image


dataset = RandomImageDataset(num_samples=100)
key_fn = lambda x: x.shape[-2:]
sampler = ClusteredBatchSampler(dataset, key=key_fn, batch_size=16)
dataloader = DataLoader(dataset, batch_sampler=sampler)

for b in dataloader:
    print(b.shape)
```

Output:
```
torch.Size([16, 3, 256, 256])
torch.Size([16, 3, 192, 192])
torch.Size([16, 3, 288, 288])
torch.Size([4, 3, 256, 256])
torch.Size([16, 3, 224, 224])
torch.Size([16, 3, 320, 320])
torch.Size([4, 3, 192, 192])
torch.Size([4, 3, 320, 320])
torch.Size([4, 3, 288, 288])
torch.Size([4, 3, 224, 224])
```

### Using Explicit Keys

```python
keys = [key_fn(dataset[i]) for i in range(len(dataset))]
sampler = ClusteredBatchSampler(dataset, key=keys, batch_size=16)
```

### Distributed Training

Similar to PyTorch's `DistributedSampler`, calling the `set_epoch()` method at the beginning of each epoch **before** creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be always used.

If `distributed=True`, either initialize the current distributed process group first or pass `num_replicas` and `rank` explicitly.

See also: https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler

### Notes and Caveats

- The sampler assumes a map-style dataset that supports both `__len__` and indexed `__getitem__` access.
- When `key` is a callable, keys are computed eagerly for every dataset item during sampler initialization.
- Keys must be hashable because they are used to group samples into clusters.
- `drop_last_samples` drops incomplete batches inside each cluster.
- `drop_last_batches` only affects distributed mode and controls whether whole batches are dropped or repeated to divide work evenly across replicas.
- Shuffling is deterministic across epochs and across distributed workers when the same `seed` and `epoch` are used.

### Parameters

- **dataset**: PyTorch dataset with `__len__` and `__getitem__`
- **key**: Either a sequence of hashable keys (one per sample) or a callable returning a hashable key for each item
- **batch_size**: Size of mini-batch
- **shuffle**: Whether to shuffle batches and samples (default: `True`)
- **seed**: Random seed for reproducibility (default: `0`)
- **drop_last_samples**: Drop incomplete batches within clusters (default: `False`)
- **drop_last_batches**: Drop batches to make data evenly divisible across replicas in distributed mode (default: `False`)
- **distributed**: Enable distributed sampling across multiple processes (default: `False`)
- **num_replicas**: Number of processes in distributed training (optional, by default retrieved from the current initialized distributed group)
- **rank**: Process rank in distributed training (optional, by default retrieved from the current initialized distributed group)

## Development And Release

Install development tools:

```bash
pip install -e .[dev]
```

Run local preflight checks before publishing:

```bash
ruff check .
pytest -q
python -m build
twine check dist/*
```

### GitHub Actions Workflows

- CI workflow: `.github/workflows/ci.yml`
    - Runs lint, tests, package build, and `twine check` on push and pull requests.
- Publish workflow: `.github/workflows/publish.yml`
    - Runs the same validation checks, builds package artifacts, and publishes with trusted publishing.
    - Triggers on published GitHub releases (publishes to PyPI).
    - Also supports manual dispatch for TestPyPI or PyPI.
