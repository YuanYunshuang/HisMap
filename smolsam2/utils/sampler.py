import torch
from torch.utils.data import Sampler
from typing import Sequence, Iterator, Optional

default_prob_dict = {
    "hameln": 1.0,
    "bayern": 1.0,
    "siegfried": 3.0,
}


def get_sampler(dataset, subset_prob_dict=None, use_default=False) -> Sampler[int]:
    if subset_prob_dict is None:
        if use_default:
            subset_prob_dict = default_prob_dict
        else:
            return ProbabilitySampler(probs=[1.0] * len(dataset))
    
    probs = []
    for filename in dataset.file_list:
        dataset_name = filename.split('.')[0]
        if dataset_name in subset_prob_dict:
            probs.append(subset_prob_dict[dataset_name])
        else:
            probs.append(1.0)


class ProbabilitySampler(Sampler[int]):
    def __init__(
        self,
        probs: Sequence[float] | torch.Tensor,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__(None)
        p = torch.as_tensor(probs, dtype=torch.double)
        if p.ndim != 1:
            raise ValueError("probs must be a 1D array/tensor")
        if p.numel() == 0:
            raise ValueError("probs must be non-empty")
        p = p.clamp_min(0)
        s = p.sum().item()
        if s <= 0:
            raise ValueError("sum(probs) must be > 0")
        self.probs = (p / s).cpu()
        self.N = self.probs.numel()
        self.num_samples = int(num_samples) if num_samples is not None else self.N
        if not replacement and self.num_samples > self.N:
            raise ValueError("num_samples cannot exceed dataset size when replacement=False")
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        idx = torch.multinomial(
            self.probs,
            num_samples=self.num_samples,
            replacement=self.replacement,
            generator=self.generator,
        )
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self.num_samples