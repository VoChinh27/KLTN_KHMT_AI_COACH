"""Small CPU count regressor. Import torch only in this opt-in module."""
import torch
from torch import nn
from .contract import Config, INPUT_SIZE


def setup(config=Config()):
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_threads)
    torch.use_deterministic_algorithms(True)


def collate(samples):
    if not samples:
        raise ValueError('Empty batch')
    lengths = torch.tensor([len(s['values']) for s in samples], dtype=torch.long)
    if (lengths <= 0).any():
        raise ValueError('Empty sequence')
    x = torch.zeros(len(samples), int(lengths.max()), INPUT_SIZE)
    complete = torch.zeros(x.shape[:2], dtype=torch.bool)
    for i, sample in enumerate(samples):
        tensor = torch.tensor(sample['values'], dtype=torch.float32)
        mask = torch.tensor(sample['complete'], dtype=torch.bool)
        if tensor.shape != (lengths[i],INPUT_SIZE) or mask.shape != (lengths[i],) or not torch.isfinite(tensor).all():
            raise ValueError('Invalid sample tensor/mask')
        x[i,:lengths[i]] = tensor
        complete[i,:lengths[i]] = mask
    duration = torch.tensor([s['duration_seconds'] for s in samples], dtype=torch.float32)
    if not torch.isfinite(duration).all() or (duration<=0).any() or not complete.any(dim=1).all():
        raise ValueError('No valid support/duration')
    targets = [s['target'] for s in samples]
    if any(t is not None and (type(t) is not int or t<0) for t in targets):
        raise ValueError('Targets must be nonnegative integer AI counts or null')
    return dict(x=x, lengths=lengths, complete=complete, duration=duration,
                target=torch.tensor([0 if t is None else t for t in targets],dtype=torch.float32),
                target_mask=torch.tensor([t is not None for t in targets],dtype=torch.bool),
                video_ids=[s['video_id'] for s in samples])


class CountGRU(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.cell = nn.GRUCell(INPUT_SIZE,config.hidden_size)
        self.head = nn.Linear(config.hidden_size+1,1)

    def forward(self, batch):
        x, complete = batch['x'], batch['complete']
        h = x.new_zeros(x.shape[0],self.cell.hidden_size)
        total = torch.zeros_like(h)
        weights = x.new_zeros(x.shape[0],1)
        # Never bridge an incomplete frame with recurrent state, including padding.
        for t in range(x.shape[1]):
            valid = (complete[:,t] & (t<batch['lengths'])).unsqueeze(1)
            h = torch.where(valid,self.cell(x[:,t],h),torch.zeros_like(h))
            total = total+h
            weights = weights+valid.to(x.dtype)
        if (weights<=0).any():
            raise ValueError('Count requires observed support')
        pooled = total/weights
        return nn.functional.softplus(self.head(torch.cat([pooled,batch['duration'][:,None]],dim=1))).squeeze(1)


def count_loss(prediction,batch):
    mask = batch['target_mask']
    if not mask.any() or not torch.isfinite(prediction).all():
        raise ValueError('No supervised targets or non-finite prediction')
    return nn.functional.smooth_l1_loss(prediction[mask],batch['target'][mask],beta=1.0,reduction='mean')
