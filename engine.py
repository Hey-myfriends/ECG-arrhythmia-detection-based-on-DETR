from typing import Iterable
import torch, math, sys, pdb
from tqdm import tqdm
from collections import deque, defaultdict

class SmootheValue(object):
    def __init__(self, window_size=20) -> None:
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
    data_loader: Iterable, optimizer: torch.optim.Optimizer,
    device: torch.device, epoch: int, max_norm: float = 0):
    
    model.train()
    criterion.train()
    header = 'Epoch: [{}] Training: '.format(epoch)
    meters = defaultdict(SmootheValue)
    meters["loss"] = SmootheValue(window_size=20)
    meters["lr"] = SmootheValue(window_size=1)
    pbar = tqdm(data_loader)

    # pdb.set_trace()
    for samples, targets in pbar:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        for k in loss_dict.keys():
            if k not in meters.keys():
                meters[k] = SmootheValue(window_size=20)
            meters[k].update(loss_dict[k].cpu().item())
            
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k]*weight_dict[k] for k in loss_dict.keys() if k in weight_dict.keys())

        loss_value = losses.cpu().item()

        if not math.isfinite(loss_value):
            print(f"loss is {loss_value}, stop training...")
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        meters["loss"].update(loss_value)
        meters["lr"].update(optimizer.param_groups[0]["lr"])
        pbar.set_description("Train-> lr: {:.5f} loss: {:.4f} loss_ce: {:.4f}, loss_bbox: {:.4f}, loss_giou: {:.4f}, class_error: {:.4f}%".format(
            meters["lr"].value, meters["loss"].avg, meters["loss_ce"].avg, meters["loss_bbox"].avg, meters["loss_giou"].avg, meters["class_error"].avg
        ))

    stats = {k: meter.global_avg for k, meter in meters.items()}
    print(header, stats)
    return stats

@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor: torch.nn.Module,
            data_loader: Iterable, device: torch.device, output_dir: str):
    # pdb.set_trace()
    model.eval()
    criterion.eval()
    meters = defaultdict(SmootheValue)
    meters["loss"] = SmootheValue(window_size=20)
    pbar = tqdm(data_loader)

    for samples, targets in pbar:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        for k in loss_dict.keys():
            if k not in meters.keys():
                meters[k] = SmootheValue(window_size=20)
            meters[k].update(loss_dict[k].cpu().item())

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k]*weight_dict[k] for k in loss_dict.keys() if k in weight_dict.keys())
        loss_value = losses.cpu().item()
        meters["loss"].update(loss_value)
        pbar.set_description("Test-> loss: {:.4f} loss_ce: {:.4f}, loss_bbox: {:.4f}, loss_giou: {:.4f}, class_error: {:.4f}%".format(
            meters["loss"].avg, meters["loss_ce"].avg, meters["loss_bbox"].avg, meters["loss_giou"].avg, meters["class_error"].avg
        ))
    stats = {k: meter.global_avg for k, meter in meters.items()}
    print("Testing: ", stats, "\n\n")
    return stats
