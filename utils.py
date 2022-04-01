from contextlib import contextmanager

import torch
import torch.nn.functional as F

def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]

@contextmanager
def rename_logger(logger, new_name):
    old_name = logger.name
    if new_name is not None:
        logger.name = new_name
    yield logger
    logger.name = old_name

def should_stop_early(patience, valid_auroc: float) -> bool:
    def is_better(a, b):
        return a > b
    
    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_auroc, prev_best):
        should_stop_early.best = valid_auroc
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= patience:
            print("early stop since valid performance hasn't improved for last {} runs".format(
                    patience
                )
            )
            return True
        else:
            return False

def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}
    
    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x
    
    return _apply(sample)

def move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor):
        return tensor.to(device=device, non_blocking=True)
    
    return apply_to_sample(_move_to_cuda, sample)

def prepare_sample(sample):
    if torch.cuda.is_available():
        sample = move_to_cuda(sample)
    
    return sample

def log_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.log_softmax(x.float(), dim=dim)
    else:
        return F.log_softmax(x, dim=dim, dtype=torch.float32)