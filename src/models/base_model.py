import torch.nn as nn


class BaseModel(nn.Module):
    def forward(self, *inputs):
        raise NotImplementedError

    def __str__(self):
        ret = super().__str__()
        lines = ["", "Trainable parameters by tensor:"]
        total = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                count = param.numel()
                total += count
                lines.append(f"  {name}: {count:,}")
        lines.append(f"Total trainable parameters: {total:,}")
        return ret + "\n".join(lines)
