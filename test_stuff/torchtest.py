

import torch

points = torch.zeros(size = (1,20,3))
conf_logits = torch.ones(size = (1,20))
outputs = torch.cat([points, conf_logits.unsqueeze(-1)], dim=2)

print(points.shape)
print(conf_logits.shape)
print(conf_logits.unsqueeze(-1).shape)
print(outputs.shape)

print(points)
print(outputs)