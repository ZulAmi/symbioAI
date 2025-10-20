import torch
import torch.nn as nn
import torch.nn.functional as F

from training.causal_der import create_causal_der_engine

# Tiny CNN for smoke test
class TinyNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, num_classes)
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


def make_synthetic_task(n=64, num_classes=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return x, y


def train_one_step(engine, model, optimizer, x, y, task_id, device):
    model.train()
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad(set_to_none=True)
    out = model(x)
    loss, info = engine.compute_loss(model, x, y, out, task_id)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        logits = out.detach()
        engine.store(x.detach().cpu(), y.detach().cpu(), logits.detach().cpu(), task_id, model=model)
    return info


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyNet(num_classes=10).to(device)
    engine = create_causal_der_engine(
        alpha=0.5, beta=0.5, buffer_size=256,
        causal_weight=0.7, use_causal_sampling=True,
        temperature=2.0, mixed_precision=True,
        importance_weight_replay=True,
        use_mir_sampling=True, mir_candidate_factor=3,
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Two synthetic tasks
    x1, y1 = make_synthetic_task(n=64, seed=1)
    x2, y2 = make_synthetic_task(n=64, seed=2)

    for epoch in range(2):
        info1 = train_one_step(engine, model, opt, x1, y1, task_id=0, device=device)
        info2 = train_one_step(engine, model, opt, x2, y2, task_id=1, device=device)
        print(f"Epoch {epoch} | Task0 loss: {info1['total_loss']:.4f} | Task1 loss: {info2['total_loss']:.4f}")

    stats = engine.get_statistics()
    print("Buffer stats:", stats['buffer'])
    print("Training stats:", stats['training'])


if __name__ == '__main__':
    main()
