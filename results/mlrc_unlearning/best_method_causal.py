from copy import deepcopy
import torch
from torch import nn, optim
from methods.BaseMethod import BaseMethod

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    def run(self, net, retain_loader, forget_loader, val_loader):
        net.to(DEVICE)
        net.train()

        # Save original model for distillation
        orig_net = deepcopy(net).to(DEVICE)
        orig_net.eval()

        # Step 1: Gradient ascent on forget set with uniform-target KL divergence to induce forgetting
        def uniform_target_kl(logits):
            # logits: [batch_size, num_classes]
            num_classes = logits.size(1)
            log_prob = nn.functional.log_softmax(logits, dim=1)
            # uniform distribution target
            uniform_prob = torch.full_like(log_prob, 1.0 / num_classes)
            # KL divergence KL[model||uniform], but for ascent we maximize this loss
            kl = nn.functional.kl_div(log_prob, uniform_prob, reduction='batchmean')
            return kl

        optimizer_forget = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)

        forget_epochs = 1  # keep small for efficiency
        for _ in range(forget_epochs):
            for sample in forget_loader:
                if isinstance(sample, dict):
                    inputs = sample['image'] if 'image' in sample else next(iter(sample.values()))
                else:
                    inputs = sample[0]
                inputs = inputs.to(DEVICE)

                optimizer_forget.zero_grad()
                outputs = net(inputs)
                loss_forget = uniform_target_kl(outputs)
                # Gradient ascent step: maximize loss_forget
                (-loss_forget).backward()
                optimizer_forget.step()

        # Step 2: Inject small Gaussian noise to conv weights before fine-tuning
        with torch.no_grad():
            for name, param in net.named_parameters():
                if 'conv' in name and param.requires_grad:
                    noise = torch.randn_like(param) * 0.02
                    param.add_(noise)

        # Step 3: Fine-tune on retain set with cross-entropy + symmetric KL distillation
        epochs = 1
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        def symmetric_kl(p_logits, q_logits):
            p_log_prob = nn.functional.log_softmax(p_logits, dim=1)
            q_log_prob = nn.functional.log_softmax(q_logits, dim=1)
            p_prob = nn.functional.softmax(p_logits, dim=1)
            q_prob = nn.functional.softmax(q_logits, dim=1)
            kl_pq = nn.functional.kl_div(p_log_prob, q_prob, reduction='batchmean')
            kl_qp = nn.functional.kl_div(q_log_prob, p_prob, reduction='batchmean')
            return 0.5 * (kl_pq + kl_qp)

        for ep in range(epochs):
            net.train()
            for sample in retain_loader:
                if isinstance(sample, dict):
                    inputs = sample['image'] if 'image' in sample else next(iter(sample.values()))
                    targets = sample['age_group'] if 'age_group' in sample else next(iter(sample.values()))
                else:
                    inputs, targets = sample
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = net(inputs)
                with torch.no_grad():
                    orig_outputs = orig_net(inputs)

                loss_ce = criterion(outputs, targets)
                loss_kl = symmetric_kl(outputs, orig_outputs)

                loss = loss_ce + 0.1 * loss_kl
                loss.backward()
                optimizer.step()
            scheduler.step()

        net.eval()
