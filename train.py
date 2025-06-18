import torch

def train_epoch(pcn, classifier, train_loader, optimizer, loss_fn, device, epoch):
    pcn.train()
    classifier.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pcn.reset_states()
        for _ in range(5):
            _, _, _, h2 = pcn(x)

        logits = classifier(h2)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch}: Train Loss = {total_loss / len(train_loader):.4f}")
