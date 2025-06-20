import torch

def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, recurrent_steps=5):
    model.train()
    total_loss = 0
    total_samples = 0
    correct = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        logits = model(x, steps=recurrent_steps)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples

    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Train Accuracy = {accuracy:.4f}")

