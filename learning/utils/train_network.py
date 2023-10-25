import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train(model, dataloader, optimizer, criterion, logdir, device=torch.device('cpu'), epochs=10):
    writer = SummaryWriter(logdir)
    running_loss = 0.0
    for epoch in tqdm(range(epochs)):
        for i, data in tqdm(enumerate(dataloader), leave=False, total=len(dataloader)):
            optimizer.zero_grad()

            X, y = data

            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().cpu().item()
            if i % 500 == 499:
                writer.add_scalar('training loss', running_loss / 500, epoch * len(dataloader) + i)
                running_loss = 0.0