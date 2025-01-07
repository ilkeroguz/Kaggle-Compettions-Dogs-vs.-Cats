import torch
import torch.nn as nn
import torch.optim as optim
from utils import CNNCatsDogs, get_data_loaders, save_model, get_transform, evaluate_model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
train_dir = 'data/train'
val_dir = 'data/val'
batch_size = 32
learning_rate = 0.01
num_epochs = 10

# Data loaders
transform = get_transform()
train_loader, val_loader = get_data_loaders(train_dir, val_dir, batch_size)

# Model, loss, optimizer, and scheduler
model = CNNCatsDogs().to(device)
criterion = nn.CrossEntropyLoss()

#optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')

# Evaluation
accuracy = evaluate_model(model, val_loader, device)
print(f'Validation Accuracy: {accuracy:.2f}%')

# Save the model
save_model(model, 'cat_dog_classifier.pth')
