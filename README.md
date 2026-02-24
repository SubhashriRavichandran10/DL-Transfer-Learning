# DL- Developing a Neural Network Classification Model using Transfer Learning

# NAME:R.SUBHASHRI
# REG NO:212223230219

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## DESIGN STEPS
### STEP 1: 

Import required libraries and define image transform

### STEP 2: 

Load training and testing datasets using ImageFolder


### STEP 3: 

Visualize sample images from the dataset.

### STEP 4: 


Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 
Define loss function (CrossEntropyLoss) and optimizer (Adam). Train the model and plot the loss curve.


### STEP 6: 

Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.



## PROGRAM

### Name:R.SUBHASHRI

### Register Number:212223230219

```python
# Load Pretrained Model and Modify for Transfer Learning

model=models.vgg19(weights=VGG19_Weights.DEFAULT)

# Modify the final fully connected layer to match the dataset classes

model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)

# Include the Loss function and optimizer

criterion =nn.BCEWithLogitsLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

# Train the model

def train_model(model, train_loader, test_loader, num_epochs=100):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # ----- Training -----
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # ----- Validation -----
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(test_loader)
        val_losses.append(epoch_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_train_loss:.4f}, '
              f'Validation Loss: {epoch_val_loss:.4f}')

    # ----- Plot -----
    print("Name:  R.SUBHASHRI")
    print("Register Number: 212223230219")

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    return train_losses, val_losses


```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="799" height="598" alt="image" src="https://github.com/user-attachments/assets/2059ec1b-c354-4425-90b6-d376b738f2bd" />


## Confusion Matrix

<img width="903" height="605" alt="image" src="https://github.com/user-attachments/assets/c146524d-faa1-4311-bbd4-10e6771d548d" />


## Classification Report
<img width="601" height="211" alt="image" src="https://github.com/user-attachments/assets/27146d09-83e5-4fb5-852c-d52da096a29d" />


### New Sample Data Prediction

<img width="449" height="404" alt="image" src="https://github.com/user-attachments/assets/a76b86d7-0950-4e4a-9e66-09236274d418" />


<img width="468" height="401" alt="image" src="https://github.com/user-attachments/assets/57ab3d02-9ca8-4030-aaba-5f9f0bd57443" />


## RESULT

Hence a Neural Network transfer model is developed using transfer learning
