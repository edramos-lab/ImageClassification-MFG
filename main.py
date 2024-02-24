
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter, ToTensor, Normalize
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
from collections import Counter
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter, ToTensor, Normalize
from collections import Counter
from sklearn.model_selection import KFold
import wandb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef
import seaborn as sns


def preprocess_and_load_data(dataset_folder, image_size, batch_size, subset_ratio=0.1):
    """
    Preprocesses the dataset, loads it into DataLoader, and creates a balanced subset of the training dataset.

    Args:
    - dataset_folder: Path to the dataset folder.
    - image_size: Tuple of ints for the size of the images (height, width).
    - batch_size: Batch size for loading the data.
    - subset_ratio: Fraction of data to use in the subset for each class.

    Returns:
    - A dictionary containing 'train', 'val', and 'test' DataLoaders.
    - subset_dataset: A balanced subset of the training dataset.
    - balancing_efficiency: The efficiency of balancing the dataset.
    - num_classes: The number of classes in the dataset.
    """
    data_transforms = Compose([
        Resize(image_size),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(45),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dataset = ImageFolder(os.path.join(dataset_folder, 'train'), transform=data_transforms)
    val_dataset = ImageFolder(os.path.join(dataset_folder, 'valid'), transform=data_transforms)
    test_dataset = ImageFolder(os.path.join(dataset_folder, 'test'), transform=data_transforms)

    num_classes = len(train_dataset.classes)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Creating a balanced subset
    indices = []
    for class_index in range(num_classes):
        class_indices = np.where(np.array(train_dataset.targets) == class_index)[0]
        np.random.shuffle(class_indices)
        subset_size = int(len(class_indices) * subset_ratio)
        indices.extend(class_indices[:subset_size])

    subset_dataset = Subset(train_dataset, indices)
    subset_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

    # Calculate balancing efficiency
    class_counts = Counter([train_dataset.targets[i] for i in indices])
    max_samples = max(class_counts.values())
    balancing_efficiency = len(indices) / (num_classes * max_samples)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'subset': subset_loader
    }, subset_dataset, balancing_efficiency, num_classes


def train_model_kfold(subset_dataset, architecture, n_splits,epochs, num_classes, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    wandb.init(project="Research-ER1-zinc-plated-022024", config={"architecture": architecture, "epochs": epochs, "batch_size": batch_size})

    for fold, (train_idx, val_idx) in enumerate(kf.split(subset_dataset)):
        print(f"Training fold {fold+1} for {architecture}")
        
        # Model initialization
        model = timm.create_model(architecture, pretrained=True, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        criterion = nn.CrossEntropyLoss()

        # Subset training and validation loaders
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(subset_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(subset_dataset, batch_size=batch_size, sampler=val_sampler)

        for epoch in range(epochs):
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                train_correct += torch.sum(preds == labels).item()
                train_total += labels.size(0)
            
            # Validation accuracy
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    val_correct += torch.sum(preds == labels).item()
                    val_total += labels.size(0)
            
            # Logging metrics
            wandb.log({
                "fold": fold+1,
                "epoch": epoch+1,
                "train_loss": train_loss / len(train_loader),
                "train_accuracy": 100.0 * train_correct / train_total,
                "val_accuracy": 100.0 * val_correct / val_total,
            })
        scheduler.step()
    return model, optimizer, scheduler

def test_model(model, test_loader, architecture, optimizer, scheduler, batch_size, image_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_accuracy = 0  # Placeholder for accuracy calculation

    # Initialize lists to store true and predicted labels
    true_labels = []
    predicted_labels = []

    # Evaluate the model on the test split
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += preds.eq(labels).sum().item()
            total += len(labels)

            # For detailed metrics
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    # Convert lists to NumPy arrays for sklearn metrics
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Calculate metrics
    confusion = confusion_matrix(true_labels, predicted_labels)
    test_accuracy = 100 * accuracy_score(true_labels, predicted_labels)
    test_precision = 100 * precision_score(true_labels, predicted_labels, average='weighted')
    test_recall = 100 * recall_score(true_labels, predicted_labels, average='weighted')
    test_f1_score = 100 * f1_score(true_labels, predicted_labels, average='weighted')
    matthews_corr = 100 * matthews_corrcoef(true_labels, predicted_labels)

    # Log metrics

    # Calculate the confusion matrix
    confusion = confusion_matrix(true_labels, predicted_labels)

    # Plot and save the confusion matrix as a .png file
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("confusion_matrix.png")

    # Generate the confusion report
    confusion_report = classification_report(true_labels, predicted_labels)

    wandb.log({
        "Test Accuracy": test_accuracy,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Test F1 Score": test_f1_score,
        "Matthews Correlation Coefficient": matthews_corr,
        "Confusion Matrix": wandb.Image("confusion_matrix.png"),
        "Confusion Report": confusion_report
    })

    print("Test accuracy: %.3f" % test_accuracy)
    print("Confusion Matrix:\n", confusion)
    print("Test Precision: {:.6f}".format(test_precision))
    print("Test Recall: {:.6f}".format(test_recall))
    print("Test F1 Score: {:.6f}".format(test_f1_score))
    print("Matthews Correlation Coefficient: {:.6f}".format(matthews_corr))

    # Save the model to WandB
    model_path = "model_{}.pth".format(architecture)
    torch.save(model.state_dict(), model_path)
    artifact = wandb.Artifact(architecture, type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    # Save the model locally
    local_model_path = "local_model_{}.pth".format(architecture)
    torch.save(model.state_dict(), local_model_path)

    # Clean up CUDA memory
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.empty_cache()

    wandb.finish()


    

if __name__ == '__main__':
    dataset_folder = '/home/edramos/Documents/MLOPS/ImageClassification-MFG/nigel-chassises-1'
    image_size = (224, 224)  # Example image size
    batch_size = 8  # Example batch size
    n_splits=2
    epochs = 10
    data_loaders, subset_dataset, balancing_efficiency, num_classes = preprocess_and_load_data(dataset_folder, image_size, batch_size)

    # Example of how to use the data_loaders and subset_dataset
    print(f"Number of classes: {num_classes}")
    print(f"Balancing Efficiency: {balancing_efficiency}")
    for images, labels in data_loaders['train']:
        print(f'Train Batch size: {len(images)}')
        break  # Just to show the first batch, you can remove this break to iterate through the dataset

    for images, labels in data_loaders['subset']:
        print(f'Subset Batch size: {len(images)}')
        break  # Just to show the first batch from the subset

    #architectures = ["efficientnet_b0", "inception_v4", "swin_tiny_patch4_window7_224", "convnextv2_tiny", "xception41", "deit3_base_patch16_224"]
    architectures = ["convnextv2_tiny"]
    data_loaders, subset_dataset, balancing_efficiency, num_classes = preprocess_and_load_data(dataset_folder, image_size, batch_size)
    test_loader = data_loaders['test']
    for architecture in architectures:
        model, optimizer, scheduler =train_model_kfold(subset_dataset, architecture, n_splits,epochs, num_classes, batch_size)
    
    test_model(model, test_loader, architecture, optimizer, scheduler, batch_size, image_size)