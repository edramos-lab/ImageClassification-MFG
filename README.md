

main.py

Description:
    This script is designed to perform "image classification on the any unbalanced custom dataset using pre-trained ConvNeXt models ".

Dependencies:
    - Python 3.x
    - TensorFlow 2.x
    - NumPy
    - Matplotlib
    - Roboflow
    - torchvision
    - timm
    - scikit-learn
    - wandb
    - seaborn

Usage:
    Run this script from the command line using the following command :
    ```
    python main.py 
    ```

    Make sure all dependencies are installed. You can install them using:
    ```
    pip install -r requirements.txt
    ```

Functions:
    preprocess_and_load_data(dataset_folder, image_size, batch_size, subset_ratio=0.1):
        - Description: Preprocesses the dataset, loads it into DataLoader, and creates a balanced subset of the training dataset.
        - Parameters:
            - dataset_folder: Path to the dataset folder.
            - image_size: Tuple of ints for the size of the images (height, width).
            - batch_size: Batch size for loading the data.
            - subset_ratio: Fraction of data to use in the subset for each class.

    train_model_kfold(subset_dataset, architecture, n_splits, epochs, num_classes, batch_size):
        - Description: Trains a model using K-fold cross-validation.
        - Parameters:
            - subset_dataset: The dataset to use for training.
            - architecture: The architecture of the model to train.
            - n_splits: The number of splits for K-fold cross-validation.
            - epochs: The number of epochs to train for.
            - num_classes: The number of classes in the dataset.
            - batch_size: The batch size for training.

    test_model(model, test_loader, architecture, optimizer, scheduler, batch_size, image_size):
        - Description: Tests the trained model on a separate test dataset and logs metrics.
        - Parameters:
            - model: The trained model to test.
            - test_loader: DataLoader for the test dataset.
            - architecture: The architecture of the model.
            - optimizer: The optimizer used during training.
            - scheduler: The learning rate scheduler used during training.
            - batch_size: The batch size used during testing.
            - image_size: The image size used during testing.

Author:
    Edgar Rene Ramos Acosta

Date:
    [23-Feb-2024 22:01]
"""
main.py

Description:
    This script is designed to perform [Brief description of what the script does, e.g., "image classification on the Nigel Chassises dataset using pre-trained ConvNeXt models"].

Dependencies:
    - Python 3.x
    - TensorFlow 2.x
    - NumPy
    - Matplotlib
    - Roboflow
    - torchvision
    - timm
    - scikit-learn
    - wandb
    - seaborn

Usage:
    Run this script from the command line using the following command:
    ```
    python main.py
    ```

    Make sure all dependencies are installed. You can install them using:
    ```
    pip install -r requirements.txt
    ```

Functions:
    preprocess_and_load_data(dataset_folder, image_size, batch_size, subset_ratio=0.1):
        - Description: Preprocesses the dataset, loads it into DataLoader, and creates a balanced subset of the training dataset.
        - Parameters:
            - dataset_folder: Path to the dataset folder.
            - image_size: Tuple of ints for the size of the images (height, width).
            - batch_size: Batch size for loading the data.
            - subset_ratio: Fraction of data to use in the subset for each class.

    train_model_kfold(subset_dataset, architecture, n_splits, epochs, num_classes, batch_size):
        - Description: Trains a model using K-fold cross-validation.
        - Parameters:
            - subset_dataset: The dataset to use for training.
            - architecture: The architecture of the model to train.
            - n_splits: The number of splits for K-fold cross-validation.
            - epochs: The number of epochs to train for.
            - num_classes: The number of classes in the dataset.
            - batch_size: The batch size for training.

    test_model(model, test_loader, architecture, optimizer, scheduler, batch_size, image_size):
        - Description: Tests the trained model on a separate test dataset and logs metrics.
        - Parameters:
            - model: The trained model to test.
            - test_loader: DataLoader for the test dataset.
            - architecture: The architecture of the model.
            - optimizer: The optimizer used during training.
            - scheduler: The learning rate scheduler used during training.
            - batch_size: The batch size used during testing.
            - image_size: The image size used during testing.

Edgar Rene Ramos Acosta

Date:
    [24-Feb-2024 12:01]
"""
