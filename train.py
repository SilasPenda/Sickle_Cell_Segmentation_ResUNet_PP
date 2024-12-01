import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.dataset import get_data_loaders
from src.model import ResUNet_pp
from tensorflow.keras.models import load_model


def main():
    parser = argparse.ArgumentParser(description='Train a 2D U-Net Segmentation model.')
    parser.add_argument('-e', '--epochs', type=int, help='Number of Epochs', required=True)
    parser.add_argument('-p', '--checkpoint', type=str, default=None, help='Model checkpoint', required=False)
    args = parser.parse_args()

    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    model_results_dir = os.path.join(results_dir, f"train_{len(os.listdir(results_dir)) + 1}")
    os.makedirs(model_results_dir, exist_ok=True)

    last_model_path = os.path.join(model_results_dir, "last.h5")
    best_model_path = os.path.join(model_results_dir, "best.h5")
    loss_plot_path = os.path.join(model_results_dir, "loss.png")

    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512

    train_loader, val_loader = get_data_loaders(IMAGE_HEIGHT, IMAGE_WIDTH)
    train_steps = len(train_loader)
    val_steps = len(val_loader)

    num_epochs = args.epochs

    # Initialize the model
    if args.checkpoint is not None:
        model = load_model(args.checkpoint) 
    else:
        model = ResUNet_pp(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), num_classes=4)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # Use sparse categorical crossentropy for multi-class segmentation
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    # Metrics
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    val_loss = tf.keras.metrics.Mean(name="val_loss")

    best_loss = float("inf")
    train_losses, val_losses = [], []

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        train_loss.reset_state()
        for images, masks in tqdm(train_loader, desc="Training"):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                # print("unique masks: ", np.unique(masks))
                loss = loss_fn(masks, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss.update_state(loss)

        train_losses.append(train_loss.result().numpy())

        # Validation phase
        val_loss.reset_state()
        for images, masks in tqdm(val_loader, desc="Validation"):
            predictions = model(images, training=False)
            loss = loss_fn(masks, predictions)
            val_loss.update_state(loss)
            
        val_losses.append(val_loss.result().numpy())

        # Save the latest model
        model.save(last_model_path)

        print(f"Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")

        # Plot train and validation losses
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Train and Validation Loss Over Epochs")
        plt.savefig(loss_plot_path)
        plt.close()


        # Save the best model
        if val_losses[-1] < best_loss:
            model.save(best_model_path)


if __name__ == "__main__":
    main()