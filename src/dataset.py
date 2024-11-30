import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from src.utils import get_config


from PIL import Image
import numpy as np
import os
import tensorflow as tf

class LoadTransformDataset(tf.keras.utils.Sequence):
    def __init__(self, images_dir, labels_dir, transform=None, batch_size=8, image_size=(256, 256)):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.images = os.listdir(images_dir)
        self.labels = os.listdir(labels_dir)
        self.transform = transform
        self.batch_size = batch_size
        self.image_size = image_size

        self.color_to_class = {
            (0, 0, 0): 0,       # Background
            (255, 0, 0): 1,     # Elongated
            (0, 255, 0): 2,     # Circular
            (0, 0, 255): 3      # Other
        }

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def rgb_to_class(self, mask):
        """Convert an RGB mask to class indices."""
        class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        for rgb, cls in self.color_to_class.items():
            class_mask[np.all(mask == rgb, axis=-1)] = cls
        return class_mask

    def load_image(self, image_path, mask_path):
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Ensure PIL.Image for resizing
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.uint8(image * 255))
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.uint8(mask))

        # Resize and normalize
        image = np.array(image.resize(self.image_size)) / 255.0
        mask = self.rgb_to_class(np.array(mask.resize(self.image_size)))

        return image, mask

    def __getitem__(self, idx):
        """Fetch a batch of data."""
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        batch_images = self.images[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]

        # Stop processing if the batch is empty
        if not batch_images or not batch_labels:
            raise IndexError("Batch index out of range. No more batches to process.")

        # print(f"Batch {idx}: {batch_images}")

        images, masks = [], []
        for img_name, label_name in zip(batch_images, batch_labels):
            img_path = os.path.join(self.images_dir, img_name)
            label_path = os.path.join(self.labels_dir, label_name)
            image, mask = self.load_image(img_path, label_path)
            images.append(image)
            masks.append(mask)

        return np.array(images, dtype=np.float32), np.array(masks, dtype=np.int32)


def get_data_loaders(IMAGE_HEIGHT, IMAGE_WIDTH):
    # Define data augmentation
    def augment(image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.rot90(image, tf.random.uniform(shape=[], maxval=4, dtype=tf.int32))
        return image

    train_transform = augment
    val_transform = None

    # Load configuration
    config = get_config(config_filepath=os.path.join(os.getcwd(), 'config.yaml'))
    train_path = config.get('train_path', None)
    val_path = config.get('val_path', None)

    train_images_dir = os.path.join(train_path, "images")
    train_labels_dir = os.path.join(train_path, "labels")
    val_images_dir = os.path.join(val_path, "images")
    val_labels_dir = os.path.join(val_path, "labels")

    batch_size = 8
    image_size = (IMAGE_HEIGHT, IMAGE_WIDTH)

    # Create datasets
    train_dataset = LoadTransformDataset(train_images_dir, train_labels_dir, transform=train_transform, batch_size=batch_size, image_size=image_size)
    val_dataset = LoadTransformDataset(val_images_dir, val_labels_dir, transform=val_transform, batch_size=batch_size, image_size=image_size)

    return train_dataset, val_dataset


if __name__ == "__main__":
    IMAGE_HEIGHT, IMAGE_WIDTH = 512, 512
    train_loader, val_loader = get_data_loaders(IMAGE_HEIGHT, IMAGE_WIDTH)

    for images, labels in train_loader:
        # print("Batch shape:", images.shape)

        # Visualize the first image and label in the batch
        image = images[0]  # HWC format
        label = labels[0]
        print("Image shape:", image.shape)
        print("Label shape:", label.shape)
        print("Label values:", np.unique(label))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Image")

        plt.subplot(1, 2, 2)
        plt.imshow(label, cmap="gray")
        plt.title("Label")

        plt.savefig("save.png")
        break
