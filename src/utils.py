import yaml
import tensorflow as tf
from tensorflow.keras import backend as K


def get_config(config_filepath: str) -> dict:
    try:
        with open(config_filepath) as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        return {}
    
# Loss Function and coefficients to be used during training:
# def dice_coefficient(y_true, y_pred):
#     smoothing_factor = 1
#     flat_y_true = K.flatten(y_true)
#     flat_y_pred = K.flatten(y_pred)
#     return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

# def dice_coefficient_loss(y_true, y_pred):
#     return 1 - dice_coefficient(y_true, y_pred)

# def dice_coefficient(y_true, y_pred):
#     smoothing_factor = 1
#     num_classes = K.int_shape(y_pred)[-1]

#     y_true = tf.one_hot(y_true, depth=num_classes)
#     flat_y_true = K.flatten(K.cast(y_true, dtype='float32'))
#     flat_y_pred = K.flatten(K.cast(y_pred, dtype='float32'))    

#     return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    num_classes = K.int_shape(y_pred)[-1]

    # Cast y_true to int32 before using tf.one_hot
    y_true = tf.cast(y_true, dtype='int32')
    y_true = tf.one_hot(y_true, depth=num_classes)

    flat_y_true = K.flatten(K.cast(y_true, dtype='float32'))
    flat_y_pred = K.flatten(K.cast(y_pred, dtype='float32'))

    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)


def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

    
# def dice_loss(pred, target, smooth=1e-6):
#     # pred: [batch, channels, height, width], pred should be softmax probabilities
#     # target: [batch, height, width], target should be indices of classes (0 to C-1)
    
#     pred = torch.softmax(pred, dim=1)  # Apply softmax to obtain probabilities
#     C = pred.shape[1]  # Number of classes

#     dice = 0
#     for c in range(C):
#         pred_c = pred[:, c, :, :]
#         target_c = (target == c).float()  # Create a mask for class c

#         intersection = (pred_c * target_c).sum((1, 2))  # Sum over each image separately
#         # print(pred_c.shape, pred_c.sum(), target_c.shape, target_c.sum())

#         union = pred_c.sum((1, 2)) + target_c.sum((1, 2))

#         dice_c = (2. * intersection + smooth) / (union + smooth)
#         dice += dice_c.mean()  # Average over all images in the batch

#     return 1 - dice / C  # Average over all classes
    
# def dice_score(pred, target, smooth=1e-6):
#     # Convert prediction to binary using a threshold (0.5 for binary segmentation)
#     pred = (pred > 0.5).float()
    
#     # Flatten the tensors
#     preds = pred.view(-1)
#     targets = target.view(-1)

#     # Calculate intersection and union
#     intersection = (preds * targets).sum()
#     union = preds.sum() + targets.sum()

#     # Calculate Dice coefficient
#     dice = (2. * intersection + smooth) / (union + smooth)

#     return dice

# def dice_score(pred, target, num_classes):
#     """Calculate the Dice score for multiclass segmentation."""
#     smooth = 1e-6
#     dice = 0.0

#     # Apply softmax to get class probabilities and then calculate Dice per class
#     for i in range(num_classes):
#         # Binary mask for class i
#         pred_i = pred[:, i, :, :]  # Predicted probabilities for class i
#         target_i = (target == i).float()  # True mask for class i

#         # Calculate Dice for class i
#         intersection = (pred_i * target_i).sum()
#         union = pred_i.sum() + target_i.sum()
#         dice += (2.0 * intersection + smooth) / (union + smooth)

#     # Average Dice score across all classes
#     return dice / num_classes

