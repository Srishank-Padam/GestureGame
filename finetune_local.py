"""
finetune_local.py  (Phase 7: Transfer Learning / Fine-Tuning)

Run this LOCALLY after collect_webcam_data.py has gathered your own images.
Fine-tunes the Colab-trained CNN (gesture_cnn_final.keras) on your real webcam data,
so it generalizes to your actual demo conditions instead of only the IR lab dataset.

Usage:
    python finetune_local.py

Expects, in the same folder:
    gesture_cnn_final.keras         (downloaded from the Colab notebook)
    label_map.json                  (downloaded from the Colab notebook)
    webcam_data/<class>/<condition>/*.png   (produced by collect_webcam_data.py,
                                              condition in: near, far, rotated, different_light)

Outputs:
    gesture_cnn_finetuned.keras

Design notes (why it's built this way):
  - Split is by CONDITION, not by random shuffling of individual frames. Consecutive
    frames from the same recording session look nearly identical, so a random split
    would leak near-duplicates between train/val/test and make validation accuracy
    look artificially high. Holding out whole conditions avoids that.
  - Augmentation is applied during training since ~80 images/class is small and prone
    to overfitting without it.
  - Fine-tuning is staged: first the classifier head is trained with the convolutional
    base frozen (adapts the decision boundary without disturbing learned features),
    then the last conv block is unfrozen and fine-tuned at a very low learning rate
    (lets the deepest features adapt slightly to webcam-domain images).
"""

import os
import json
import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import optimizers, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 64
MODEL_PATH = "gesture_cnn_final.h5"
LABEL_MAP_PATH = "label_map.json"
DATA_DIR = "webcam_data"
OUTPUT_PATH = "gesture_cnn_finetuned.keras"

CONDITIONS = ["near", "far", "rotated", "different_light"]
TRAIN_CONDITIONS = ["near", "far"]        # 2 conditions -> train pool
VAL_CONDITIONS = ["rotated"]              # 1 condition, held out entirely -> val
TEST_CONDITIONS = ["different_light"]     # 1 condition, held out entirely -> test

STAGE1_EPOCHS = 8   # classifier-only, conv base frozen
STAGE2_EPOCHS = 10  # last conv block unfrozen, very low LR
STAGE1_LR = 1e-3
STAGE2_LR = 1e-5


def load_split(label_to_idx, conditions):
    images, labels = [], []
    for gesture_name, idx in label_to_idx.items():
        for cond in conditions:
            folder = os.path.join(DATA_DIR, gesture_name, cond)
            if not os.path.isdir(folder):
                continue
            for fpath in glob.glob(os.path.join(folder, "*.png")):
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(idx)
    images = np.array(images, dtype=np.float32)[..., np.newaxis] / 255.0
    labels = np.array(labels)
    return images, labels


def freeze_conv_layers(model, freeze=True):
    for layer in model.layers:
        if isinstance(layer, (layers.Conv2D, layers.BatchNormalization)):
            layer.trainable = not freeze
        elif isinstance(layer, layers.Dense):
            layer.trainable = True


def unfreeze_last_conv_block(model):
    conv_layers = [l for l in model.layers if isinstance(l, layers.Conv2D)]
    if conv_layers:
        conv_layers[-1].trainable = True


def main():
    with open(LABEL_MAP_PATH) as f:
        idx_to_label = {int(k): v for k, v in json.load(f).items()}
    label_to_idx = {v: k for k, v in idx_to_label.items()}

    print("Loading custom webcam data (split by condition, not randomly)...")
    X_train, y_train = load_split(label_to_idx, TRAIN_CONDITIONS)
    X_val, y_val = load_split(label_to_idx, VAL_CONDITIONS)
    X_test, y_test = load_split(label_to_idx, TEST_CONDITIONS)

    print(f"Train: {len(X_train)} images (conditions: {TRAIN_CONDITIONS})")
    print(f"Val:   {len(X_val)} images (conditions: {VAL_CONDITIONS})")
    print(f"Test:  {len(X_test)} images (conditions: {TEST_CONDITIONS})")

    if len(X_train) < 20:
        print("Very little data found. Run collect_webcam_data.py first.")
        return

    print("\nLoading base model:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    augmenter = ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1,
        height_shift_range=0.1, zoom_range=0.1
    )

    # --- Stage 1: freeze conv base, train classifier head only ---
    print("\n=== Stage 1: training classifier head (conv layers frozen) ===")
    freeze_conv_layers(model, freeze=True)
    model.compile(optimizer=optimizers.Adam(learning_rate=STAGE1_LR),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    train_gen = augmenter.flow(X_train, y_train, batch_size=16)
    model.fit(train_gen, validation_data=(X_val, y_val), epochs=STAGE1_EPOCHS, verbose=1)

    # --- Stage 2: unfreeze last conv block, fine-tune at low LR ---
    print("\n=== Stage 2: fine-tuning last conv block (very low LR) ===")
    unfreeze_last_conv_block(model)
    model.compile(optimizer=optimizers.Adam(learning_rate=STAGE2_LR),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )
    train_gen = augmenter.flow(X_train, y_train, batch_size=16)
    model.fit(train_gen, validation_data=(X_val, y_val), epochs=STAGE2_EPOCHS,
              callbacks=[early_stop], verbose=1)

    # --- Final evaluation on the held-out test condition ---
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    if len(X_test) > 0:
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    else:
        test_acc = None

    print("\n=== Results ===")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")
    if test_acc is not None:
        print(f"Test accuracy:  {test_acc:.4f}  (never trained on this condition)")
    else:
        print("Test accuracy:  N/A (no test condition data found)")

    model.save(OUTPUT_PATH)
    print(f"\nSaved fine-tuned model to {OUTPUT_PATH}")
    print("Point game_control.py at this file to use the fine-tuned version.")


if __name__ == "__main__":
    main()
