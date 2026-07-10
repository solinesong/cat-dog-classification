import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix

dataset_path = "cat and dog"
X = []
y = []

categories = ["cat", "dog"]
for label, category in enumerate(categories):
    folder = os.path.join(dataset_path, category)
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (128,128))
        X.append(image)
        y.append(label)
X = np.array(X)
y = np.array(y)

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42)
model = Sequential([  
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(128,128,3)
    ),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense( 1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.1
)

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

predictions = model.predict(X_test)

y_pred = (predictions > 0.5).astype(int)

cm = confusion_matrix(
    y_test,
    y_pred
)

model.save("cat_dog_model.h5")

# ============================================================
# TP5 - Cat vs Dog Classification using CNN
# ============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# ============================================================
# 1. LOAD DATASET
# ============================================================

dataset_path = "cat and dog"

X = []
y = []

categories = ["cat", "dog"]

for label, category in enumerate(categories):

    folder = os.path.join(dataset_path, category)

    for file in os.listdir(folder):

        img_path = os.path.join(folder, file)

        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.resize(image, (128, 128))

        X.append(image)
        y.append(label)

# ============================================================
# 2. CONVERT TO NUMPY ARRAY
# ============================================================

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("===================================")
print("Dataset Loaded Successfully")
print("===================================")

print("Images Shape :", X.shape)
print("Labels Shape :", y.shape)

# ============================================================
# 3. NORMALIZATION
# ============================================================

X = X / 255.0

# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Images :", len(X_train))
print("Testing Images  :", len(X_test))

# ============================================================
# 5. DATA AUGMENTATION
# ============================================================

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

datagen.fit(X_train)

# ============================================================
# 6. BUILD CNN MODEL
# ============================================================

model = Sequential([

    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(128,128,3)
    ),

    MaxPooling2D((2,2)),

    Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D((2,2)),

    Conv2D(
        128,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D((2,2)),

    Flatten(),

    Dense(
        128,
        activation='relu'
    ),

    Dropout(0.5),

    Dense(
        1,
        activation='sigmoid'
    )

])

# ============================================================
# 7. MODEL SUMMARY
# ============================================================

print("\n===================================")
print("MODEL SUMMARY")
print("===================================")

model.summary()

# ============================================================
# 8. COMPILE MODEL
# ============================================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# 9. SAVE BEST MODEL
# ============================================================

checkpoint = ModelCheckpoint(
    "best_cat_dog_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ============================================================
# 10. TRAIN MODEL
# ============================================================

history = model.fit(
    datagen.flow(
        X_train,
        y_train,
        batch_size=32
    ),
    epochs=25,
    validation_data=(
        X_test,
        y_test
    ),
    callbacks=[checkpoint]
)

# ============================================================
# 11. EVALUATE MODEL
# ============================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n===================================")
print("TEST RESULTS")
print("===================================")

print(f"Test Accuracy : {accuracy:.4f}")
print(f"Test Loss     : {loss:.4f}")

# ============================================================
# 12. ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# ============================================================
# 13. LOSS GRAPH
# ============================================================

plt.subplot(1,2,2)

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

predictions = model.predict(X_test)

y_pred = (predictions > 0.5).astype(int)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

print(cm)

# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Cat", "Dog"]
    )
)

# ============================================================
# 16. SAVE FINAL MODEL
# ============================================================

model.save("cat_dog_model.h5")

print("\nModel Saved Successfully")

# ============================================================
# 17. DEMO PREDICTIONS
# ============================================================

demo_folder = "demo"

print("\n===================================")
print("DEMO PREDICTIONS")
print("===================================")

for file in os.listdir(demo_folder):

    img_path = os.path.join(
        demo_folder,
        file
    )

    image = cv2.imread(img_path)

    if image is None:
        continue

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image_resized = cv2.resize(
        image,
        (128,128)
    )

    image_resized = image_resized / 255.0

    image_resized = np.expand_dims(
        image_resized,
        axis=0
    )

    prediction = model.predict(
        image_resized,
        verbose=0
    )

    dog_probability = prediction[0][0]
    cat_probability = 1 - dog_probability

    if dog_probability > 0.5:
        label = "DOG"
        confidence = dog_probability
    else:
        label = "CAT"
        confidence = cat_probability

    plt.figure(figsize=(5,5))

    plt.imshow(image_rgb)

    plt.title(
        f"Prediction: {label}\n"
        f"Confidence: {confidence:.2%}"
    )

    plt.axis("off")
    plt.show()

    # ============================================================
# TP5 - Cat vs Dog Classification
# Transfer Learning with MobileNetV2
# ============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# ============================================================
# 1. LOAD DATASET
# ============================================================

dataset_path = "cat and dog"

X = []
y = []

categories = ["cat", "dog"]

for label, category in enumerate(categories):

    folder = os.path.join(dataset_path, category)

    for file in os.listdir(folder):

        img_path = os.path.join(folder, file)

        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (224,224))

        X.append(image)
        y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("Dataset Shape:", X.shape)
print("Labels Shape :", y.shape)

# ============================================================
# 2. NORMALIZE
# ============================================================

X = X / 255.0

# ============================================================
# 3. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Images:", len(X_train))
print("Testing Images :", len(X_test))

# ============================================================
# 4. DATA AUGMENTATION
# ============================================================

datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.30,
    width_shift_range=0.20,
    height_shift_range=0.20,
    horizontal_flip=True,
    brightness_range=[0.8,1.2]
)

datagen.fit(X_train)

# ============================================================
# 5. PRETRAINED MODEL
# ============================================================

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

# ============================================================
# 6. BUILD MODEL
# ============================================================

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(
        128,
        activation='relu'
    ),

    Dropout(0.5),

    Dense(
        1,
        activation='sigmoid'
    )

])

model.summary()

# ============================================================
# 7. COMPILE
# ============================================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# 8. SAVE BEST MODEL
# ============================================================

checkpoint = ModelCheckpoint(
    "best_cat_dog_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ============================================================
# 9. TRAIN
# ============================================================

history = model.fit(
    datagen.flow(
        X_train,
        y_train,
        batch_size=16
    ),
    epochs=20,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]
)

# ============================================================
# 10. EVALUATE
# ============================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n===========================")
print("TEST RESULTS")
print("===========================")

print("Accuracy:", accuracy)
print("Loss:", loss)

# ============================================================
# 11. ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend([
    "Train",
    "Validation"
])

# ============================================================
# 12. LOSS GRAPH
# ============================================================

plt.subplot(1,2,2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend([
    "Train",
    "Validation"
])

plt.tight_layout()
plt.show()

# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

predictions = model.predict(X_test)

y_pred = (
    predictions > 0.5
).astype(int)

print("\nConfusion Matrix")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nClassification Report")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Cat",
            "Dog"
        ]
    )
)

# ============================================================
# 14. SAVE FINAL MODEL
# ============================================================

model.save(
    "cat_dog_model.h5"
)

print("\nModel Saved!")

# ============================================================
# 15. DEMO PREDICTIONS
# ============================================================

demo_folder = "demo"

for file in os.listdir(demo_folder):

    img_path = os.path.join(
        demo_folder,
        file
    )

    image = cv2.imread(img_path)

    if image is None:
        continue

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image_resized = cv2.resize(
        image_rgb,
        (224,224)
    )

    image_resized = image_resized / 255.0

    image_resized = np.expand_dims(
        image_resized,
        axis=0
    )

    prediction = model.predict(
        image_resized,
        verbose=0
    )

    dog_prob = prediction[0][0]
    cat_prob = 1 - dog_prob

    if dog_prob > 0.5:
        label = "DOG"
        confidence = dog_prob
    else:
        label = "CAT"
        confidence = cat_prob

    plt.figure(figsize=(5,5))

    plt.imshow(image_rgb)

    plt.title(
        f"{label}\n"
        f"Confidence: {confidence:.2%}"
    )

    plt.axis("off")
    plt.show()

# ============================================================
# Cat vs Dog Classification
# Transfer Learning Version (MobileNetV2)
# ============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# ============================================================
# LOAD DATASET
# ============================================================

dataset_path = "cat and dog"
X = []
y = []
categories = ["cat", "dog"]

for label, category in enumerate(categories):
    folder = os.path.join(dataset_path, category)
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        X.append(image)
        y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("Images:", X.shape)
print("Labels:", y.shape)

# ============================================================
# CLASS BALANCE CHECK
# ============================================================

unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {categories[u]}: {c} images")

# ============================================================
# NORMALIZATION — MobileNetV2 expects [-1, 1] not [0, 1]
# ============================================================

X = (X / 127.5) - 1.0        # matches MobileNetV2 preprocess_input

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training:", len(X_train))
print("Testing :", len(X_test))

# ============================================================
# DATA AUGMENTATION
# ============================================================

datagen = ImageDataGenerator(
    rotation_range=25,
    zoom_range=0.25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

# ============================================================
# TRANSFER LEARNING — MobileNetV2 pretrained on ImageNet
# ============================================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,           # remove ImageNet classifier head
    weights='imagenet'           # use pretrained weights
)

base_model.trainable = False     # freeze all pretrained layers

# ============================================================
# ADD CLASSIFICATION HEAD
# ============================================================

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# CALLBACKS — both monitor val_accuracy
# ============================================================

checkpoint = ModelCheckpoint(
    "best_cat_dog_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=7,
    restore_best_weights=True
)

# ============================================================
# PHASE 1 — Train only the head (base frozen)
# ============================================================

print("\n--- Phase 1: Training classification head ---")

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=15,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stop]
)

# ============================================================
# PHASE 2 — Fine-tune: unfreeze top layers of base model
# ============================================================

print("\n--- Phase 2: Fine-tuning top layers ---")

base_model.trainable = True

# Only unfreeze the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=20,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stop]
)

# ============================================================
# EVALUATE
# ============================================================

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\nAccuracy:", accuracy)
print("Loss    :", loss)

# ============================================================
# CONFUSION MATRIX
# ============================================================

predictions = model.predict(X_test)
y_pred = (predictions > 0.5).astype(int)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

# ============================================================
# PLOT ACCURACY / LOSS
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'])
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'])
plt.axvline(x=len(history.history['accuracy']), color='gray', linestyle='--', label='Fine-tune start')
plt.title("Accuracy")
plt.legend(["Train", "Validation", "Fine-tune start"])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + history_fine.history['loss'])
plt.plot(history.history['val_loss'] + history_fine.history['val_loss'])
plt.axvline(x=len(history.history['loss']), color='gray', linestyle='--')
plt.title("Loss")
plt.legend(["Train", "Validation", "Fine-tune start"])

plt.tight_layout()
plt.show()

# ============================================================
# SAVE MODEL
# ============================================================

model.save("cat_dog_model.h5")

# ============================================================
# DEMO PREDICTIONS
# ============================================================

demo_folder = "demo"

for file in os.listdir(demo_folder):

    img_path = os.path.join(demo_folder, file)
    image = cv2.imread(img_path)

    if image is None:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224))

    # Must match training normalization: [-1, 1]
    image_input = image_resized.astype(np.float32)
    image_input = (image_input / 127.5) - 1.0
    image_input = np.expand_dims(image_input, axis=0)

    prediction = model.predict(image_input, verbose=0)[0][0]

    if prediction > 0.5:
        label = "DOG"
        confidence = prediction
    else:
        label = "CAT"
        confidence = 1 - prediction

    plt.figure(figsize=(5, 5))
    plt.imshow(image_rgb)
    plt.title(f"{label}\nConfidence: {confidence:.2%}")
    plt.axis("off")
    plt.show()

    # ============================================================
# TP5 - Cat vs Dog — Full Evaluation
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    matthews_corrcoef
)

# ============================================================
# PREDICT
# ============================================================

y_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_prob > 0.5).astype(int)

# ============================================================
# CORE METRICS
# ============================================================

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
mcc       = matthews_corrcoef(y_test, y_pred)

print("=" * 40)
print("         EVALUATION METRICS")
print("=" * 40)
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  MCC       : {mcc:.4f}")
print("=" * 40)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Cat", "Dog"]
))

# ============================================================
# CONFIDENCE STATISTICS (Standard Deviation)
# ============================================================

cat_probs = y_prob[y_test == 0]
dog_probs = y_prob[y_test == 1]

print("=" * 40)
print("     CONFIDENCE STATISTICS")
print("=" * 40)
print(f"  CAT predictions (raw prob):")
print(f"    Mean : {cat_probs.mean():.4f}")
print(f"    Std  : {cat_probs.std():.4f}")
print(f"    Min  : {cat_probs.min():.4f}")
print(f"    Max  : {cat_probs.max():.4f}")
print()
print(f"  DOG predictions (raw prob):")
print(f"    Mean : {dog_probs.mean():.4f}")
print(f"    Std  : {dog_probs.std():.4f}")
print(f"    Min  : {dog_probs.min():.4f}")
print(f"    Max  : {dog_probs.max():.4f}")
print("=" * 40)

# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Cat vs Dog — Full Evaluation", fontsize=15, fontweight='bold')

# ── Plot 1: Confusion Matrix ─────────────────────────────────

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["Cat", "Dog"],
    yticklabels=["Cat", "Dog"],
    ax=axes[0, 0]
)
axes[0, 0].set_title("Confusion Matrix")
axes[0, 0].set_xlabel("Predicted")
axes[0, 0].set_ylabel("Actual")

# ── Plot 2: Metrics Bar Chart ────────────────────────────────

metrics       = ["Accuracy", "Precision", "Recall", "F1 Score"]
metric_values = [accuracy, precision, recall, f1]
colors        = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

bars = axes[0, 1].bar(metrics, metric_values, color=colors, width=0.5)
axes[0, 1].set_ylim(0, 1.15)
axes[0, 1].set_title("Core Metrics")
axes[0, 1].set_ylabel("Score")

for bar, val in zip(bars, metric_values):
    axes[0, 1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.3f}",
        ha='center', fontsize=11, fontweight='bold'
    )

# ── Plot 3: ROC Curve ────────────────────────────────────────

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc      = auc(fpr, tpr)

axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2,
                label=f"AUC = {roc_auc:.4f}")
axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--',
                label="Random classifier")
axes[1, 0].set_xlim([0.0, 1.0])
axes[1, 0].set_ylim([0.0, 1.05])
axes[1, 0].set_title("ROC Curve")
axes[1, 0].set_xlabel("False Positive Rate")
axes[1, 0].set_ylabel("True Positive Rate")
axes[1, 0].legend(loc="lower right")

# ── Plot 4: Confidence Distribution ─────────────────────────

axes[1, 1].hist(cat_probs, bins=20, alpha=0.6, color='steelblue', label='Cat (true)')
axes[1, 1].hist(dog_probs, bins=20, alpha=0.6, color='tomato',    label='Dog (true)')
axes[1, 1].axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Threshold = 0.5')
axes[1, 1].set_title("Confidence Distribution")
axes[1, 1].set_xlabel("Predicted Probability (toward Dog)")
axes[1, 1].set_ylabel("Count")
axes[1, 1].legend()

plt.tight_layout()
plt.show()