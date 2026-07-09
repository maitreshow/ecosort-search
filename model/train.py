"""
Jalon 1 - Script d'entrainement du modele EcoSort.
Auteur: ARZIKA

Approche : Transfer Learning avec MobileNetV2 (Keras/TensorFlow)
Dataset  : Garbage Classification (Kaggle) - 6 classes
           cardboard, glass, metal, paper, plastic, trash

Usage :
    python model/train.py
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
DATA_DIR = "data/Garbage classification/Garbage classification"
IMG_SIZE = (224, 224)          # taille standard attendue par MobileNetV2
BATCH_SIZE = 32
EPOCHS_HEAD = 10                # entrainement de la "tete" (couches ajoutees)
EPOCHS_FINETUNE = 5             # fine-tuning des dernieres couches de MobileNetV2
SEED = 42
MODEL_OUTPUT_PATH = "model/modele_eco_sort.h5"

# ---------------------------------------------------------------------------
# 2. CHARGEMENT DES DONNEES
# ---------------------------------------------------------------------------
print("Chargement des donnees...")

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes detectees ({num_classes}) : {class_names}")

# On garde une reference AVANT d'appliquer .map() (qui casse .file_paths)
# pour pouvoir calculer les poids de classes ensuite.
train_labels = []
for _, labels in train_ds.unbatch():
    train_labels.append(int(labels.numpy()))

# ---------------------------------------------------------------------------
# 3. POIDS DES CLASSES (pour compenser le desequilibre, ex: "trash" sous-represente)
# ---------------------------------------------------------------------------
from collections import Counter

counts = Counter(train_labels)
total = sum(counts.values())
class_weight = {
    i: total / (num_classes * counts[i]) for i in range(num_classes)
}
print("Poids par classe (pour compenser le desequilibre) :", class_weight)

# ---------------------------------------------------------------------------
# 4. PREPARATION DES PIPELINES (performance + augmentation)
# ---------------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def prepare(ds, training=False):
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.cache().prefetch(buffer_size=AUTOTUNE)

train_ds_prepared = prepare(train_ds, training=True)
val_ds_prepared = prepare(val_ds, training=False)

# ---------------------------------------------------------------------------
# 5. CONSTRUCTION DU MODELE (Transfer Learning)
# ---------------------------------------------------------------------------
print("Construction du modele MobileNetV2...")

base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,       # on retire la derniere couche (ImageNet, 1000 classes)
    weights="imagenet",
)
base_model.trainable = False  # on gele le modele pre-entraine dans un premier temps

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------------------------------------------------------------------
# 6. ENTRAINEMENT - PHASE 1 : la "tete" seulement (base gelee)
# ---------------------------------------------------------------------------
print("\n=== Phase 1 : entrainement de la tete (base MobileNetV2 gelee) ===")

callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
]

history1 = model.fit(
    train_ds_prepared,
    validation_data=val_ds_prepared,
    epochs=EPOCHS_HEAD,
    class_weight=class_weight,
    callbacks=callbacks,
)

# ---------------------------------------------------------------------------
# 7. ENTRAINEMENT - PHASE 2 : fine-tuning des dernieres couches
# ---------------------------------------------------------------------------
print("\n=== Phase 2 : fine-tuning des dernieres couches de MobileNetV2 ===")

base_model.trainable = True
# On ne debloque que les 30 dernieres couches pour eviter l'overfitting
# et limiter le temps de calcul sur CPU.
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # taux tres bas pour le fine-tuning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history2 = model.fit(
    train_ds_prepared,
    validation_data=val_ds_prepared,
    epochs=EPOCHS_FINETUNE,
    class_weight=class_weight,
    callbacks=callbacks,
)

# ---------------------------------------------------------------------------
# 8. EVALUATION FINALE
# ---------------------------------------------------------------------------
print("\n=== Evaluation finale sur le jeu de validation ===")
loss, accuracy = model.evaluate(val_ds_prepared)
print(f"Validation accuracy: {accuracy:.4f} | Validation loss: {loss:.4f}")

# Matrice de confusion + rapport de classification
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

y_true = []
y_pred = []
for images, labels in val_ds_prepared:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

print("\nRapport de classification :")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\nMatrice de confusion :")
print(confusion_matrix(y_true, y_pred))

# ---------------------------------------------------------------------------
# 9. SAUVEGARDE DU MODELE
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
model.save(MODEL_OUTPUT_PATH)
print(f"\nModele sauvegarde : {MODEL_OUTPUT_PATH}")

# Sauvegarde aussi la liste des classes (utile pour l'app plus tard,
# car l'ordre alphabetique des classes doit correspondre a la sortie du modele)
with open("model/class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")
print("Liste des classes sauvegardee : model/class_names.txt")

# ---------------------------------------------------------------------------
# 10. GRAPHIQUES (accuracy / loss)
# ---------------------------------------------------------------------------
acc = history1.history["accuracy"] + history2.history["accuracy"]
val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss_hist = history1.history["loss"] + history2.history["loss"]
val_loss_hist = history1.history["val_loss"] + history2.history["val_loss"]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label="Train accuracy")
plt.plot(val_acc, label="Validation accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(loss_hist, label="Train loss")
plt.plot(val_loss_hist, label="Validation loss")
plt.legend()
plt.title("Loss")

plt.savefig("model/training_history.png")
print("Graphique sauvegarde : model/training_history.png")