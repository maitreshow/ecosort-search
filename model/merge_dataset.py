"""
Jalon 1 (bis) - Fusion des datasets pour ajouter la classe "electronic".

Ce script cree un nouveau dossier data/merged_dataset/ contenant 7 classes :
cardboard, glass, metal, paper, plastic, trash, electronic

- Les 6 premieres classes sont copiees depuis "Garbage classification"
- La classe "electronic" regroupe TOUTES les images du dataset e-waste
  (peu importe la sous-categorie d'origine : batterie, clavier, etc.)

Usage :
    python model/merge_dataset.py
"""

import os
import shutil

GARBAGE_DIR = "data/Garbage classification/Garbage classification"
EWASTE_DIR = "data/modified-dataset"
OUTPUT_DIR = "data/merged_dataset"

GARBAGE_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def copy_garbage_classes():
    """Copie les 6 classes existantes telles quelles."""
    for class_name in GARBAGE_CLASSES:
        src = os.path.join(GARBAGE_DIR, class_name)
        dst = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(dst, exist_ok=True)

        count = 0
        for fname in os.listdir(src):
            src_file = os.path.join(src, fname)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, os.path.join(dst, fname))
                count += 1
        print(f"  {class_name}: {count} images copiees")


def copy_electronic_class():
    """Regroupe toutes les sous-categories e-waste en une seule classe 'electronic'."""
    dst = os.path.join(OUTPUT_DIR, "electronic")
    os.makedirs(dst, exist_ok=True)

    count = 0
    for split in ["train", "test", "val"]:
        split_dir = os.path.join(EWASTE_DIR, split)
        if not os.path.isdir(split_dir):
            continue

        for sub_category in os.listdir(split_dir):
            sub_dir = os.path.join(split_dir, sub_category)
            if not os.path.isdir(sub_dir):
                continue

            for fname in os.listdir(sub_dir):
                src_file = os.path.join(sub_dir, fname)
                if os.path.isfile(src_file):
                    # Prefixe pour eviter les collisions de noms entre sous-categories
                    new_name = f"{split}_{sub_category}_{fname}".replace(" ", "_")
                    shutil.copy2(src_file, os.path.join(dst, new_name))
                    count += 1

    print(f"  electronic: {count} images copiees (fusion de 10 sous-categories)")


if __name__ == "__main__":
    print("Fusion des datasets en cours...\n")
    print("Copie des classes existantes (Garbage Classification) :")
    copy_garbage_classes()

    print("\nCopie et fusion de la classe electronic (E-Waste Dataset) :")
    copy_electronic_class()

    print(f"\nDataset fusionne cree dans : {OUTPUT_DIR}")
    print("\nRecapitulatif final :")
    for class_name in os.listdir(OUTPUT_DIR):
        class_path = os.path.join(OUTPUT_DIR, class_name)
        if os.path.isdir(class_path):
            n = len(os.listdir(class_path))
            print(f"  {class_name}: {n} images")
