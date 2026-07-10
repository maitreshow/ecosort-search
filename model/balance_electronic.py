"""
Jalon 1 (bis) - Sous-echantillonnage de la classe 'electronic'.

La classe electronic a 3000 images contre 137-594 pour les autres classes.
Ce script en garde aleatoirement un nombre limite (600 par defaut) pour
eviter un desequilibre trop fort dans le dataset.

Usage :
    python model/balance_electronic.py
"""

import os
import random

ELECTRONIC_DIR = "data/merged_dataset/electronic"
MAX_IMAGES = 600
SEED = 42

if __name__ == "__main__":
    random.seed(SEED)

    all_files = [
        f for f in os.listdir(ELECTRONIC_DIR)
        if os.path.isfile(os.path.join(ELECTRONIC_DIR, f))
    ]
    print(f"Nombre d'images actuel dans 'electronic' : {len(all_files)}")

    if len(all_files) <= MAX_IMAGES:
        print(f"Deja en dessous de {MAX_IMAGES}, rien a faire.")
    else:
        # On choisit aleatoirement les fichiers a GARDER
        files_to_keep = set(random.sample(all_files, MAX_IMAGES))
        files_to_remove = [f for f in all_files if f not in files_to_keep]

        for fname in files_to_remove:
            os.remove(os.path.join(ELECTRONIC_DIR, fname))

        print(f"{len(files_to_remove)} images supprimees.")
        print(f"Il reste {MAX_IMAGES} images dans 'electronic'.")
