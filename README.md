# ♻️ EcoSort-Search

Application Web containerisée d'aide au tri sélectif : l'utilisateur recherche un produit,
l'application le trouve sur Jumia, puis une IA (Deep Learning) détermine sa consigne de tri
et colore l'écran en conséquence.

## 🧱 Architecture

```
ecosort-search/
├── data/               # Dataset Kaggle (non versionné, voir .gitignore)
├── model/              # Script d'entraînement + modèle sauvegardé (.h5)
├── scraping/           # Module de scraping Jumia
├── app/                # Application web (Streamlit/Flask)
├── notebooks/          # Exploration / expérimentation
├── Dockerfile
├── requirements.txt
└── .gitignore
```

## 🏷️ Catégories de tri

| Catégorie | Couleur | Matières |
|---|---|---|
| Poubelle JAUNE | 🟡 | plastic, metal, cardboard |
| Poubelle VERTE | 🟢 | glass |
| Poubelle BLEUE | 🔵 | paper |
| Bac Électronique (D3E) | 🎛️ | mots-clés / classe dédiée |
| Poubelle MARRON/NOIRE | ⚫ | trash |

## 🚀 Lancer le projet

```bash
docker build -t ecosort .
docker run -p 8501:8501 ecosort
```

ou

```bash
docker-compose up -d --build
```

Puis ouvrir : http://localhost:8501

## 👥 Équipe & workflow Git

- 3 branches de développement : `dev-etudiantA`, `dev-etudiantB`, `dev-etudiantC`
- Aucun push direct sur `main`
- Toute contribution passe par une Pull Request relue par un autre membre

## 📅 Deadline

25/07/2026 23:59:59
