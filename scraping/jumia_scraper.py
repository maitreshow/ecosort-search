"""
Jalon 2a - Scraping Jumia.
Auteur: ARZIKA

Fonction principale : search_jumia(keyword, max_results=5)
Retourne une liste de dicts : {"nom": ..., "prix": ..., "image_url": ..., "lien": ...}

Usage (test rapide) :
    python scraping/jumia_scraper.py
"""

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.jumia.ci"
SEARCH_URL = f"{BASE_URL}/catalog/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9",
}


def search_jumia(keyword: str, max_results: int = 5) -> list[dict]:
    """
    Recherche des produits sur Jumia CI a partir d'un mot-cle.
    Retourne une liste de dicts avec : nom, prix, image_url, lien
    """
    params = {"q": keyword}

    try:
        response = requests.get(
            SEARCH_URL, params=params, headers=HEADERS, timeout=10
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERREUR] Impossible de contacter Jumia : {e}")
        return []

    soup = BeautifulSoup(response.text, "lxml")

    # Structure habituelle des sites Jumia : chaque produit est dans
    # une balise <article class="prd _fb col c-prd">
    articles = soup.select("article.prd")

    if not articles:
        print("[AVERTISSEMENT] Aucun produit trouve. "
              "La structure HTML de Jumia a peut-etre change.")
        return []

    results = []
    for article in articles[:max_results]:
        try:
            link_tag = article.select_one("a.core")
            lien = BASE_URL + link_tag["href"] if link_tag else None

            nom_tag = article.select_one("h3.name")
            nom = nom_tag.get_text(strip=True) if nom_tag else "Nom inconnu"

            prix_tag = article.select_one("div.prc")
            prix = prix_tag.get_text(strip=True) if prix_tag else "Prix inconnu"

            img_tag = article.select_one("img.img")
            image_url = None
            if img_tag:
                # Jumia utilise souvent data-src pour le lazy loading
                image_url = img_tag.get("data-src") or img_tag.get("src")

            results.append({
                "nom": nom,
                "prix": prix,
                "image_url": image_url,
                "lien": lien,
            })
        except Exception as e:
            print(f"[AVERTISSEMENT] Erreur sur un produit : {e}")
            continue

    return results


if __name__ == "__main__":
    # Test rapide en ligne de commande
    mot_cle = input("Rechercher un produit sur Jumia : ")
    produits = search_jumia(mot_cle, max_results=5)

    if not produits:
        print("\nAucun resultat.")
    else:
        print(f"\n{len(produits)} produit(s) trouve(s) :\n")
        for i, p in enumerate(produits, 1):
            print(f"{i}. {p['nom']}")
            print(f"   Prix  : {p['prix']}")
            print(f"   Image : {p['image_url']}")
            print(f"   Lien  : {p['lien']}")
            print()
