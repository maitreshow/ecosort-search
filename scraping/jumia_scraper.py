"""
Jalon 2a - Scraping Jumia.
Auteur: ARZIKA

Fonction principale : search_jumia(keyword, max_results=5)
Retourne une liste de dicts : {"nom": ..., "prix": ..., "image_url": ..., "lien": ...}

Usage (test rapide) :
    python scraping/jumia_scraper.py
"""

import time
import random
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.jumia.ci"
SEARCH_URL = f"{BASE_URL}/catalog/"

# En-tetes complets pour ressembler a un vrai navigateur (Jumia bloque
# les requetes trop "robotiques" avec des en-tetes minimalistes)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# On utilise une Session : elle garde les cookies entre les requetes,
# comme le ferait un vrai navigateur (visite la page d'accueil d'abord,
# puis la recherche, plutot qu'une requete "hors sol")
_session = requests.Session()
_session.headers.update(HEADERS)


def _warm_up_session():
    """Visite la page d'accueil pour recuperer des cookies valides avant de scraper."""
    try:
        _session.get(BASE_URL, timeout=10)
        time.sleep(1)  # petite pause, comportement plus humain
    except requests.RequestException:
        pass  # si ca echoue, on tente quand meme la recherche directement

def search_jumia(keyword: str, max_results: int = 5, max_retries: int = 3) -> list[dict]:
    """
    Recherche des produits sur Jumia CI a partir d'un mot-cle.
    Reessaie automatiquement en cas de blocage temporaire (403).
    """
    params = {"q": keyword}
    _warm_up_session()

    for tentative in range(1, max_retries + 1):
        try:
            response = _session.get(SEARCH_URL, params=params, timeout=10)
            response.raise_for_status()
            break  # succes, on sort de la boucle
        except requests.RequestException as e:
            print(f"[TENTATIVE {tentative}/{max_retries}] Echec : {e}")
            if tentative == max_retries:
                print("[ERREUR] Abandon apres plusieurs tentatives.")
                return []
            attente = tentative * 2 + random.uniform(0, 1)
            time.sleep(attente)

    soup = BeautifulSoup(response.text, "lxml")
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
                image_url = img_tag.get("data-src") or img_tag.get("src")
            results.append({
                "nom": nom, "prix": prix,
                "image_url": image_url, "lien": lien,
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
