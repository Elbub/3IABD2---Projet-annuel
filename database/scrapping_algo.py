# important de faire un : pip install google-search-results
from serpapi import GoogleSearch
import json
import requests

# Clé d'API SerpApi
API_KEY = "4540644eb4bb34b2eb353bcbc8de301b18cff345ae3e5bb4d1669e0e4a001cfe"

# Paramètres de recherche
params = {
    "api_key": API_KEY,
    "q": "Greek ruins",
    "engine": "google_images",
    "cc":"US",
    "ijn" : 1 # ce paramètre correspond à la page de google image à télécharger
}

# headers pour passer dans certains cas une sécurité qui bloque les algo de scrapping
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Dossier de destination pour enregistrer les images
destination_folder = "/Users/devarieuxclement/Desktop/ESGI/L3/S2/Projet Annuel/git_pa/3IABD2---Projet-annuel/database/photos_brutes/rome_grece"

# Appel de l'API SerpApi
search = GoogleSearch(params)       # nous donne la réponse de l'appel api
results = search.get_dict()         # JSON -> Python dict


if 'error' not in results and 'images_results' in results:
    images = results['images_results']
    for i, image in enumerate(images):
        image_url = image['original']
        if image_url:
            num = i + 100 # penser à mettre la bonne centaine ici, cela a pour but de nommer différemment nos images
            file_name = f"image_{num}.jpg" # nom du fichier
            response = requests.get(image_url, headers= headers)
            # Enregistrer l'image dans le dossier de destination
            if response.status_code == 200:
                with open(f"{destination_folder}/{file_name}", "wb") as file:
                    file.write(response.content)
                    print(f"Image enregistrée : {file_name}")
            else :
                print(f"Echec du téléchargement de l'image : {image_url}")


print(json.dumps(results, indent=2, ensure_ascii=False))

