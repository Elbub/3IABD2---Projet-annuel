# faire un : pip install google-search-results
from serpapi import GoogleSearch
import json
import requests
import os

# Clé d'API SerpApi
API_KEY = "4540644eb4bb34b2eb353bcbc8de301b18cff345ae3e5bb4d1669e0e4a001cfe"

# Paramètres de recherche
params = {
    "api_key": API_KEY,
    "q": "Greek ruins",
    "engine": "google_images",
    "cc":"US",
    "first": 1,
    "count": 50
}

# Dossier de destination pour enregistrer les images
destination_folder = "/Users/devarieuxclement/Desktop/ESGI/L3/S2/Projet Annuel/git_pa/3IABD2---Projet-annuel/database/photos_brutes/rome_grece"

# Appel de l'API SerpApi
search = GoogleSearch(params)       # data extraction on the SerpApi backend
results = search.get_dict()         # JSON -> Python dict

google_images_results = {
    'images_results': [],
    'suggested_searches': [],
    'refined_searches': results.get('refined_searches', []),
    'related_searches': results.get('related_searches', []),
    'shopping_results': results.get('shopping_results', [])
}

page_count = 0
page_limit = 50

while 'error' not in results and page_count < page_limit:
    # data extraction from current page will be here

    params['first'] += params['count']
    page_count += 1
    results = search.get_dict()

# Enregistrer les images dans le dossier de destination
if 'error' not in results and 'images_results' in results:
    images = results['images_results']
    for i, image in enumerate(images):
        image_url = image['original']

        # Vérifier le format de l'image (JPG ou PNG)
        if image_url.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_data = requests.get(image_url).content

            # Enregistrer l'image dans le dossier de destination
            file_name = f"image_{i}.{image_url.split('.')[-1]}"
            file_path = os.path.join(destination_folder, file_name)

            with open(file_path, "wb") as file:
                file.write(image_data)

            google_images_results['images_results'].append({'original': image_url, 'file_path': file_path})

            print(f"Image enregistrée : {file_name}")

print(json.dumps(google_images_results, indent=2, ensure_ascii=False))


#link = results['images_results'][0]['original']
#print(link)
#print(json.dumps(google_images_results, indent=2, ensure_ascii=False))
