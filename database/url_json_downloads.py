import requests
import json

# URL de l'API SerpApi

#recherches sur la grece :
#url = "https://serpapi.com/search.json?engine=google_images&q=greek+ruins&google_domain=google.com&gl=us&hl=en&ijn=0&api_key=4540644eb4bb34b2eb353bcbc8de301b18cff345ae3e5bb4d1669e0e4a001cfe"

#recherches sur l'asie du sud est
#url = "https://serpapi.com/search.json?engine=google_images&q=south+east+Asia+ruins&google_domain=google.com&gl=us&hl=en&ijn=0&api_key=4540644eb4bb34b2eb353bcbc8de301b18cff345ae3e5bb4d1669e0e4a001cfe"

#recherches amerique du sud
url = "https://serpapi.com/search.json?engine=google_images&q=South+America+ancient+ruins&google_domain=google.com&gl=us&hl=en&ijn=0&api_key=4540644eb4bb34b2eb353bcbc8de301b18cff345ae3e5bb4d1669e0e4a001cfe"

# Dossier de destination pour enregistrer les images
# folder grece antique
#destination_folder = "/Users/devarieuxclement/Desktop/ESGI/L3/S2/Projet Annuel/git_pa/3IABD2---Projet-annuel/database/photos_brutes/rome_grece"

#folder asie sud est
#destination_folder = "/Users/devarieuxclement/Desktop/ESGI/L3/S2/Projet Annuel/git_pa/3IABD2---Projet-annuel/database/photos_brutes/asie_sud_est"

#folder amerique du sud
destination_folder = "/Users/devarieuxclement/Desktop/ESGI/L3/S2/Projet Annuel/git_pa/3IABD2---Projet-annuel/database/photos_brutes/amerique_sud"


# Envoi de la requête GET à l'URL
response = requests.get(url)

#headers pour blocage
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Vérification du statut de la réponse
if response.status_code == 200:
    # Conversion de la réponse JSON en dictionnaire Python
    results = json.loads(response.text)

    # Récupération des résultats de recherche d'images
    images_results = results.get("images_results", [])

    # Vérification si la liste des résultats d'images n'est pas vide
    if images_results:
        # Parcourir les résultats et enregistrer les images
        for i, image_result in enumerate(images_results):
            print(i)
            image_url = image_result.get('original')
            print(image_url)
            if image_url:
                file_name = f"image_{i}.jpg"  # Nom du fichier d'image, peut être ajusté en fonction de vos besoins

                # Téléchargement de l'image
                response = requests.get(image_url, headers=headers)
                print(response)
                if response.status_code == 200:
                    # Enregistrement de l'image dans le dossier de destination
                    with open(f"{destination_folder}/{file_name}", "wb") as file:
                        file.write(response.content)
                        print(f"Image enregistrée : {file_name}")
                else:
                    print(f"Échec du téléchargement de l'image : {image_url}")
            else:
                print(f"URL de l'image non disponible pour l'élément {i}")
            print(image_result)
    else:
        print("Aucun résultat d'image trouvé.")

    # Affichage des résultats de recherche complets
    print(json.dumps(results, indent=2, ensure_ascii=False))
else:
    print(f"Échec de la requête : {response.status_code} {response.reason}")
