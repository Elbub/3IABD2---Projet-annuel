from serpapi import GoogleSearch
import json
import requests
from environs import Env
from datetime import datetime


env = Env()
env.read_env()

# Clé d'API SerpApi
API_KEY = env.str("SERPAPI_KEY")


def get_images(api_key, search: str, page: int = 1, pays_requete: str = "US"):
    # Paramètres de recherche
    params = {
        "api_key": api_key,
        "q": search,
        "engine": "google_images",
        "cc": pays_requete,
        "ijn": page,  # ce paramètre correspond à la page de google image à télécharger
    }

    # Appel de l'API SerpApi
    search = GoogleSearch(params)  # nous donne la réponse de l'appel api
    results = search.get_dict()  # JSON -> Python dict
    return results


def store_images(results, destination_folder: str):
    day = datetime.now()

    # Dossier de destination pour enregistrer les images
    destination_folder = f"../photos_brutes/{destination_folder}"

    # headers pour passer dans certains cas une sécurité qui bloque les algo de scrapping
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    if "error" not in results and "images_results" in results:
        images = results["images_results"]
        for i, image in enumerate(images):
            image_url = image["original"]
            if image_url:
                # num = i + 100 # penser à mettre la bonne centaine ici, cela a pour but de nommer différemment nos images
                file_name = (
                    f"image_{day.strftime('%Y-%m-%d_%H-%M')}_{i}.jpg"  # nom du fichier
                )
                try:
                    response = requests.get(image_url, headers=headers)
                    # Enregistrer l'image dans le dossier de destination
                    if response.status_code == 200:
                        print(f"image : {i}")
                        with open(f"{destination_folder}/{file_name}", "wb") as file:
                            file.write(response.content)
                            print(f"Image enregistrée : {file_name}")
                    else:
                        print(f"Echec du téléchargement de l'image : {image_url}")
                except Exception as e:
                    print(e)

        # print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    print("hello")
    res = get_images(API_KEY, "ruines greco romaines", 1, "FR")
    destination_folder = ["rome_grece", "amerique_sud", "asie_sud_est"]
    store_images(res, destination_folder[0])
