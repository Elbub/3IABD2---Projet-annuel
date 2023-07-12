# 3IABD2---Projet-annuel
Annual project for our bachelor in AI &amp; Big Data

## Lancement UI
Pour lancer l'UI, se positionner dans le dossier 3IABD2---Projet-annuel dans le terminal puis lancer la commande "python .\UI\UI.py" afin de lancer le script UI.py

Avec l'UI il est possible de charger un modèle déjà existant qui vont se trouver dans le dossier "/save/model".
En cas d'erreur, mettez à jour les chemins vers les datasets

Si vous souhaitez créer votre propre modèle, vous avez le choix entre le modèle linéaire, un PMC et un RBF:
- Pour se faire cliquez sur "Edit Model"

- Vous avez aussi la possibilité de sauvegarder votre modèle

- Lorsqu'un entrainement est finalisé, il est possible de relancer un autre entrainement en gardant le modèle déjà entrainé. Pour cela il faut simplement relancer un "Train Model". 

## Pour build et générer un nouveau fichier DLL

Se positionner dans le dossier "rust_lib" puis lancer la commande "cargo build --release --lib"
Le fichier DLL va ensuite se situer dans "rust_lib/target/release/rust_lib.dll"


## Les scripts cas de tests

Les scripts vérifiant les différents cas de tests vont être séparés en plusieurs notebook en fonction du modèle (linéaire, regression, pmc, rbf).
Ils sont disponibles dans le dossier tests_interops.

## Les scripts d'entrainement d'un modèle sur un dataset

Afin de tester un entrainement de modèle sur un dataset, un notebook est à disposition dans "implementation_dataset/mlp_dataset_bigger_batch.ipynb"

Attention à bien mettre le bon chemin dans la variable "directory_inputs" et "directory_test_inputs"

## Les datasets

Nous avons 2 datasets différents:
- le premier permet de classifier des monuments antiques gréco-romains, d'asie du sud est et des chateaux antiques
- le second permet de classifier le monument Angkor (Cambodge), le chateau bodiam (Angleterre) et le parthénon (Grèce)

Ces datasets sont disponibles sous deux formes : 
- dans leurs formes originales dans le dossier "database/original_size_dataset" 
- sous forme resized dans le dossier "database/..", plusieurs dossiers sont à disposition en fonction de la taille du resize

## Script Resizing d'images

Afin de resize des images à la taille souhaitée, un script est disponible dans "database/scripts/resizing.ipynb"


### Intervenants
Gabriel Bonjour
Badr Bouaissa
Clément Devarieux
