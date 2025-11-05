#coding:utf8

import pandas as pd
import matplotlib.pyplot as plt
import re

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/
with open("./data/resultats-elections-presidentielles-2022-1er-tour.csv","r") as fichier:
    contenu = pd.read_csv(fichier)

# Mettre dans un commentaire le numéro de la question
# Question 5
print("=== Contenu du tableau ===")
print(contenu)

# Question 6
nb_lignes = len(contenu)
nb_colonnes = len(contenu.columns)
print(f"Nombre de lignes : {nb_lignes}")
print(f"Nombre de colonnes : {nb_colonnes}")

# Question 7
types_colonnes = contenu.dtypes

for colonne, type_val in types_colonnes.items():
    if "int" in str(type_val):
        type_simple = "int"
    elif "float" in str(type_val):
        type_simple = "float"
    elif "bool" in str(type_val):
        type_simple = "bool"
    else : 
        type_simple = "str"
    print(f"Colonne '{colonne}' => {type_simple}") 

# Question 8
print("=== Noms des colonnes ===")
print(contenu.columns)

print("=== Aperçu de la première ligne ===")
print(contenu.head(1))

# Question 9
print("=== Liste complète des colonnes ===")
for col in contenu.columns:
    print(col) 
print("=== Nombre d'inscrits ===")
print(contenu["Inscrits"])

# Question 10
print("=== Somme des colonnes quantitatives ===")
somme_colonnes = []

for colonne, type_val in contenu.dtypes.items():
    if "int" in str(type_val) or "float" in str(type_val):
        somme = contenu[colonne].sum()
        somme_colonnes.append((colonne, somme))
        print(f"{colonne} => {somme}")

print("\nListe des sommes :", somme_colonnes)

# Question 11
import os 

os.makedirs("images", exist_ok=True)

print("=== Création des diagrammes en barres par département ===")

col_departements = "Libellé du département"
col_inscrits = "Inscrits"
col_votants = "Votants"

for index, row in contenu.iterrows():
    departement = row[col_departements]
    inscrits = row[col_inscrits]
    votants = row[col_votants]

    plt.figure(figsize=(5, 4))

    plt.bar(["Inscrits", "Votants"], [inscrits, votants], color=['blue', 'orange'])
    plt.title(f"Département : {departement}")
    plt.ylabel("Nombre de personnes")
    
    plt.tight_layout()

    nom_fichier = f"images/{departement.replace('/', '_')}.png"
    plt.savefig(nom_fichier)
    plt.close()

print("Diagramme sauvegardé dans le dossier 'image/'")

# Question 12
print("=== Création des diagrammes circulaires par département ===")

col_departements = "Libellé du département"
col_blancs = "Blancs"
col_nuls = "Nuls"
col_exprimés = "Exprimés"
col_abstentions = "Abstentions"

import re
os.makedirs("images_pie", exist_ok=True)

for dep in contenu["Libellé du département"]:
    dep_nom = re.sub(r'[\\/*?:"<>|]', "_", dep)
    
    data_dep = contenu[contenu["Libellé du département"] == dep]
    
    valeurs = [
        data_dep["Blancs"].sum(),
        data_dep["Nuls"].sum(),
        data_dep["Exprimés"].sum(),
        data_dep["Abstentions"].sum()
    ]

    labels = ["Blancs", "Nuls", "Exprimés", "Abstentions"]

    plt.figure(figsize=(6, 6))
    plt.pie(valeurs, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f"Répartition des votes - {dep}")

    plt.savefig(f"images_pie/repartition_votes_{dep_nom}.png")
    plt.close()

    print("=== Diagramme circulaire sauvegardé dans le dossier 'images_pie/'===")

# Question 13

import matplotlib.pyplot as plt
import os

print("=== Création de l'histogramme de la distribution des inscrits ===")
 
col_inscrits = "Inscrits"

os.makedirs("images_histogrammes", exist_ok=True)

plt.figure(figsize=(8, 6))
plt.hist(contenu[col_inscrits], bins=10, edgecolor='black', alpha=0.7)
plt.title("Distribution des inscrits par département")
plt.xlabel("Nombre d'inscrits")
plt.ylabel("Fréquence")

plt.savefig("images_histogrammes/distribution_inscrits.png")

print("=== Histogramme sauvegardé dans 'images_histogrammes/distribution_inscrits.png' ===")
print("Fin du scrpt atteinte")

