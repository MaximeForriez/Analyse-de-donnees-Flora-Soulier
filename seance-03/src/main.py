#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/

# Sources des données : production de M. Forriez, 2016-2023

# Question 4 
with open("data/resultats-elections-presidentielles-2022-1er-tour.csv", "r") as fichier:
    contenu = pd.read_csv(fichier)

print(contenu.head())

# Question 5
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

colonnes_quanti = contenu.select_dtypes(include=["number"])
print("\nColonnes quantitatives :")
print(colonnes_quanti.head())

moyennes = colonnes_quanti.mean().round(2)

print("Moyennes des colonnes quantitatives :")
print(moyennes)

medians = colonnes_quanti.median().round(2)

print("Médianes des colonnes quantitatives :")
print(medians)

modes = colonnes_quanti.mode().iloc[0].round(2)

print("Modes des colonnes quantitatives :")
print(modes)

ecarts_types = colonnes_quanti.std().round(2)

print("Écarts-types des colonnes quantitatives :")
print(ecarts_types)

ecart_absolu_moyenne = np.abs(colonnes_quanti - colonnes_quanti.mean()).mean().round(2)

print("Écart absolu à la moyenne pour chaque colonnes quantitatives :")
print(ecart_absolu_moyenne)

etendue = (colonnes_quanti.max() - colonnes_quanti.min()).round(2)

print("Étendues des colonnes quantitatives :")
print(etendue)

# Question 6
parametres = pd.DataFrame({
    "Moyenne": moyennes,
    "Médiane": medians,
    "Mode": modes,
    "Écart-type": ecarts_types,
    "Écart absolu à la moyenne": ecart_absolu_moyenne,
    "Étendue": etendue
})

print("=== Résumé des paramètres ===")
print(parametres)

# Question 7 
colonnes_quanti = contenu.select_dtypes(include=["float64", "int64"]).columns
stats_quartiles_deciles = {}

for colonne in colonnes_quanti:
    Q1 = contenu[colonne].quantile(0.25)
    Q3 = contenu[colonne].quantile(0.75)
    IQR = round(Q3 - Q1, 2)

    D1 = contenu[colonne].quantile(0.10)
    D9 = contenu[colonne].quantile(0.90)
    IDR = round(D9 - D1, 2)

    stats_quartiles_deciles[colonne] = {"IQR" : IQR, "IDR" : IDR}

    stats_contenu = pd.DataFrame(stats_quartiles_deciles).T
    print(stats_contenu)

# Question 8
colonnes_quanti = contenu.select_dtypes(include=["float64", "int64"]).columns
for colonne in colonnes_quanti:
    plt.figure(figsize=(6,4))
    plt.boxplot(contenu[colonne], vert=True)
    plt.title(f"Boîte à moustaches de la colonne '{colonne}'")
    plt.ylabel(colonne)

    plt.savefig(f"IMG/boxplot_{colonne}.png")
    plt.close()

# Question 9 & 10

csv_path = "data/island-index.csv"
with open(csv_path, "r", encoding="utf-8") as fichier2:
    df = pd.read_csv(fichier2)

# vérification nom des colonnes
print(df.head())
print (df.columns.tolist())

# catégorisation & dénombrement
surface = df["Surface (km²)"]
bornes = [0, 10, 25, 50, 100, 2500, 5000, 10000, float('inf')]
intervalles = [
    "0-10",
    "10-25",
    "25-50",
    "50-100",
    "100-2500",
    "2500-5000",
    "5000-10000",
    "10000+"
]
categories = pd.cut(surface, bins=bornes, labels=intervalles, right=True, include_lowest=True)
df["Categorie_de_surface"] = categories
compte_categories = df["Categorie_de_surface"].value_counts().sort_index()
print("nombre d'îles par catégorie de surface :")
print(compte_categories)




