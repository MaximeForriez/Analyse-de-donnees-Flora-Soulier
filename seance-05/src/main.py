#coding:utf8

import pandas as pd
import math
import scipy
import scipy.stats
import numpy as np

#C'est la partie la plus importante dans l'analyse de données. D'une part, elle n'est pas simple à comprendre tant mathématiquement que pratiquement. D'autre, elle constitue une application des probabilités. L'idée consiste à comparer une distribution de probabilité (théorique) avec des observations concrètes. De fait, il faut bien connaître les distributions vues dans la séance précédente afin de bien pratiquer cette comparaison. Les probabilités permettent de définir une probabilité critique à partir de laquelle les résultats ne sont pas conformes à la théorie probabiliste.
#Il n'est pas facile de proposer des analyses de données uniquement dans un cadre univarié. Vous utiliserez la statistique inférentielle principalement dans le cadre d'analyses multivariées. La statistique univariée est une statistique descriptive. Bien que les tests y soient possibles, comprendre leur intérêt et leur puissance d'analyse dans un tel cadre peut être déroutant.
#Peu importe dans quelle théorie vous êtes, l'idée de la statistique inférentielle est de vérifier si ce que vous avez trouvé par une méthode de calcul est intelligent ou stupide. Est-ce que l'on peut valider le résultat obtenu ou est-ce que l'incertitude qu'il présente ne permet pas de conclure ? Peu importe également l'outil, à chaque mesure statistique, on vous proposera un test pour vous aider à prendre une décision sur vos résultats. Il faut juste être capable de le lire.

#Par convention, on place les fonctions locales au début du code après les bibliothèques.
def ouvrirUnFichier(nom):
    with open(nom, "r") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

#Théorie de l'échantillonnage (intervalles de fluctuation)
#L'échantillonnage se base sur la répétitivité.
print("Résultat sur le calcul d'un intervalle de fluctuation")

donnees = pd.DataFrame(ouvrirUnFichier("./data/Echantillonnage-100-Echantillons.csv"))
print(donnees)

#calcul de la moyenne 
df = pd.read_csv(r"C:\Users\flora\Documents\Soulier-2025-2026-Analyse-de-donnees\seance-05\src\data\Echantillonnage-100-Echantillons.csv")
df = df.apply(pd.to_numeric, errors="coerce")
print(df.select_dtypes(include="number").columns)
moyennes = df.mean()
moyennes_arrondies = moyennes.apply(lambda x: round(x, 0))
print("calcul moyennes")
moyenne_pour = round(df["Pour"].astype(float).mean(), 0)
print("moyenne_pour")
print(moyenne_pour)
moyenne_contre = round(df["Contre"].astype(float).mean(), 0)
print("moyenne_contre")
print(moyenne_contre)
moyenne_sans_opinion = round(df["Sans opinion"].astype(float).mean(), 0)
print("moyenne_sans_opinion")
print(moyenne_sans_opinion)

#calcul des fréquences échantillon
moyennes = donnees.mean().round(0)
somme_moyennes= moyennes.sum()
frequences = moyennes / somme_moyennes.round(2)

print("Moyennes des colonnes :")
print(moyennes)
print("\nSomme des moyennes :", somme_moyennes)
print("\nFréquences (échantillon) :")
print(frequences)

#calcul des fréquences population mère
moy_pop = pd.DataFrame(ouvrirUnFichier("./data/Echantillonnage-Population-reelle.csv"))

pop_pour = 852
pop_contre = 911
pop_sans_opinion = 422
pop_totale = 2185

frequence_pop = pd.Series({
    "Pour": round(pop_pour / pop_totale, 2),
    "Contre": round(pop_contre / pop_totale, 2),
    "Sans Opinion": round(pop_sans_opinion / pop_totale, 2)
})
print(frequence_pop)
#intervalle de fluctuation

n = len(donnees)
p = frequence_pop
zc = 1.96
intervalle_fluctuation = {}

for cat, p in frequences.items():
    sigma = np.sqrt(p * (1 - p) / n)
    IF_inf = p - 1.96 * sigma
    IF_sup = p + 1.96 * sigma
    intervalle_fluctuation[cat] = (IF_inf, IF_sup)

print("intervalles de fluctuation à 95%):")
for cat, (IF_inf, IF_sup) in intervalle_fluctuation.items():
    print(f"{cat} : [{IF_inf}, {IF_sup}]")


#Théorie de l'estimation (intervalles de confiance)
#L'estimation se base sur l'effectif.

premier_echantillon = donnees.iloc[0]
ligne = list(premier_echantillon.astype(int))
colonnes = list(donnees.columns)
print("premier_echantillon")
for nom, val in zip(colonnes, ligne):
    print(f"{nom} : {val}")

n = sum(ligne)
print(f"\nEffectif du premier échantillon : {n}")

frequences = {nom: round(val / n, 2) for nom, val in zip(colonnes, ligne)}
print("Fréquences du premier échantillon :")
for nom, freq in frequences.items():
    print(f"{nom} : {freq}")

#intervalle de confiance
zc = 1.96
intervalle_confiance = {}
for nom, val in zip(colonnes, ligne):
    p = val / n
    sigma = math.sqrt((p * (1 - p)) / n)
    IC_inf = max(0.0, round(p - sigma, 3))
    IC_sup = min(1.0, round(p + sigma, 3))
    intervalle_confiance[nom] = (IC_inf, IC_sup)

print("Intervalles de confiance à 95% :")
for nom, (IC_inf, IC_sup) in intervalle_confiance.items():
    print(f"{nom} : [{IC_inf}, {IC_sup}]")

#Théorie de la décision (tests d'hypothèse)
#La décision se base sur la notion de risques alpha et bêta.
#Comme à la séance précédente, l'ensemble des tests se trouve au lien : https://docs.scipy.org/doc/scipy/reference/stats.html
print("Théorie de la décision")

from scipy.stats import shapiro

fichier_test_1 = pd.DataFrame(ouvrirUnFichier("./data/Loi-normale-test-1.csv"))
fichier_test_2 = pd.DataFrame(ouvrirUnFichier("./data/Loi-normale-test-2.csv"))

#test de l'ouverture des fichiers : 
print (fichier_test_1)
print (fichier_test_2)

stat1, p_value1 = shapiro(fichier_test_1)
stat2, p_value2 = shapiro(fichier_test_2)


print ("résultat du test de normalité de Shapiro-Wilk pour fichier 1")
print ("statistique = ", round(stat1, 4), "p-value=", round(p_value1, 6))
if p_value1 > 0.05:
        print("Conclusion : Les données suivent une loi normale (on ne rejette pas H0)\n")
else:
        print("Conclusion : Les données ne suivent pas une loi normale (on rejette H0)\n")

print ("résultat du test de normalité de Shapiro-Wilk pour fichier 2")
print ("statistique = ", round(stat2, 4), "p-value=", round(p_value2, 6))
if p_value2 > 0.05:
        print("Conclusion : Les données suivent une loi normale (on ne rejette pas H0)\n")
else:
        print("Conclusion : Les données ne suivent pas une loi normale (on rejette H0)\n")
   