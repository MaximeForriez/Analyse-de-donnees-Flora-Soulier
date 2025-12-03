#coding:utf8

import pandas as pd
import math
import scipy
import scipy.stats

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

#calcul des fréquences
somme_moyennes= moyennes.sum()
frequences = moyennes / somme_moyennes

print("Moyennes des colonnes :")
print(moyennes)
print("\nSomme des moyennes :", somme_moyennes)
print("\nFréquences (échantillon) :")
print(frequences)

population = pd.DataFrame(ouvrirUnFichier("./data/Echantillonnage-Population-reelle.csv"))

print("Moyennes des colonnes de la pop mère :")
moyennes_population = population.mean()
somme_moyennes_pop = moyennes_population.sum()
frequences_population = moyennes_population / somme_moyennes_pop
print(moyennes_population)
print("\nSomme des moyennes pop :", somme_moyennes_pop)
print("\nFréquences (pop mère) :")
print(frequences_population)


#Théorie de l'estimation (intervalles de confiance)
#L'estimation se base sur l'effectif.
print("Résultat sur le calcul d'un intervalle de confiance")

#Théorie de la décision (tests d'hypothèse)
#La décision se base sur la notion de risques alpha et bêta.
#Comme à la séance précédente, l'ensemble des tests se trouve au lien : https://docs.scipy.org/doc/scipy/reference/stats.html
print("Théorie de la décision")
