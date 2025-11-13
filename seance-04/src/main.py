#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.stats import rv_discrete, binom, randint, poisson, norm, lognorm, uniform, chi2, pareto

#https://docs.scipy.org/doc/scipy/reference/stats.html


dist_names = ['norm', 'beta', 'gamma', 'pareto', 't', 'lognorm', 'invgamma', 'invgauss',  'loggamma', 'alpha', 'chi', 'chi2', 'bradford', 'burr', 'burr12', 'cauchy', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'genpareto', 'gausshyper', 'gibrat', 'gompertz', 'gumbel_r', 'pareto', 'pearson3', 'powerlaw', 'triang', 'weibull_min', 'weibull_max', 'bernoulli', 'betabinom', 'betanbinom', 'binom', 'geom', 'hypergeom', 'logser', 'nbinom', 'poisson', 'poisson_binom', 'randint', 'zipf', 'zipfian']

print(dist_names)

# Question 1 : Loi Dirac 

x0 = 5
valeurs = [x0]
probabilites = [1.0]

dirac_dist = rv_discrete(name='dirac', values=(valeurs, probabilites))

sample = dirac_dist.rvs(size=1000)

plt.bar([5], [1.0], width=0.5, color='skyblue', edgecolor='black')
plt.xlabel("Valeurs")
plt.ylabel("Fréquence")
plt.title("Distribution discrète : Loi de Dirac")

plt.savefig("dirac_distribution.png")
print("graphique enregistré")

# Loi uniforme discrète
plt.clf()

a, b = 1, 7
uniform_dist = randint(a, b)

sample_size = 1000
sample = uniform_dist.rvs(size=sample_size)

valeurs = np.arange(a, b)

probabilites = [uniform_dist.pmf(x) for x in valeurs]

plt.bar(valeurs, probabilites, width=0.6, color='lightgreen', edgecolor='black')
plt.xlabel("Valeurs")
plt.ylabel("Probabilités")
plt.title("Distribution discrète : Loi uniforme discrète")
plt.xticks(valeurs)

plt.savefig("uniforme_discrete_distribution.png")
print("graphique 2 enregistré")

# Loi binomiale
plt.clf()

n = 10
p = 0.5

valeurs = np.arange(0, n + 1)
probabilites = binom.pmf(valeurs, n, p)

from scipy.stats import binom
probabilites = binom.pmf(valeurs, n, p)

binom_dist = rv_discrete(name='binom', values=(valeurs, probabilites))

sample_size = 1000
sample = binom_dist.rvs(size=sample_size)

plt.bar(valeurs, probabilites, width=0.6, color='salmon', edgecolor='black')
plt.xlabel("Valeurs")
plt.ylabel("Probabilités")
plt.title("Distribution discrète : Loi binomiale")
plt.xticks(valeurs)

plt.savefig("binomial_distribution.png")
print("graphique 3 enregistré")

# Loi de Poisson

plt.clf()

lambda_poisson = 3

max_val = 15
valeurs = np.arange(0, max_val + 1)
probabilites = poisson.pmf(valeurs, lambda_poisson)
poisson_dist = rv_discrete(name='poisson', values=(valeurs, probabilites))

sample_size = 1000
sample = poisson_dist.rvs(size=sample_size)

plt.bar(valeurs, probabilites, width=0.6, color='orchid', edgecolor='black')
plt.xlabel("nombre d'événements")
plt.ylabel("Probabilités")
plt.title("Distribution discrète : Loi de Poisson")
plt.xticks(valeurs)
plt.savefig("poisson_distribution.png")
print("graphique 4 enregistré")

# Loi de Zipf
plt.clf()

s = 1.2
q = 1
N = 10

valeurs = np.arange(1, N + 1)

raw_prob = 1 / (valeurs + q) ** s
probabilites = raw_prob / raw_prob.sum()

zipf_mandelbrot_dist = rv_discrete(name='zipf_mandelbrot', values=(valeurs, probabilites))
sample_size = 1000
sample = zipf_mandelbrot_dist.rvs(size=sample_size)

plt.bar(valeurs, probabilites, width=0.6, color='gold', edgecolor='black')
plt.xlabel("Valeurs")
plt.ylabel("Probabilités")
plt.title("Distribution discrète : Loi de Zipf-Mandelbrot")
plt.xticks(valeurs)

plt.savefig("zipf_mandelbrot_distribution.png")
print("graphique 5 enregistré")

# distributions continues : La loi poisson 
plt.clf()

lambda_poisson = 20
x = np.linspace(0, 40, 500)

mu = lambda_poisson
sigma = np.sqrt(lambda_poisson)
pdf_values = norm.pdf(x, loc=mu, scale=sigma)

plt.plot(x, pdf_values, color='blue', lw=2)
plt.xlabel("valeurs")
plt.ylabel("Densité de probabilité")
plt.title("distribution continue approximative de la loi Poisson") 
plt.grid(True)
plt.savefig("poisson_continuous_approximation.png")
print("graphique 6 enregistré")

# Loi normale
plt.clf()

mu = 0
sigma = 1
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
pdf_values = norm.pdf(x, loc=mu, scale=sigma)

plt.plot(x, pdf_values, color='purple', lw=2)
plt.xlabel("valeurs")
plt.ylabel("Densité de probabilité")
plt.title("Distribution continue : Loi normale")
plt.grid(True)
plt.savefig("normal_distribution.png")
print("graphique 7 enregistré")

# Loi log-normale
plt.clf()

sigma = 0.5
mu = 0
x = np.linspace(0.01, 5, 500)
pdf_values = lognorm.pdf(x, s=sigma, scale=np.exp(mu))

plt.plot(x, pdf_values, color='brown', lw=2)
plt.xlabel("valeurs")
plt.ylabel("Densité de probabilité")
plt.title("Distribution continue : Loi log-normale")
plt.grid(True)
plt.savefig("lognormal_distribution.png")
print("graphique 8 enregistré")

# Loi uniforme continue
plt.clf()

a = 0
b = 1
x = np.linspace(a, b, 500)

pdf_values = uniform.pdf(x, loc=a, scale=b - a)

plt.plot(x, pdf_values, color='teal', lw=2)
plt.xlabel("valeurs")
plt.ylabel("Densité de probabilité")
plt.title("Distribution continue : Loi uniforme continue")
plt.grid(True)
plt.savefig("uniform_continuous_distribution.png")
print("graphique 9 enregistré")

# Loi de x²
plt.clf()

df = 5
x = np.linspace(0, 20, 500)
pdf_values = chi2.pdf(x, df=df)

plt.plot(x, pdf_values, color='orange', lw=2)
plt.xlabel("valeurs")
plt.ylabel("Densité de probabilité")
plt.title("Distribution continue : Loi x²")
plt.grid(True)
plt.savefig("x²_distribution.png")
print("graphique 10 enregistré")

# Loi de Pareto
plt.clf()

b = 2.62
scale = 1
x = np.linspace(scale, 5, 500)
pdf_values = pareto.pdf(x, b, scale=scale)

plt.plot(x, pdf_values, color='red', lw=2)
plt.xlabel("valeurs")
plt.ylabel("Densité de probabilité")
plt.title("Distribution continue : Loi de Pareto")
plt.grid(True)
plt.savefig("pareto_distribution.png")
print("graphique 11 enregistré")

# Question 2 Fonctions de caclul de moyenne et d'écart type
def calculer_moyenne(distribution, n_points=10000):
    if hasattr(distribution, "rvs"):
        sample = distribution.rvs(size=n_points)
        return sample.mean()
    else:
        raise ValueError("distrib invalide")

def calculer_ecart_type(distribution, n_points=10000):
    if hasattr(distribution, "rvs"):
        sample = distribution.rvs(size=n_points)
        return sample.std()
    else:
        raise ValueError("distrib non reconnue")

# variables discrètes 
# Dirac
valeurs_dirac = np.array([5])
prob_dirac = np.array([1])
dirac_dist = rv_discrete(name='dirac', values=(valeurs_dirac, prob_dirac))

# uniforme discrète
valeurs_uniforme = np.arange(1, 6)
prob_uniforme = np.ones_like(valeurs_uniforme) / len(valeurs_uniforme)
uniform_dist = rv_discrete(name='uniform_discrete', values=(valeurs_uniforme, prob_uniforme))

# Binomiale
n, p = 10, 0.5
valeurs_binom = np.arange(0, 11)
prob_binom = binom.pmf(valeurs_binom, n=10, p=0.5)
prob_binom = prob_binom / prob_binom.sum()

# Poisson
lambda_poisson = 3
valeurs_poisson = np.arange(0, 15)
prob_poisson = poisson.pmf(valeurs_poisson, mu=lambda_poisson)
poisson_dist = rv_discrete(name='poisson', values=(valeurs_poisson, prob_poisson))

# Zipf
s, q, N = 1.2, 1, 10
valeurs_zipf = np.arange(1, N+1)
raw_prob_zipf = 1 / (valeurs_zipf + q)**s
prob_zipf = raw_prob_zipf / raw_prob_zipf.sum()
zipf_dist = rv_discrete(name='zipf_mandelbrot', values=(valeurs_zipf, prob_zipf))

# variables continues
# poisson
lambda_cont = 20
poisson_continu = norm(loc=lambda_cont, scale=np.sqrt(lambda_cont))

# normale
mu, sigma = 0, 1
normal_dist = norm(loc=mu, scale=sigma)

#log normale
mu_logn, sigma_logn = 0, 0.5
lognormal_dist = lognorm(s=sigma_logn, scale=np.exp(mu_logn))

# uniforme
a, b = 0, 1
uniforme_continu = uniform(loc=a, scale=b-a)

# x²
df = 5
chi2_dist = chi2(df=df)

# Pareto 
b_pareto, scale_pareto = 2.62, 1
pareto_dist = pareto(b_pareto, scale=scale_pareto)

distributions = {
    "Dirac": dirac_dist,
    "Uniforme discrète": uniform_dist,
    "Binomiale": binom_dist,
    "Poisson discrète": poisson_dist,
    "Zipf": zipf_dist,
    "poisson continue": poisson_continu,
    "normale": normal_dist,
    "lognormale": lognormal_dist,
    "uniforme continue": uniforme_continu,
    "x²": chi2_dist,
    "pareto": pareto_dist
}

for nom, dist in distributions.items():
    moy = calculer_moyenne(dist)
    ecart = calculer_ecart_type(dist)
    print(f"{nom} : moyenne = {moy:.2f}, écart-type= {ecart:.2f}")
