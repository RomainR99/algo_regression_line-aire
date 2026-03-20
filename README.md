## Descente de gradient en ML (intuition)

En machine learning, on veut ajuster les parametres d'un modele pour minimiser une fonction de perte (ex: MSE/RMSE) mesuree entre les predictions `y_hat` et les valeurs reelles `y`.

La descente de gradient consiste a:

1. calculer le gradient de la perte par rapport aux parametres du modele
2. mettre a jour les parametres en suivant la direction qui reduit le plus la perte (avec un pas `learning_rate`, note `eta`)
3. repeter sur plusieurs `epochs` jusqu'a convergence ou stabilisation

Dans ce projet, on standardise aussi les variables (`surface`, `prix`) pour rendre la descente de gradient plus stable et plus efficace.

# Regressions (linéaire et quadratique) sur prix de maisons

Ce mini-projet contient des implémentations Python de régression par descente de gradient (avec visualisation) :

- régression linéaire : `y = a*x + b`
- régression quadratique : `y = a*x^2 + b*x + c`

Les modèles apprennent à prédire `prix` à partir de `surface`.

## Données

Le script s'appuie sur le fichier `prix_maisons.csv` (à la racine du projet), avec les colonnes :

- `surface` : variable explicative
- `prix` : valeur à prédire

Avant l'entraînement, les scripts standardisent les colonnes (moyenne 0, écart-type 1) pour stabiliser la descente de gradient.

## Dépendances

Installation :

```bash
pip install -r requirements.txt
```

Les dépendances attendues sont `pandas`, `numpy` et `matplotlib`.

## Lancer les exemples

Exécute depuis la racine du projet (`algo_regression_linéaire/`) :

```bash
python lineare_prof.py
```

```bash
python quadratique.py
```

## Ce que ça affiche

Selon le script :

- un nuage de points (`surface`, `prix`) et la courbe prédite
- l'évolution de l'erreur via la RMSE en fonction du nombre d'epochs (dans `lineare_prof.py` et `quadratique.py`)

## Personnalisation rapide

Tu peux ajuster `learning_rate` et `epochs` directement dans les scripts.
La graine `random.seed(1)` sert à rendre les résultats reproductibles.

