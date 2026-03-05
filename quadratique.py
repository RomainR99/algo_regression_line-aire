import pandas
import numpy 
import matplotlib.pyplot as plt

#si besoin plus tard
##initialiser (a, b, c) aléatoirement la seed permet visuellement de mieux comprendre. le random reste le meme
import random
random.seed(1)

#on cré le plot pour la visualisation du graphique , récuperé sur exo lineare.py
def plot(problems, solutions, predictions):
	plt.figure(figsize=(15, 6))
	plt.plot(problems, solutions, "ro")
	plt.plot(problems, predictions, "g")
	plt.show()

def plot_rmse(rmse_values_per_epoch):
	plt.figure(figsize=(15, 6))
	plt.plot(rmse_values_per_epoch, "b")
	plt.xlabel("epochs") # 5e1 - tracer RMSE en fonction des epochs
	plt.ylabel("rmse") # 5e1 - tracer RMSE en fonction des epochs
	plt.show()

def quadratic_regression(a, b, c, problems):
	return a * problems**2 + b * problems + c

#B2 — Implémenter la MSE et la RMSE
#rmse racine carré de la mmse
def compute_rmse(predictions, solutions):
	n = len(predictions)
	return (1/n) * numpy.sqrt(numpy.sum(((predictions - solutions))**2))

def backpropagation(a, b, c, n, problems, solutions, learning_rate, visual=False):
	dmse_da = 2/n * sum(((a * problems**2 + b * problems + c) - solutions) * problems**2)
	dmse_db = 2/n * sum(((a * problems**2 + b * problems + c) - solutions) * problems)
	dmse_dc = 2/n * sum((a * problems**2 + b * problems + c) - solutions)


	a -= learning_rate * dmse_da
	b -= learning_rate * dmse_db
	c -= learning_rate * dmse_dc

	predictions = quadratic_regression(a, b, c, problems)
	rmse = compute_rmse(predictions, solutions)


	if visual :
		plot(problems=problems, solutions=solutions, predictions=predictions)

	return a, b, c, rmse

def gradient_descent(problems, solutions, learning_rate=10**(-3), epochs=100, a=None, b=None):
	#initialiser (a, b, c) aléatoirement
	a, b, c = random.random(), random.random(), random.random()

	n = len(problems)
	#stocker la RMSE à chaque epoch
	rmse_values_per_epoch = []

	#entraîner sur plusieurs epochs

	for index_epoch in range(epochs):

		a, b,c, rmse = backpropagation(a, b, c, n, problems, solutions, learning_rate)
		rmse_values_per_epoch.append(rmse)#stocker la RMSE à chaque epoch


	predictions = quadratic_regression(a, b, c, problems)

	#tracer :1. nuage de points + courbe prédite
	plot(problems=problems, solutions=solutions, predictions=predictions)

	return a, b, c, rmse_values_per_epoch

if __name__ == "__main__":
	house_prices_df = pandas.read_csv("prix_maisons.csv")

	x_mean, x_std = house_prices_df["surface"].mean(), house_prices_df["surface"].std()
	y_mean, y_std = house_prices_df["prix"].mean(), house_prices_df["prix"].std()

	#Prétraitement identique au TP lineare.py : standardisation de surface et prix (moyenne 0, écart-type 1). x_std:écart type de x
	#2. Calculer puis appliquer la standardisation
	house_prices_df["surface"] = (house_prices_df["surface"] - x_mean )/ x_std
	house_prices_df["prix"] = (house_prices_df["prix"] - y_mean )/ y_std

	#afficher : — les 5 premières lignes 
	#house_prices_df.head() est le nom de notre dataframe head donne les 5 premiere lignes
	# définir les variables pour le plot
	problems = house_prices_df["surface"]
	solutions = house_prices_df["prix"]
	#a = random.random() deja mis dans fonction gradient_descent
	#b = random.random()
	#prédiction lineaire
	#predictions = a * problems + b
	#c = random.random()
	#prédiction quadratique
	#predictions = a * problems**2 + b * problems + c
	plot(problems=problems, solutions=solutions, predictions=predictions)
	#5.1 E1 — Gradient Descent
	# gradient descent
	a, b, c, rmse_values = gradient_descent(problems, solutions)
	plot_rmse(rmse_values) # 5e1 -tracer RMSE en fonction des epochs
	print(house_prices_df.head())

	#Questions
	#— Pourquoi la standardisation aide la descente de gradient ?
		# La moyenne permet d'enlever et centré les datas: donnée centrées norméess la moyenne devient 0.
		# remet toute les grandeurs à la mem échelle : moyenne nulle et écart type de 1. 
	#— Que se passe-t-il si on ne normalise pas et que les surfaces sont en dizaines alors que les prix
	#sont en centaines de milliers ?
		# si on normalise pas, la mse ne sera pas pertinentes. les poids du modele a et b seront trop grand.	
	
	#1.2 A2 — Visualisation
	#Question : la relation semble-t-elle parfaitement linéaire? Que pourrait apporter un modèle quadratique ?
	#cf 1er_affichage.png , les points ne sont pas au cotes de la droite. c'est la regression lineaire
	# modèle quadratique miss square error permet de réduire la distance du résidu au carré. et rapprocher la droite des points.
	#cf 2eme_affichage.png le modele quadratique utilse une courbe pour etre plus proche des points.

	#2.1 B1 — Fonction de prédiction
	#quadratic_regression(a, b, c, x) = ax2 + bx + c
	#Quelles sont les trois poids du modèle ? : a b et c
	#En quoi ce modèle est-il plus flexible qu’un modèle affine ?le modele quadratique utilse une courbe pour etre plus proche des points.

	#B2 — Implémenter la MSE et la RMSE
	#Quelle différence d’interprétation entre MSE et RMSE ? — Pourquoi la RMSE est parfois plus lisible ?
	#les données étant normalisé, standardisé,la rmse est un calcul de performance. plus la rmse est faible, plus le modele est performant.

	#3 Partie C — Dérivation des gradients
	#erreur : ei = moy(yi) - yi
	#avec yi = axi^2 + bxi + c
	#1. Quelle est la dérivée de (moy(yi) − y)^2 par rapport à yˆ ? 2(moy(yi) - y) : (derivé de a carré )= 2a 
	#2. Quelleestladérivéede moy(yi) = ax^2 + bx +c parrapport à a,b,c? 2axi + b 
	#Où intervient le facteur 1/n ? c'est pour la rmse derivé de racine de n ici carré donc 2 donc 1/2.

	#4 Partie D — Backpropagation
	#À quoi sert le learning rate η ?
	# # c'est le pas de la derivé à une autre si il est tropr grand , loss en dents de scie et impossible de convergé.La dérivé n'est plus local.
	# Je risque de suater le minimum local.
	#Donner un symptôme d’un learning rate trop petit.On trop de calcul pour convergé donc trop d'énergie dépensé, on vit dans un monde de ressource limité.

	#5 Partie E — Boucle d’entraînement
	#5.1 E1 — Gradient Descent
	#La RMSE doit-elle être strictement décroissante à chaque epoch ?