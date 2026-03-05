import numpy
import pandas
import matplotlib.pyplot as plt
import random

random.seed(1)

def plot(problems, solutions, predictions):
	plt.figure(figsize=(15, 6))
	plt.plot(problems, solutions, "ro")
	plt.plot(problems, predictions, "g")
	plt.show()



def plot_rmse(rmse_values_per_epoch):
	plt.figure(figsize=(15, 6))
	plt.plot(rmse_values_per_epoch, "b")
	plt.xlabel("epochs")
	plt.ylabel("rmse")
	plt.show()



def linear_regression(a, b, problems):
	return a * problems + b



def backpropagation(a, b, n, problems, solutions, learning_rate, visual=False):
	dmse_da = 2/n * sum((a * problems + b - solutions) * problems)
	dmse_db = 2/n * sum(a * problems + b - solutions)

	a -= learning_rate * dmse_da
	b -= learning_rate * dmse_db
	
	predictions = linear_regression(a, b, problems)
	rmse = compute_rmse(predictions, solutions)


	if visual :
		plot(problems=problems, solutions=solutions, predictions=predictions)

	return a, b, rmse


def gradient_descent(problems, solutions, learning_rate=10**(-3), epochs=100, a=None, b=None):
	a, b = random.random(), random.random()

	n = len(problems)
	
	rmse_values_per_epoch = []

	for index_epoch in range(epochs):

		a, b, rmse = backpropagation(a, b, n, problems, solutions, learning_rate)
		rmse_values_per_epoch.append(rmse)


	predictions = linear_regression(a, b, problems)
	plot(problems=problems, solutions=solutions, predictions=predictions)

	return a, b, rmse_values_per_epoch



def compute_rmse(predictions, solutions):
	n = len(predictions)
	return (1/n) * numpy.sqrt(numpy.sum(((predictions - solutions))**2))





if __name__ == "__main__":
	house_prices_df = pandas.read_csv("prix_maisons.csv")

	x_mean, x_std = house_prices_df["surface"].mean(), house_prices_df["surface"].std()
	y_mean, y_std = house_prices_df["prix"].mean(), house_prices_df["prix"].std()


	house_prices_df["surface"] = (house_prices_df["surface"] - x_mean )/ x_std
	house_prices_df["prix"] = (house_prices_df["prix"] - y_mean )/ y_std

	# ##### l'implémentation de notre optimizer
	learning_rate=0.05
	print("Il s'agit du modèle entrainé sur 50 epochs")
	a, b, rmse_values_per_epoch = gradient_descent(house_prices_df["surface"], house_prices_df["prix"], learning_rate=learning_rate, epochs=50)
	
	plot_rmse(rmse_values_per_epoch)
	print(learning_rate, rmse_values_per_epoch[-1])