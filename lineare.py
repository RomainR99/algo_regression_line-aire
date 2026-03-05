## A refaire
# A refaire
import pandas
import numpy as np
import matplotlib.pyplot as plt

#epsilon à 0,0001
# a et b aléatoire

if __name__ == "___main__":
    house_prices_df = pandas.read_csv("prix_maison.csv")
    x_mean, x_std =house_prices_df["surface"].mean(), house_prices_df["surface"].std()
    y_mean, y_std = house_prices_df["prix"].mean(), house_prices_df["prix"].std()

    house_prices_df["surface"] = (house_prices_df["surface"] - x_mean)/ x_std
    house_prices_df["price"] = (house_prices_df["price"] -y_mean)/ x_std

    a = np.random.randint(["surface"])
    b = np.random.randint(["price"])
    b = 0
    epsilon = 0.0001
    y_mean = a * x_mean + b
    L = [number for number in range (pandas.min.a,pandas.max.a,epsilon)]
    
    r = a * house_prices_df["surface"] + b


    for index, number in enumerate(L):
        mse = 1 / L (np.sum (( a - b )* ( a - b )))

        a = a - epsilon * 2 / L ((a * ["surface"] + b - ["price"]) * ["surface"])
        b = b - epsilon * 2 / L ((a * ["surface"] + b - ["price"]) * 1)

    

# plot
fig, ax = plt.subplots()

ax.plot(["surface"])
ax.plot(["price"])
ax.plot(r)
plt.show()
    
fig, ax = plt.subplots()
for pos in np.linspace(-2, 1, 10):
    ax.axline((pos, 0), slope=0.5, color='k', transform=ax.transAxes)
ax.set(xlim=(0, 1), ylim=(0, 1))
plt.show()
