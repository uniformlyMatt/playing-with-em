import numpy as np
import pandas as pd


def gaussian(x: np.array, mu: float, sigma_squared: float) -> pd.Series:
    """ Compute Gaussian function with parameters mu and sigma """
    return (2*np.pi*sigma_squared)**(-0.5)*np.exp(-0.5*(x-mu)**2/sigma_squared)


def responsibility(y: np.array, pi_hat: float, mu_1: float, mu_2: float, sigma_1: float, sigma_2: float):
    """ Calculate the responsibilities for class 1 and class 2 """
    phi_1 = gaussian(y, mu_1, sigma_1)
    phi_2 = gaussian(y, mu_2, sigma_2)

    # expectation step
    gamma = pi_hat*phi_2/((1-pi_hat)*phi_1 + pi_hat*phi_2)

    return gamma


def updates(gamma: np.array, y: np.array, mu_1: float, mu_2: float):
    """ Compute the weighted means and variances """
    mu_hat_1 = np.sum((1-gamma)*y)/np.sum(1-gamma)
    mu_hat_2 = np.sum(gamma*y)/np.sum(gamma)

    sigma_hat_1 = np.sum((1-gamma)*(y - mu_1)**2)/np.sum(1-gamma)
    sigma_hat_2 = np.sum(gamma*(y - mu_2)**2)/np.sum(gamma)

    pi_hat = np.sum(gamma)/len(gamma)

    return pi_hat, mu_hat_1, mu_hat_2, sigma_hat_1, sigma_hat_2


def fit(y1, y2, pi_initial: float, tolerance: float):
    """ Perform EM on the data y1, y2 """

    # define the initial guesses
    y = np.concatenate((y1, y2), axis=None)
    pi = pi_initial
    mu_1 = y1.mean()
    mu_2 = y2.mean()
    sigma_1 = y1.var()
    sigma_2 = y2.var()

    error = 1
    gamma1 = np.ones(len(y))
    iterations = 0

    while error > tolerance:
        # expectation step
        gamma2 = responsibility(
            y,
            pi_hat=pi,
            mu_1=mu_1,
            mu_2=mu_2,
            sigma_1=sigma_1,
            sigma_2=sigma_2
        )

        # update the previous guesses
        pi, mu_1, mu_2, sigma_1, sigma_2 = updates(
            gamma2,
            y=np.concatenate((y1, y2), axis=None),
            mu_1=mu_1,
            mu_2=mu_2
        )

        # calculate the error
        error = np.sum((gamma1-gamma2)**2)
        iterations += 1

        # update gamma1
        gamma1 = gamma2

    return gamma1, mu_1, mu_2, sigma_1, sigma_2, pi, error, iterations
