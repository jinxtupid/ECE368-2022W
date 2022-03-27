import numpy as np
import matplotlib.pyplot as plt
import util


def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)

    Inputs:
    ------
    beta: hyperparameter in the prior distribution

    Outputs: None
    -----
    """

    # a0, a1 range is set between -1 and 1
    a0 = np.linspace(-1, 1, 100)
    a1 = np.linspace(-1, 1, 100)
    A0, A1 = np.meshgrid(a0, a1)

    """ 
    A0:  [-1, -0.98 .... 0.98, 1]
         ...
        [-1, -0.98.... 0.98, 1]
    
    A1:  [-1, -1 ... -1, -1]
        [-0.98 ...  -0.98]
        ...
        [0.98 ... 0.98]
        [1 ... 1] 
    """

    a0 = A0[0].reshape(100, 1)

    mu = [0, 0]
    cov = [[beta, 0], [0, beta]]
    contour = []

    for i in range(0, 100):
        a1 = A1[i].reshape(100, 1)

        # construct 100 x 2 matrix
        data = np.concatenate((a0, a1), 1)
        contour.append(util.density_Gaussian(mu, cov, data))

    # plot true value of a
    plt.plot([-0.1], [-0.5], 'o')

    # plot the contours
    plt.contour(A0, A1, contour)
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.title('Prior Distribution')
    plt.savefig("prior.pdf")
    plt.show()


def posteriorDistribution(x, z, beta, sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)

    Inputs:
    ------
    x: inputs from training set
    z: targets from training set
    beta: hyperparameter in the prior distribution
    sigma2: variance of Gaussian noise

    Outputs:
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """

    # covariances
    cov_a_inv = [[1 / beta, 0], [0, 1 / beta]]

    # Given x with 1 column of 1s
    X = np.append(np.ones((len(x), 1), dtype=int), x, 1)

    # covariance of posterior
    cov = np.linalg.inv(cov_a_inv + (1 / sigma2) * np.matmul(X.T, X))

    # means of posterior
    mu = (1 / sigma2) * np.matmul(np.matmul(cov, X.T), z)
    mu = mu.reshape(2, 1).squeeze()

    # a0, a1 range is set between -1 and 1
    a0 = np.linspace(-1, 1, 100)
    a1 = np.linspace(-1, 1, 100)
    A0, A1 = np.meshgrid(a0, a1)

    """ 
    A0:  [-1, -0.98 .... 0.98, 1]
         ...
        [-1, -0.98.... 0.98, 1]

    A1:  [-1, -1 ... -1, -1]
        [-0.98 ...  -0.98]
        ...
        [0.98 ... 0.98]
        [1 ... 1] 
    """

    a0 = A0[0].reshape(100, 1)

    contour = []

    for i in range(0, 100):
        a1 = A1[i].reshape(100, 1)

        # construct 100 x 2 matrix
        data = np.concatenate((a0, a1), 1)
        contour.append(util.density_Gaussian(mu.T, cov, data))

    # plot true value of a
    plt.plot([-0.1], [-0.5], 'o')

    # plot the contours
    plt.contour(A0, A1, contour)
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.title('Posterior Distribution with ' + str(len(x)) + ' Training Samples')
    plt.savefig("posterior" + str(len(x)) + ".pdf")
    plt.show()

    return mu, cov

def predictionDistribution(x, beta, sigma2, mu, Cov, x_train, z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results

    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the prior distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot

    Outputs: None
    -----
    """

    # construct input matrix with new input data
    X = np.append(np.ones((len(x), 1), dtype=int), np.expand_dims(x, 1), 1)

    # mean
    mu_z = np.matmul(X, mu)

    # covariance
    cov_z = sigma2 + np.matmul(X, np.matmul(Cov, X.T))

    # standard deviation
    std_dev = np.sqrt(np.diag(cov_z))

    plt.xlabel('x')
    plt.ylabel('z')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])

    plt.scatter(x_train, z_train)
    plt.errorbar(x, mu_z, yerr=std_dev, color='red')
    plt.title('Prediction with ' + str(len(x_train)) + ' Training Samples')
    plt.savefig("predict" + str(len(x_train)) + ".pdf")
    plt.show()

if __name__ == '__main__':

    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction
    x_test = [x for x in np.arange(-4, 4.01, 0.2)]

    # known parameters
    sigma2 = 0.1
    beta = 1

    # number of training samples used to compute posterior
    ns = 100

    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]

    # prior distribution p(a)
    priorDistribution(beta)

    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x, z, beta, sigma2)

    # distribution of the prediction
    predictionDistribution(x_test, beta, sigma2, mu, Cov, x, z)









