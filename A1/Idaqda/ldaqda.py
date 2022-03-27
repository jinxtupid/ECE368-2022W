import numpy as np
import matplotlib.pyplot as plt
import util


def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models

    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples

    y: a N-by-1 1D array contains the labels of the N samples

    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA
    and 1 figure for QDA in this function
    """

    num_male, num_female = 0, 0
    male_heights, male_weights, female_heights, female_weights = 0, 0, 0, 0
    male_height, male_weight, female_height, female_weight = [], [], [], []

    # get heights and weights
    for item in range(0, len(y)):
        if y[item] == 1:
            num_male += 1

            male_heights += x[item][0]
            male_height.append(x[item][0])

            male_weights += x[item][1]
            male_weight.append(x[item][1])
        else:
            num_female = num_female + 1

            female_heights += x[item][0]
            female_height.append(x[item][0])

            female_weights += x[item][1]
            female_weight.append(x[item][1])

    # calculate means for male, female and everyone
    mu_male = [male_heights / num_male, male_weights / num_male]
    mu_female = [female_heights / num_female, female_weights / num_female]

    num_total = len(y)
    mu_height = (male_heights + female_heights) / num_total
    mu_weight = (male_weights + female_weights) / num_total

    # cov for females and males separately
    cov_male_0_0, cov_male_0_1, cov_male_1_1, cov_female_0_0, cov_female_0_1, cov_female_1_1 = 0, 0, 0, 0, 0, 0

    # calculate cov for male and female
    for item in range(0, len(y)):
        if y[item] == 1:
            cov_male_0_0 += (x[item][0] - mu_male[0]) ** 2
            cov_male_0_1 += (x[item][0] - mu_male[0]) * (x[item][1] - mu_male[1])
            cov_male_1_1 += (x[item][1] - mu_male[1]) ** 2
        else:
            cov_female_0_0 += (x[item][0] - mu_female[0]) ** 2
            cov_female_0_1 += (x[item][0] - mu_female[0]) * (x[item][1] - mu_female[1])
            cov_female_1_1 += (x[item][1] - mu_female[1]) ** 2

    cov_male = [[cov_male_0_0 / num_male, cov_male_0_1 / num_male], [cov_male_0_1 / num_male, cov_male_1_1 / num_male]]
    cov_female = [[cov_female_0_0 / num_female, cov_female_0_1 / num_female],
                  [cov_female_0_1 / num_female, cov_female_1_1 / num_female]]

    # initialization
    cov_0_0, cov_0_1, cov_1_1 = 0, 0, 0

    # calculate cov for both genders
    for item in range(0, len(y)):
        cov_0_0 += (x[item][0] - mu_height) ** 2
        cov_0_1 += (x[item][0] - mu_height) * (x[item][1] - mu_weight)
        cov_1_1 += (x[item][1] - mu_weight) ** 2

    cov = [[cov_0_0 / num_total, cov_0_1 / num_total], [cov_0_1 / num_total, cov_1_1 / num_total]]


    # compute plotting data
    x_grid = np.linspace(50, 80, 100)
    y_grid = np.linspace(80, 280, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    male_lda, female_lda, male_qda, female_qda = [], [], [], []
    x_prime = X[0].reshape(100, 1)

    for i in range(0, 100):
        samples = np.concatenate((x_prime, Y[i].reshape(100, 1)), 1)
        male_lda.append(util.density_Gaussian(mu_male, cov, samples))
        female_lda.append(util.density_Gaussian(mu_female, cov, samples))
        male_qda.append(util.density_Gaussian(mu_male, cov_male, samples))
        female_qda.append(util.density_Gaussian(mu_female, cov_female, samples))

    # plot lda
    plt.scatter(male_height, male_weight, color='blue')
    plt.scatter(female_height, female_weight, color='red')

    # plot contours
    plt.contour(X, Y, male_lda, colors='b')
    plt.contour(X, Y, female_lda, colors='r')

    # plot decision boundary
    boundary_lda = np.asarray(male_lda) - np.asarray(female_lda)
    plt.contour(X, Y, boundary_lda, 0)
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.title('lda')
    plt.savefig("lda.pdf")
    plt.show()

    # # plot qda
    plt.scatter(male_height, male_weight, color='blue')
    plt.scatter(female_height, female_weight, color='red')

    # plot contours
    plt.contour(X, Y, male_qda, colors='b')
    plt.contour(X, Y, female_qda, colors='r')

    # plot decision boundary
    boundary_qda = np.asarray(male_qda) - np.asarray(female_qda)
    plt.contour(X, Y, boundary_qda, 0)
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.title('qda')
    plt.savefig("qda.pdf")
    plt.show()

    return np.asarray(mu_male), np.asarray(mu_female), np.asarray(cov), np.asarray(cov_male), np.asarray(cov_female)


def misRate(mu_male, mu_female, cov, cov_male, cov_female, x, y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate

    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis

    x: a N-by-2 2D array contains the height/weight data of the N samples

    y: a N-by-1 1D array contains the labels of the N samples

    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    # get mis_lda
    lda_male = compute_lda(mu_male, cov, x)
    lda_female = compute_lda(mu_female, cov, x)
    diff = lda_male - lda_female

    diff[diff > 0] = 1
    diff[diff < 0] = 2
    mis_lda = np.sum(diff != y)/x.shape[0]
    print(mis_lda)

    # get mis_qda
    qda_male = compute_qda(mu_male, cov_male, x)
    qda_female = compute_qda(mu_female, cov_female, x)
    diff = qda_male - qda_female

    diff[diff > 0] = 1
    diff[diff < 0] = 2

    mis_qda = np.sum(diff.transpose() != y) / x.shape[0]
    print(mis_qda)

    return mis_lda, mis_qda


def compute_qda(mu, cov, x):
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)

    qda = np.zeros((x.shape[0], 1))
    for i in range(0, x.shape[0]):
        left = -0.5 * np.log(cov_det)

        temp = np.subtract(x[i], mu)
        right = np.matmul(temp.transpose(), cov_inv)
        right = -0.5 * np.matmul(right, temp)

        qda[i] = right + left
    return qda


def compute_lda(mu, cov, x):
    cov_inv = np.linalg.inv(cov)

    right = np.transpose(np.matmul(cov_inv, mu))
    right = np.matmul(right, np.transpose(x))

    left = np.matmul(np.transpose(mu), cov_inv)
    left = 0.5 * np.matmul(left, mu)

    return right - left

if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male, mu_female, cov, cov_male, cov_female = discrimAnalysis(x_train, y_train)
    
    # misclassification rate computation
    mis_LDA, mis_QDA = misRate(mu_male, mu_female, cov, cov_male, cov_female, x_test, y_test)
    

    
    
    

    
