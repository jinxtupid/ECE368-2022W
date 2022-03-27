import os.path
import numpy as np
import matplotlib.pyplot as plt
import util


def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """

    ham_dict = util.get_word_freq(file_lists_by_category[1])
    spam_dict = util.get_word_freq(file_lists_by_category[0])
    total_dic = util.get_word_freq(file_lists_by_category[1] + file_lists_by_category[0])

    total_number_ham = sum(ham_dict.values())
    total_number_spam = sum(spam_dict.values())

    total_number_distinct = len(total_dic)

    for items in ham_dict:
        ham_dict[items] = (ham_dict[items] + 1) / (total_number_ham + total_number_distinct)

    for items in spam_dict:
        spam_dict[items] = (spam_dict[items] + 1) / (total_number_spam + total_number_distinct)

    # pass the frequency of the word inside the vocabulary but doesn't exist in another class
    smooth_ham = 1 / (total_number_ham + total_number_distinct)
    smooth_spam = 1 / (total_number_spam + total_number_distinct)

    return (spam_dict, ham_dict), smooth_spam, smooth_ham


def classify_new_email(filename, probabilities_by_category, prior_by_category, smooth_spam, smooth_ham, ratio):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """

    filename_list = (filename, 1)
    words_freq_dict = util.get_word_freq(filename_list[0:1])

    log_multiplication_of_prob_1 = 0
    for word in words_freq_dict:
        if word in probabilities_by_category[0]:
            log_multiplication_of_prob_1 += np.log(probabilities_by_category[0][word]) * words_freq_dict[word]
        elif word in probabilities_by_category[1]:
            log_multiplication_of_prob_1 += np.log(smooth_spam) * words_freq_dict[word]

    log_multiplication_of_prob_0 = 0
    for word in words_freq_dict:
        if word in probabilities_by_category[1]:
            log_multiplication_of_prob_0 += np.log(probabilities_by_category[1][word]) * words_freq_dict[word]
        elif word in probabilities_by_category[0]:
            log_multiplication_of_prob_0 += np.log(smooth_ham) * words_freq_dict[word]

    prediction_1 = log_multiplication_of_prob_1 + np.log(prior_by_category[0])
    prediction_0 = log_multiplication_of_prob_0 + np.log(prior_by_category[1])

    if prediction_1 - np.log(ratio) < prediction_0:
        result = 'ham'
    else:
        result = 'spam'

    classify_result = (result, (prediction_1, prediction_0))

    return classify_result


if __name__ == '__main__':

    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))

    # Learn the distributions    
    probabilities_by_category, smooth_spam, smooth_ham = learn_distributions(file_lists)

    # prior class distribution
    priors_by_category = [0.5, 0.5]

    # Store the classification results
    performance_measures = np.zeros([2, 2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    # for filename in (util.get_files_in_folder(test_folder)):
    #     # Classify
    #     label, log_posterior = classify_new_email(filename,
    #                                               probabilities_by_category,
    #                                               priors_by_category,
    #                                               smooth_spam, smooth_ham, ratio=1)
    #
    #     # Measure performance (the filename indicates the true label)
    #     base = os.path.basename(filename)
    #     true_index = ('ham' in base)
    #     guessed_index = (label == 'ham')
    #     performance_measures[int(true_index), int(guessed_index)] += 1
    #
    # template = "You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # # Correct counts are on the diagonal
    # correct = np.diag(performance_measures)
    # # totals are obtained by summing across guessed labels
    # totals = np.sum(performance_measures, 1)
    # print(template % (correct[0], totals[0], correct[1], totals[1]))

    '''Modified code for to plot tradeoffs '''
    error1 = []
    error2 = []
    # TODO: Write your code here to modify the decision rule such that

    # Type 1 and Type 2 errors can be traded off, plot the trade-off curve

    for ratio in [1e-3, 1e-2, 1, 10, 50, 100, 1e10, 1e15, 1e20, 1e25]:
        performance_measures = np.zeros([2, 2])
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label, log_posterior = classify_new_email(filename,
                                                      probabilities_by_category,
                                                      priors_by_category, smooth_spam, smooth_ham, ratio)

            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base)
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        template = "You correctly classified %d out of %d spam emails, and %d out of %d ham emails."

        # Correct counts are on the diagonal
        correct = np.diag(performance_measures)

        # totals are obtained by summing across guessed labels
        totals = np.sum(performance_measures, 1)
        print(template % (correct[0], totals[0], correct[1], totals[1]))
        error1.append(totals[0] - correct[0])
        error2.append(totals[1] - correct[1])

    plt.plot(error1, error2)
    plt.xlabel('number of type 1 errors')
    plt.ylabel('number of type 2 errors')
    plt.title('trade off of error type 1 and error type 2')
    plt.savefig("nbc.pdf")
    plt.show()
