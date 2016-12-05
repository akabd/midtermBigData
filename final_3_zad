import main
import util
import matplotlib.pyplot as plt
import sklearn.feature_extraction.text
import sklearn.datasets
import sklearn.naive_bayes
import sklearn.svm
import sklearn.cross_validation
import numpy
import sklearn.neighbors


def _main():
    path = 'dataset'
    NB(path)
def SVM(path):
    print "Classifier: Support Vector Machine"
    print "Train-Test Split"

    # preprocess
    main.reorganize_dataset(path)
    main.remove_incompatible_files(path)

    # load data
    files = sklearn.datasets.load_files(path, shuffle=True)

    # refine emails - delete unwanted text form them
    util.refine_all_emails(files.data)

    # feature Extractoin
    # BOW
    BOW = util.bagOfWords(files.data)
    # TF
    tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=False).fit(BOW)
    TF = tf_transformer.transform(BOW)
    # TFIDF
    tfidf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=False).fit(BOW)
    TFIDF = tfidf_transformer.transform(BOW)

    # build classifier
    clf = sklearn.svm.LinearSVC()

    # calculate results
    i, BOW_results = split_test_classifier(clf, BOW, files.target)
    i, TF_results = split_test_classifier(clf, TF, files.target)
    i, TFIDF_results = split_test_classifier(clf, TFIDF, files.target)

    # plot
    plot_results(i, [BOW_results, TF_results, TFIDF_results], ['BOW', 'TF', 'TFIDF'])
    def NB(path):
    print "Classifier: Naive Bayes"
    print "Train-Test Split"

    # preprocess
    main.reorganize_dataset(path)
    main.remove_incompatible_files(path)

    # load data
    files = sklearn.datasets.load_files(path, shuffle=True)

    # refine emails - delete unwanted text form them
    util.refine_all_emails(files.data)

    # feature Extractoin
    # BOW
    BOW = util.bagOfWords(files.data)
    # TF
    tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=False).fit(BOW)
    TF = tf_transformer.transform(BOW)
    # TFIDF
    tfidf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=False).fit(BOW)
    TFIDF = tfidf_transformer.transform(BOW)

    # build classifier
    clf = sklearn.naive_bayes.MultinomialNB()

    # calculate results
    i, BOW_results = split_test_classifier(clf, BOW, files.target)
    i, TF_results = split_test_classifier(clf, TF, files.target)
    i, TFIDF_results = split_test_classifier(clf, TFIDF, files.target)

    # plot
    plot_results(i, [BOW_results, TF_results, TFIDF_results], ['BOW', 'TF', 'TFIDF'])
    def KFOLD_KNN_parameter_test(X, y, n_cross_val=5, n_neighbors=5):
    weights1 = 'uniform'
    weights2 = 'distance'

    results_1 = []
    results_2 = []
    i = []

    for n_neighbors in range(2, 21):
        print 'number of neighbors:', n_neighbors
        # build two classifiers
        clf1 = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weights1)
        clf2 = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weights2)

        scores1 = util.cross_validation(X, y, clf1, cv=n_cross_val)
        scores2 = util.cross_validation(X, y, clf2, cv=n_cross_val)

        i.append(n_neighbors)
        results_1.append(scores1.mean())
        results_2.append(scores2.mean())

    return i, results_1, results_2


def plot_results(i, results_list, labels_list):
    colors_list = ['red', 'blue', 'black', 'green', 'cyan', 'yellow']

    if not len(results_list) == len(labels_list):
        print 'un equal len in results and labels'
        raise Exception

    for (result, label, color) in zip(results_list, labels_list, colors_list):
        plt.plot(i, result, color=color, lw=2.0, label=label)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    _main()
