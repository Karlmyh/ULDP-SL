from distribution import TestDistribution
from ULDPFS import ULDPFS

def test_distribution():
    sample_generator = TestDistribution(1, 100).returnDistribution()


    X, y = sample_generator.generate(100)
    print(X.shape)
    print(y.shape)
    print(X)
    print(y)
    print(sample_generator.beta)
    print(sample_generator.nonzero_index)


    m = 150
    n = 500
    X_y_list = [sample_generator.generate(m) for _ in range(n)]
    X_list = [X_y[0] for X_y in X_y_list]
    y_list = [X_y[1] for X_y in X_y_list]

    clf = ULDPFS(epsilon = 8,
                 selector = "postlasso",
                 ).fit(X_list, y_list)
    print(clf.selected_indexes)
    print(clf.coef_)