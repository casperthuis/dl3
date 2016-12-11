import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import svm


def linear_classifier(name):
    # TODO ADD NAMES CHANGES
    print("========== %s ===========" %name)
    data = np.load("tsne_data/%s_pca.dat"%name)
    labels = np.load("tsne_data/%s_labels.dat"%name)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        one_x = data[labels == i]
        rest_x = data[labels != i][np.random.choice((len(data)-len(one_x)), len(one_x), replace=False)]

        # rest_x_i = np.random.choice(len(rest_x), len(one_x), replace=False)
        x_total = np.concatenate((one_x, rest_x), axis=0)
        y1 = np.zeros(len(x_total), dtype=int)
        y1[0:len(one_x)] = 1


        lin_clf = svm.LinearSVC()
        lin_clf.fit(x_total, y1)
        pred_y = lin_clf.predict(data)
        acc = sum(1 for x, y in zip(pred_y, y1) if x == y) / float(len(y1))
        print("SVM predicts for %s a accuracy %f" %(classes[i], acc))


if __name__ == '__main__':
    linear_classifier("tsne_fcl2")
    linear_classifier("tsne_fcl1")
    linear_classifier("tsne_flatten")