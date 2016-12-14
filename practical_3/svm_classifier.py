import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import svm


def linear_classifier(name):
    # TODO ADD NAMES CHANGES
    print("========== %s ===========" %name)
    data = np.load("tsne_data/%s_tnse.dat"%name)
    labels = np.load("tsne_data/%s_labels.dat"%name)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    train_acc = 0
    test_acc = 0
    for i in range(len(classes)):
        random_indexes = np.random.choice(9000, 1000, replace=False)

        one_x_train = data[labels == i]

        rest_x_train = data[labels != i][random_indexes]

        true_classes = np.ones(len(one_x_train), dtype=int)
        false_classes = np.zeros(len(rest_x_train), dtype=int)

        data_points = np.concatenate((one_x_train, rest_x_train), axis=0)

        label = np.concatenate((true_classes, false_classes), axis=0)

        lin_clf = svm.LinearSVC()
        lin_clf.fit(data_points, label)
        predictions = lin_clf.predict(data_points)
        acc = sum(1 for x, y in zip(predictions, label) if x == y) / float(len(label))
        train_acc += acc
        print("SVM predicts for %s a train accuracy %f" % (classes[i], acc))

    train_acc = train_acc/float(len(classes))

    print("Average train accuracy: %f "%(train_acc))

if __name__ == '__main__':
    linear_classifier("conv_fcl2")
    linear_classifier("conv_fcl1")
    linear_classifier("conv_flatten")
