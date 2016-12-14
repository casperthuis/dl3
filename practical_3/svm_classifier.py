import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import svm


def linear_classifier(name):
    # TODO ADD NAMES CHANGES
    print("========== %s ===========" %name)
    data = np.load("tsne_data/%s_tsne.dat"%name)
    labels = np.load("tsne_data/%s_labels.dat"%name)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    train_acc = 0
    test_acc = 0
    for i in range(len(classes)):
        random_indexes = np.random.choice(9000, 1000, replace=False)
        random_indexes_train = random_indexes[0:800]
        random_indexes_test = random_indexes[800:]
        one_x_train = data[labels == i][0:800]
        one_x_test = data[labels == i][800:]
        rest_x_train = data[labels != i][random_indexes_train]
        rest_x_test = data[labels != i][random_indexes_test]

        # print(one_x_train)
        # print(rest_x_train)
        true_classes = np.ones(len(one_x_train), dtype=int)
        false_classes = np.zeros(len(rest_x_train), dtype=int)
        #print(true_classes)
        #print(false_classes)
        x = np.concatenate((one_x_train, rest_x_train), axis=0)
        #print(x)
        label = np.concatenate((true_classes, false_classes), axis=0)
        #print(y)
        lin_clf = svm.LinearSVC()
        lin_clf.fit(x, label)
        predictions = lin_clf.predict(x)
        acc = sum(1 for x, y in zip(predictions, label) if x == y) / float(len(label))
        train_acc += acc
        print("SVM predicts for %s a train accuracy %f" % (classes[i], acc))
        x = np.concatenate((one_x_test, rest_x_test), axis=0)
        true_classes = np.ones(len(one_x_test), dtype=int)
        false_classes = np.zeros(len(rest_x_test), dtype=int)
        label = np.concatenate((true_classes, false_classes), axis=0)
        predictions = lin_clf.predict(x)
        acc = sum(1 for x, y in zip(predictions, label) if x == y) / float(len(label))
        test_acc += acc
        print("SVM predicts for %s a test accuracy %f" % (classes[i], acc))

        # rest_x = data[labels != i][np.random.choice((len(data)-len(one_x)), len(one_x), replace=False)]

        # rest_x_i = np.random.choice(len(rest_x), len(one_x), replace=False)
        # x_total = np.concatenate((one_x, rest_x), axis=0)
        # y1 = np.zeros(len(x_total), dtype=int)
        # y1[0:len(one_x)] = 1
        #
        # lin_clf = svm.LinearSVC()
        # lin_clf.fit(x_total, y1)
        # pred_y = lin_clf.predict(data)
        # acc = sum(1 for x, y in zip(pred_y, y1) if x == y) / float(len(y1))
        #print("SVM predicts for %s a accuracy %f" %(classes[i], acc))
    train_acc = train_acc/float(len(classes))
    test_acc = test_acc/float(len(classes))
    print("Average train accuracy: %f , test accuracy: %f " %(train_acc, test_acc))

if __name__ == '__main__':
    linear_classifier("conv_fcl2")
    linear_classifier("conv_fcl1")
    linear_classifier("conv_flatten")
    linear_classifier("siamese_fcl2")