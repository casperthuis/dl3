from cifar10_siamese_utils import get_cifar10 as get_cifar_10_siamese
import time
cifar10 = get_cifar_10_siamese('cifar10/cifar-10-batches-py')
start = time.time()
x1, x2, y = cifar10.train.next_batch(200, fraction_same=0.35)
print time.time() - start