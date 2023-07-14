from libsvm.svmutil import *
from Utils.load_mnist import *
from Utils.PCA import *
from Utils.MDA import *
import matplotlib.pyplot as plt

def train(p, x, y):
    prob = svm_problem(y[:10000], x[:10000])
    m = svm_train(prob, p)
    return m

def predict(model, x,y):
    p_label, p_acc, p_val = svm_predict(y[:1000], x[:1000], model)
    return p_acc

x_train, x_test = get_data()
y_train, y_test = get_labels()

ch = int(input('Enter the choice 1.PCA 2.MDA'))

if ch == 1:
    x_train_reduced, eig_vecs = PCA(x_train)
    x_test_reduced = np.real(np.matmul(x_test, eig_vecs))
else:
    x_train_reduced, t_matrix = MDA(x_train, y_train)
    x_test_reduced = np.real((np.matmul(t_matrix.T, x_test.T)).T)
    print('MDA done on train data')


choice = int(input('Enter the choice of kernel 1:linear 2:polynomial 3:RBF'))
accuracies = []
if choice == 1:
    param = svm_parameter('-q -t 0')
    model = train(param, x_train_reduced, y_train)
    predict(model, x_test_reduced, y_test)

elif choice == 2:
    degree = [' -d 2']
    for i in degree:
        param = svm_parameter('-q -t 1 -g 0.001 -r 1' + i)
        model = train(param, x_train_reduced, y_train)
        acc = predict(model, x_test_reduced, y_test)
        accuracies.append(acc[0])
    plt.plot(degree, accuracies)
    plt.show()
    plt.savefig('polynomial.png')
        
else:
    gamma = [' -g 1000']
    for i in gamma:
        param = svm_parameter('-q -t 2' + i)
        model = train(param, x_train_reduced, y_train)
        acc = predict(model, x_test_reduced, y_test)
        accuracies.append(acc[0])
    print(accuracies)
    plt.plot(gamma, accuracies)
    plt.show()
    plt.savefig('rbf.png')

