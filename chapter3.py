from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from helpers import plot_decision_regions
from LogisticRegressionGD import LogisticRegressionGD
import numpy as np
import matplotlib.pyplot as plt
from pydotplus import graph_from_dot_data

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
print('Etykiety klas:', np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

print('Liczba etykiet w zbiorze y:', np.bincount(y))
print('Liczba etykiet w zbiorze y_train:', np.bincount(y_train))
print('Liczba etykiet w zbiorze y_test:', np.bincount(y_test))

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# ppn = Perceptron(eta0=0.1, random_state=1)
# ppn.fit(X_train_std, y_train)

# y_pred = ppn.predict(X_test_std)

# print('Dokładność: %.3f' % accuracy_score(y_test, y_pred))
# print('Dokładność: %.3f' % ppn.score(X_test_std, y_test))

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
# plot_decision_regions(X=X_combined_std,
#                         y=y_combined,
#                         classifier=ppn,
#                         test_idx=range(105, 150))
# plt.xlabel('Długość płatka [standaryzowana]')
# plt.ylabel('Szerokość płatka [standaryzowana]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()


# def sigmoid(z):
#     return 1.0 / (1.0 + np.exp(-z))
# z = np.arange(-7, 7, 0.1)
# phi_z = sigmoid(z)
# plt.plot(z, phi_z)
# plt.axvline(0.0, color='k') 
# plt.ylim(-0.1, 1.1)
# plt.xlabel('z')
# plt.ylabel('$\phi (z)$')
# # jednostki osi y i siatka
# plt.yticks([0.0, 0.5, 1.0]) 
# ax = plt.gca()
# ax.yaxis.grid(True)
# plt.tight_layout()
# plt.show()

# def cost_1(z):
#     return - np.log(sigmoid(z))
# def cost_0(z):
#     return - np.log(1 - sigmoid(z))
# z = np.arange(-10, 10, 0.1)
# phi_z = sigmoid(z)
# c1 = [cost_1(x) for x in z]
# plt.plot(phi_z, c1, label='J(w) jeśli y=1')
# c0 = [cost_0(x) for x in z]
# plt.plot(phi_z, c0, linestyle='--', label='J(w) jeśli y=0')
# plt.ylim(0.0, 5.1)
# plt.xlim([0, 1])
# plt.xlabel('$\phi$(z)')
# plt.ylabel('J(w)')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()

X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

# lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
# lrgd.fit(X_train_01_subset, y_train_01_subset)
# plot_decision_regions(X=X_train_01_subset,y=y_train_01_subset, classifier=lrgd)
# plt.xlabel('Długość płatka [standaryzowana]')
# plt.ylabel('Szerokość płatka [standaryzowana]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

# lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
# lr.fit(X_train_std, y_train)
# plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# # plt.savefig('images/03_06.png', dpi=300)
# plt.show()

# print(lr.predict_proba(X_test_std[:3, :]).argmax(axis=1))

# svm = SVC(kernel='linear', C=1.0, random_state=1)
# svm.fit(X_train_std, y_train)
# plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
# plt.xlabel('Długość płatka [standaryzowana]')
# plt.ylabel('Szerokość płatka [standaryzowana]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

# plt.scatter(X_xor[y_xor == 1, 0],
#             X_xor[y_xor == 1, 1],
#             c='b', marker='x',
#             label='1')
# plt.scatter(X_xor[y_xor == -1, 0],
#             X_xor[y_xor == -1, 1],
#             c='r',
#             marker='s',
#             label='-1')

# plt.xlim([-3, 3])
# plt.ylim([-3, 3])
# plt.legend(loc='best')
# plt.tight_layout()
# #plt.savefig('images/03_12.png', dpi=300)
# plt.show()

# svm = SVC(kernel='rbf', random_state=1, gamma=0.5, C=10.0)
# svm.fit(X_xor, y_xor)
# plot_decision_regions(X_xor, y_xor,
#                       classifier=svm)

# plt.legend(loc='upper left')
# plt.tight_layout()
# #plt.savefig('images/03_14.png', dpi=300)
# plt.show()


# svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
# svm.fit(X_train_std, y_train)

# plot_decision_regions(X_combined_std, y_combined,
#                       classifier=svm, test_idx=range(105, 150))
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# #plt.savefig('images/03_15.png', dpi=300)
# plt.show()

# svm = SVC(kernel='rbf', random_state=1, gamma=10, C=1.0)
# svm.fit(X_train_std, y_train)

# plot_decision_regions(X_combined_std, y_combined,
#                       classifier=svm, test_idx=range(105, 150))
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# #plt.savefig('images/03_15.png', dpi=300)
# plt.show()

# tree_model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=1)
# tree_model.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
# plot_decision_regions(X_combined, y_combined, 
#                       classifier=tree_model,
#                       test_idx=range(105, 150))
# plt.xlabel('petal length [cm]')
# plt.ylabel('petal width [cm]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# #plt.savefig('images/03_20.png', dpi=300)
# plt.show()

# tree.plot_tree(tree_model)
# #plt.savefig('images/03_21_1.pdf')
# plt.show()

# forest = RandomForestClassifier(criterion='entropy',
#                                 n_estimators=25, 
#                                 random_state=1,
#                                 n_jobs=2)
# forest.fit(X_train, y_train)

# plot_decision_regions(X_combined, y_combined, 
#                       classifier=forest, test_idx=range(105, 150))

# plt.xlabel('petal length [cm]')
# plt.ylabel('petal width [cm]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# #plt.savefig('images/03_22.png', dpi=300)
# plt.show()

# knn = KNeighborsClassifier(n_neighbors=3, 
#                            p=2, 
#                            metric='minkowski')
# knn.fit(X_xor, y_xor)

# plot_decision_regions(X_xor, y_xor, 
#                       classifier=knn, test_idx=range(105, 150))

# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# #plt.savefig('images/03_24.png', dpi=300)
# plt.show()
