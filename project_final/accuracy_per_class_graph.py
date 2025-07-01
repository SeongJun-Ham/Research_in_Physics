import matplotlib.pyplot as plt
import numpy as np

c1 = [4290, 15399, 480, 673]
c2 = [5035, 15326, 453, 668]
c3 = [4858, 15915, 411, 662]
tot = [8458, 17048, 612, 720]

c1.append(sum(c1))
c2.append(sum(c2))
c3.append(sum(c3))
tot.append(sum(tot))


y1 = np.array(c1)/np.array(tot)
y2 = np.array(c2)/np.array(tot)
y3 = np.array(c3)/np.array(tot)



X_label = ['class 1', 'class 2', 'class 3', 'class 4', 'Total']

plt.plot(np.array(X_label), y1, 'ro-', label = '(b)')
plt.plot(np.array(X_label), y2, 'go-', label = '(c)')
plt.plot(np.array(X_label), y3, 'bo-', label = '(d)')
plt.legend()
plt.ylabel('Accuracy per class')

plt.show()