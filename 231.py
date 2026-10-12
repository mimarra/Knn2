import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 1000
X = 10 * np.random.rand(n, 1)
Y = 2 * X + np.sin(X) + np.random.randn(n, 1)

plt.figure()
plt.scatter(X, Y, 15, facecolors='none', edgecolors='b')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Початковий набір даних')
plt.grid(True)
plt.show()

X_norm = (X - X.min()) / (X.max() - X.min())
Y_norm = (Y - Y.min()) / (Y.max() - Y.min())

print(f"X після нормалізації: {X_norm.min():.2f} - {X_norm.max():.2f}")
print(f"Y після нормалізації: {Y_norm.min():.2f} - {Y_norm.max():.2f}")

indices = np.arange(n)
np.random.shuffle(indices)
split = int(n * 0.8)
train_idx, test_idx = indices[:split], indices[split:]

X_train, Y_train = X_norm[train_idx], Y_norm[train_idx]
X_test, Y_test = X_norm[test_idx], Y_norm[test_idx]

print(f"Навчальна вибірка: {len(X_train)} значень")
print(f"Тестова вибірка: {len(X_test)} значень")

plt.figure()
plt.scatter(X_train, Y_train, 15, 'b', label='Навчальна вибірка')
plt.scatter(X_test, Y_test, 15, 'r', label='Тестова вибірка')
plt.xlabel('Нормалізований X')
plt.ylabel('Нормалізований Y')
plt.title('Поділ даних на навчальну та тестову вибірки')
plt.legend()
plt.grid(True)
plt.show()

K_values = [1, 3, 5, 7, 9, 11, 15, 20]
MSE_values = []

for K in K_values:
    Y_pred = np.zeros(len(X_test))
    for j in range(len(X_test)):
        distances = np.abs(X_train - X_test[j]).ravel()
        idx = np.argsort(distances)
        neighbors = Y_train[idx[:K]]
        Y_pred[j] = np.mean(neighbors)

    mse = np.mean((Y_test.ravel() - Y_pred) ** 2)
    MSE_values.append(mse)
    print(f"K = {K}, MSE = {mse:.6f}")

plt.figure()
plt.plot(K_values, MSE_values, '-o', linewidth=2, markersize=7)
plt.xlabel('Кількість сусідів K')
plt.ylabel('MSE')
plt.title('Залежність похибки MSE від K')
plt.grid(True)
plt.show()

best_idx = np.argmin(MSE_values)
bestK = K_values[best_idx]
minMSE = MSE_values[best_idx]

print(f"\nНайкраще K = {bestK}")
print(f"Мінімальна MSE = {minMSE:.6f}")

Y_best = np.zeros(len(X_test))
for j in range(len(X_test)):
    distances = np.abs(X_train - X_test[j]).ravel()
    idx = np.argsort(distances)
    neighbors = Y_train[idx[:bestK]]
    Y_best[j] = np.mean(neighbors)

plt.figure()
plt.scatter(X_test, Y_test, 20, 'b', label='Реальні тестові значення')
sort_idx = np.argsort(X_test.ravel())
plt.plot(X_test[sort_idx], Y_best[sort_idx], 'r-', linewidth=2, label='Прогноз KNN')
plt.xlabel('Нормалізований X')
plt.ylabel('Нормалізований Y')
plt.title(f'KNN-регресія при K = {bestK}')
plt.legend()
plt.grid(True)
plt.show()

np.savez('lab2_results.npz', X=X, Y=Y, X_train=X_train, Y_train=Y_train,
         X_test=X_test, Y_test=Y_test, K_values=K_values, MSE_values=MSE_values,
         bestK=bestK, minMSE=minMSE, Y_best=Y_best)
print("\nРезультати збережено у файл lab2_results.npz")
