import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import dblquad

# 被積函數
def f(x, y):
    return 2 * y * np.sin(x) + np.cos(x)**2

# a. Composite Simpson's rule
def simpsons_double_integral(f, x_range, n, m):
    a, b = x_range
    hx = (b - a) / n
    total = 0
    for i in range(n + 1):
        x = a + i * hx
        wx = 1 if i == 0 or i == n else (4 if i % 2 == 1 else 2)

        y1 = np.sin(x)
        y2 = np.cos(x)
        hy = (y2 - y1) / m

        inner_sum = 0
        for j in range(m + 1):
            y = y1 + j * hy
            wy = 1 if j == 0 or j == m else (4 if j % 2 == 1 else 2)
            inner_sum += wy * f(x, y)

        total += wx * (hy / 3) * inner_sum

    return (hx / 3) * total

# b. Gaussian Quadrature for double integral
def gaussian_double_integral(f, x_range, n, m):
    a, b = x_range
    x_nodes, x_weights = leggauss(n)
    sum_result = 0

    for i in range(n):
        # 對 x 做變換
        x = 0.5 * (b - a) * x_nodes[i] + 0.5 * (b + a)
        wx = x_weights[i]

        y1 = np.sin(x)
        y2 = np.cos(x)

        y_nodes, y_weights = leggauss(m)
        inner_sum = 0
        for j in range(m):
            # 對 y 做變換
            y = 0.5 * (y2 - y1) * y_nodes[j] + 0.5 * (y1 + y2)
            wy = y_weights[j]
            inner_sum += wy * f(x, y)

        sum_result += wx * (0.5 * (y2 - y1)) * inner_sum

    return 0.5 * (b - a) * sum_result

# 計算值
a = 0
b = np.pi / 4
I_simpson = simpsons_double_integral(f, (a, b), n=4, m=4)
I_gauss = gaussian_double_integral(f, (a, b), n=3, m=3)

# Exact value using SciPy
I_exact, _ = dblquad(f, a, b, lambda x: np.sin(x), lambda x: np.cos(x))

# 輸出
print(f"Simpson's Rule: {I_simpson:.8f}")
print(f"Gaussian Quadrature: {I_gauss:.8f}")
print(f"Exact value (dblquad): {I_exact:.8f}")
print(f"誤差 (Simpson): {abs(I_simpson - I_exact):.2e}")
print(f"誤差 (Gaussian): {abs(I_gauss - I_exact):.2e}")
