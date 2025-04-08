import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy.integrate as spi

# 被積函數
def f(x):
    return x**2 * np.log(x)

# 變數變換後的函數：x = (b-a)/2 * t + (a+b)/2
def gaussian_integrate(f, a, b, n):
    [nodes, weights] = leggauss(n)
    transformed_x = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    fx = f(transformed_x)
    return 0.5 * (b - a) * np.sum(weights * fx)

# 設定區間
a, b = 1, 1.5

# 計算 Gaussian Quadrature
I_gauss3 = gaussian_integrate(f, a, b, n=3)
I_gauss4 = gaussian_integrate(f, a, b, n=4)

# 用 scipy 計算 exact value
I_exact, _ = spi.quad(f, a, b)

print(f"Gaussian Quadrature n=3: {I_gauss3:.8f}")
print(f"Gaussian Quadrature n=4: {I_gauss4:.8f}")
print(f"Exact value (scipy.quad): {I_exact:.8f}")
print(f"誤差 (n=3): {abs(I_gauss3 - I_exact):.2e}")
print(f"誤差 (n=4): {abs(I_gauss4 - I_exact):.2e}")
