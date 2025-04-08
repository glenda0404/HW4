import numpy as np
import warnings
warnings.filterwarnings("ignore")  # 關閉所有警告訊息

eps = 1e-6  # 避開 x=0 與 t=0 的端點問題

# (a)
def f1(x):
    return x**(-0.25) * np.sin(x)

# (b)
def f2(t):
    return t**2 * np.sin(1 / t)

def simpson(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("n 必須是偶數")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (h / 3) * (y[0] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n-1:2]) + y[n])

I_a = simpson(f1, eps, 1, 4)
I_b = simpson(f2, eps, 1, 4)

print(f"(a) ∫₀¹ x^(-1/4) sin(x) dx ≈ {I_a:.8f}")
print(f"(b) ∫₁^∞ x^(-4) sin(x) dx ≈ {I_b:.8f}")
