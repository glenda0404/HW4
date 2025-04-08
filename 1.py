import numpy as np

# 定義被積分函數
def f(x):
    return np.exp(x) * np.sin(4 * x)

# 積分區間與步長
a = 1
b = 2
h = 0.1
n = int((b - a) / h)

# a. Composite Trapezoidal Rule
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * (0.5 * y[0] + np.sum(y[1:n]) + 0.5 * y[n])

# b. Composite Simpson's Rule
def simpsons_rule(f, a, b, n):
    if n % 2 == 1:
        raise ValueError("n 必須是偶數才可使用 Simpson's rule")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (h / 3) * (y[0] + 2 * np.sum(y[2:n:2]) + 4 * np.sum(y[1:n:2]) + y[n])

# c. Composite Midpoint Rule
def midpoint_rule(f, a, b, n):
    h = (b - a) / n
    midpoints = a + h * (np.arange(n) + 0.5)
    return h * np.sum(f(midpoints))

# 計算
I_trap = trapezoidal_rule(f, a, b, n)
I_simp = simpsons_rule(f, a, b, n)
I_mid  = midpoint_rule(f, a, b, n)

print(f"Composite Trapezoidal Rule: {I_trap:.8f}")
print(f"Composite Simpson's Rule:   {I_simp:.8f}")
print(f"Composite Midpoint Rule:    {I_mid:.8f}")
