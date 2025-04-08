import numpy as np

# Composite Simpson's Rule
def simpsons_rule(f, a, b, n):
    if n % 2 == 1:
        raise ValueError("n must be even for Simpson's rule")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    integral = (h / 3) * (y[0] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n-1:2]) + y[n])
    return integral

# Transformed integrand for part (a)
# Original f(x) = x^(-1/4) * sin(x), with x = 1/t, dx = -1/t^2 dt
def transformed_f1(t):
    x = 1 / t
    return (x**(-1/4) * np.sin(x)) * (1 / t**2)

# Transformed integrand for part (b)
# Original f(x) = x^(-4) * sin(x), with x = 1/t, dx = -1/t^2 dt
# So after transformation: ∫₀¹ t^2 * sin(1/t) dt
def transformed_f2(t):
    result = np.zeros_like(t)
    nonzero = t != 0
    result[nonzero] = t[nonzero]**2 * np.sin(1 / t[nonzero])
    result[~nonzero] = 0
    return result

# Number of intervals
n = 4

# Integration bounds
# For (a): ∫₁^∞ ...  → Use [1, 100]
# For (b): ∫₀^1  → Use [0.01, 1] to avoid t=0 singularity
I1 = simpsons_rule(transformed_f1, 1, 100, n)
I2 = simpsons_rule(transformed_f2, 0.01, 1, n)

print(f"(a) ∫₀¹ x^(-1/4) sin(x) dx ≈ {I1:.8f}")
print(f"(b) ∫₁^∞ x^(-4) sin(x) dx ≈ {I2:.8f}")
