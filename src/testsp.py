import sympy as sp
from sympy.parsing.latex import parse_latex

s = r"\int \frac {2+ \sin x}{ \sin x (1+ \cos x)} \, dx"
f = parse_latex(s)

# Extract the function and the limits of integration
integrand = f.args[0]
limits = f.args[1]
print(integrand,limits)

# Perform the integration
integral = sp.integrate(integrand, limits)
print(f"Integral: {integral}")
