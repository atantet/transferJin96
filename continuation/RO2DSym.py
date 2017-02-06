import numpy as np
import sympy as sp

gam, b0, mu, c, en, r, alpha, b, R \
    = sp.symbols('gam, b0, mu, c, en, r, alpha, b, R')
b = b0 * mu
R = gam * b - c
TE, hW = sp.symbols('TE, hW')
F = sp.Matrix([[R * TE + gam * hW - en * (hW + b * TE)**3],
               [-r * hW - alpha * b * TE]])
P = sp.solve(F, TE, hW)
P0 = P[0]
Jac = F.jacobian((TE, hW))
Jac0 = Jac.subs(TE, P0[0]).subs(hW, P0[1])

spec = Jac0.eigenvals()
eval0 = sp.simplify(spec.items()[0][0].subs(b0, 2.5).subs(c, 1.).subs(en, 1.).subs(r, 0.25).subs(mu, 0.8))
eval1 = sp.simplify(spec.items()[1][0].subs(b0, 2.5).subs(c, 1.).subs(en, 1.).subs(r, 0.25).subs(mu, 0.8))
print eval0
print eval1
