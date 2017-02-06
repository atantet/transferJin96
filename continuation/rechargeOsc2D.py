import numpy as np

def getPeriod(xt):
    # Check for fixed point
    if np.sum((xt[-2] - xt[-1])**2) < 1.e-8:
        T, dist0 = 0., 0.
    else:
        nt = time.shape[0]
        dist0 = np.sum((xt + xt)**2) / nt
        distp1 = np.sum((xt[1:] + xt[:-1])**2) / (nt - 1)
        for t in np.arange(1, nt):
            distm1 = dist0
            dist0 = distp1
            distp1 = np.sum((xt[t+1:] + xt[:-t-1])**2) / (nt - t-1)
            if (dist0 < distm1) & (dist0 < distp1):
                break
        T = t * 2
    
    return (T, dist0)


def fieldRO2D(x, p, dummy=None):
    gam, b0, mu, c, en, r, alpha = p
    b = b0 * mu
    R = gam * b - c
    TE, hW = x[0], x[1]
    
    f = np.array([R * TE + gam * hW - en * (hW + b * TE)**3,
                  -r * hW - alpha * b * TE])
    
    return f

def JacobianRO2D(x, p):
    gam, b0, mu, c, en, r, alpha = p
    b = b0 * mu
    R = gam * b - c
    TE, hW = x[0], x[1]

    J = np.array([[R - 3*en*b * (hW + b*TE)**2,
                   gam - 3*en * (hW + b*TE)**2],
                  [-alpha * b, -r]])
    return J

def JacFieldRO2D(dx, p, x):
    return np.dot(JacobianRO2D(x, p), dx)


def stepRK4(x0, field, p, dt, costate=None):
    # Step solution forward
    k1 = field(x0, p, costate) * dt
    tmp = k1 * 0.5 + x0
    
    k2 = field(tmp, p, costate) * dt
    tmp = k2 * 0.5 + x0
    
    k3 = field(tmp, p, costate) * dt
    tmp = k3 + x0

    k4 = field(tmp, p, costate) * dt
    tmp = (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return x0 + tmp


def propagate(x0, field, p, scheme, dt, nt):
    '''Propagate solution of ODE according to the vector field field \
    with Euler scheme from x0 for nt time steps of size dt.'''
    xt = np.empty((nt, x0.shape[0]))
    xt[0] = x0.copy()
    for t in np.arange(1, nt):
        xt[t] = scheme(xt[t-1], field, p, dt)
        
    return xt

def propagateFundamental(x0, field, M0, JacField, p, scheme, dt, nt):
    '''Propagate solution of ODE according to the vector field field \
    with Euler scheme from x0 for nt time steps of size dt.'''
    dim = x0.shape[0]
    xt = np.empty((nt, dim))
    Mt = np.empty((nt, dim, dim))
    xt[0] = x0.copy()
    Mt[0] = M0.copy()
    for t in np.arange(1, nt):
        xt[t] = scheme(xt[t-1], field, p, dt)
        for d in np.arange(dim):
            Mt[t, :, d] = scheme(Mt[t - 1, :, d], JacField, p, dt, xt[t-1])
        
    return (xt, Mt)


# Config model
dim = 2
gam = 0.8
b0 = 2.5
c = 1.
en = 1.
r = 0.25
alpha = 0.125
mu = 0.8
b = b0 * mu

# Config simulation
spinup = 6 * 40 # 10 years
dt = 1. / 60 # 1 days
tdim = dt / 6
ntSpinup = int(spinup / dt + 0.1)

# Define diffusion matrix
sigma1 = 0.
sigma2 = 2. / 15 * 10
Sigma = np.matrix([[-alpha * sigma1, 0],
                   [gam * sigma1, sigma2]])
Q = np.dot(Sigma, Sigma.T)

gamRng = np.arange(0.7, 0.9, 0.02)
alphaRng = np.arange(.1, 0.15, 0.02)
(X, Y) = np.meshgrid(gamRng, alphaRng, indexing='ij')
FloquetExpRng = np.empty((gamRng.shape[0], alphaRng.shape[0], 2),
                         dtype=complex)
PhiRng = np.empty((gamRng.shape[0], alphaRng.shape[0]))
singularRng = np.zeros((gamRng.shape[0], alphaRng.shape[0]), bool)
detMTRng = np.zeros((gamRng.shape[0], alphaRng.shape[0]))
for igam in np.arange(gamRng.shape[0]):
    gam = gamRng[igam]
    R = gam * b - c
    print '\ngamma = ', gam
    for ialpha in np.arange(alphaRng.shape[0]):
        alpha = alphaRng[ialpha]
        print '\nalpha = ', alpha
        p = (gam, b0, mu, c, en, r, alpha)

        # Get initial state from converged simulation
        M0 = np.eye(dim)
        xs = np.array([0., 1.])
        xt = propagate(xs, fieldRO2D, p, stepRK4, dt, ntSpinup)
        xt = xt[ntSpinup/2:]
        time = np.arange(xt.shape[0]) * tdim
        x0 = xt[-1]
        (T, dist) = getPeriod(xt)
        print 'dist = ', dist
        if np.abs(T) < 1.e-8:
            print 'Fixed point'
            J = JacobianRO2D(x0, p)
            (eigVal, eigVec) = np.linalg.eig(J)
            isort = np.argsort(-np.abs(eigVal))
            eigVal = eigVal[isort]
            FloquetExpRng[igam, ialpha] = eigVal
            PhiRng[igam, ialpha] = np.nan
        elif dist > 1.e-5:
            print 'Not closed'
            FloquetExpRng[igam, ialpha] = np.nan
            PhiRng[igam, ialpha] = np.nan
        else:
            print 'Period = ', time[T]
            
            # Integrate
            (xt, Mt) = propagateFundamental(x0, fieldRO2D, M0, JacFieldRO2D,
                                            p, stepRK4, dt, T)
            MT = Mt[-1]
            (eigVal, eigVec) = np.linalg.eig(MT)
            isort = np.argsort(-np.abs(eigVal))
            eigVal = eigVal[isort]
            eigVec = eigVec[:, isort]
            (eigValLeft, eigVecLeft) = np.linalg.eig(MT.T)
            tmpVal = eigValLeft.copy()
            tmpVec = eigVecLeft.copy()
            for d in np.arange(dim):
                idx = np.argmin(np.abs(eigVal[d] - np.conjugate(tmpVal)))
                eigValLeft[d] = tmpVal[idx]
                eigVecLeft[:, d] = tmpVec[:, idx]

            FloquetExp = np.log(eigVal) / (T * tdim)
            print 'Exp 0 = ', FloquetExp[0]
            print 'Exp 1 = ', FloquetExp[1]

            colors = ['r', 'k']
            scale = 0.1
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(xt[:, 0], xt[:, 1], '-k')
            nevPlot = 10
            for t in np.linspace(0, T-1, nevPlot).astype(int):
                ax.scatter(xt[t, 0], xt[t, 1], s=40, c='k', edgecolors='face')
                for d in np.arange(dim):
                    ev = np.dot(Mt[t], eigVec[:, d])
                    ev *= scale / np.sqrt(np.sum(ev**2))
                    ax.plot([xt[t, 0], xt[t, 0] + ev[0]],
                            [xt[t, 1], xt[t, 1] + ev[1]],
                            linestyle='-', color=colors[d])
                    ax.set_xlabel(r'TE', fontsize='x-large')
                    ax.set_ylabel(r'hW', fontsize='x-large')
                    ax.set_title(r'$\alpha = %.3lf$ and $\gamma = %.3lf$' \
                                 % (alpha, gam), fontsize='xx-large')
                    #fig.savefig('test.eps')
            # Get fixed points
            xP = (R - gam*alpha*b/r) / (en*(1-alpha/r)**3*b**3)
            print 'xP = ', xP
            if xP > 0:
                xP = np.sqrt(xP)
                xP = np.array([xP, -alpha*b/r * xP])
                (eigFP, eigVecFP) = np.linalg.eig(JacobianRO2D(xP, p))
                for d in np.arange(dim):
                    ev = eigVecFP[:, d]
                    ev *= scale / np.sqrt(np.sum(ev**2))
                    ax.plot([xP[0], xP[0] + ev[0]],
                            [xP[1], xP[1] + ev[1]],
                            linestyle='-', color=colors[d])
                    ax.plot([-xP[0], -xP[0] - ev[0]],
                            [-xP[1], -xP[1] - ev[1]],
                            linestyle='-', color=colors[d])
                ax.scatter(xP[0], xP[1], s=40, c='r',
                           edgecolors='face')
                ax.scatter(-xP[0], -xP[1], s=40, c='r',
                           edgecolors='face')


            # Get correlation matrix and phase diffusion coefficient
            CT = np.zeros((dim, dim))
            detMTRng[igam, ialpha] = np.abs(np.linalg.det(MT))
            print 'det MT = ', detMTRng[igam, ialpha]
            if detMTRng[igam, ialpha] > 1.e-8:
                for t in np.arange(T):
                    iMt = np.linalg.inv(Mt[t])
                    CT += np.dot(np.dot(iMt, Q), iMt.T) * tdim
                #CT = np.dot(np.dot(MT, CT), MT.T)
                norm = np.dot(eigVecLeft[:, 0], eigVec[:, 0])
                print 'Norm = ', norm
                Phi = - np.dot(np.dot(eigVecLeft[:, 0], CT),
                               eigVecLeft[:, 0]) / norm / (T * tdim)
                print 'Phi = ', Phi
            else:
                singularRng[igam, ialpha] = True
                print 'Singular matrix at time ', t * tdim
                Phi = np.nan

            # Save
            FloquetExpRng[igam, ialpha] = FloquetExp
            PhiRng[igam, ialpha] = Phi
            

fig = plt.figure()
ax = fig.add_subplot(111)
cf = ax.contourf(X, Y, FloquetExpRng[:, :, 0].real)
plt.colorbar(cf)
ax.set_xlabel(r'$\alpha$', fontsize='xx-large')
ax.set_ylabel(r'$\gamma$', fontsize='xx-large')
ax.set_title(r'$\alpha_0$', fontsize='xx-large')
fig.savefig('FloquetExp0.eps')

fig = plt.figure()
ax = fig.add_subplot(111)
cf = ax.contourf(X, Y, FloquetExpRng[:, :, 1].real)
plt.colorbar(cf)
ax.set_ylabel(r'$\alpha$', fontsize='xx-large')
ax.set_xlabel(r'$\gamma$', fontsize='xx-large')
ax.set_title(r'$\alpha_1$', fontsize='xx-large')
fig.savefig('FloquetExp1.eps')

fig = plt.figure()
ax = fig.add_subplot(111)
vmax = Q[1, 1] * 5
vmin = 0.
PhiRngCut = np.abs(PhiRng.copy())
iszero = PhiRngCut < 1.e-8
if np.any(iszero):
    PhiRngCut[iszero] = np.nan
PhiRngCut = PhiRngCut
PhiRngCut[PhiRngCut > vmax] = vmax
PhiRngCut[PhiRngCut < vmin] = vmin
cf = ax.contourf(X, Y, PhiRngCut, 20)
plt.colorbar(cf)
ax.set_ylabel(r'$\alpha$', fontsize='xx-large')
ax.set_xlabel(r'$\gamma$', fontsize='xx-large')
ax.set_title(r'$\Phi_0$', fontsize='xx-large')
fig.savefig('Phi.eps')

fig = plt.figure()
ax = fig.add_subplot(111)
cf = ax.contourf(X, Y, np.log10(detMTRng))
plt.colorbar(cf)
ax.set_ylabel(r'$\alpha$', fontsize='xx-large')
ax.set_xlabel(r'$\gamma$', fontsize='xx-large')
ax.set_title(r'$\alpha_1$', fontsize='xx-large')

