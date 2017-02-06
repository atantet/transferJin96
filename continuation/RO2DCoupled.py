import numpy as np
import scipy.optimize
from ergoInt import *

import numpy as np
import scipy.optimize
from ergoInt import *

def nl(x):
    return (x - x**3 / 3)
def dnl(x):
    return 1. - x**2
#def nl(x):
#   return np.tanh(x)
#def dnl(x):
#   return 1. - np.tanh(x)**2

def H(x):
    return ((np.sign(x) + 1) / 2)


def fieldRO2D(X, p, dummy=None):
    (x, y) = X
    tau = p['tauExtp'] + p['mu'] * x
    w = -p['deltas'] * tau + p['w0p']
    mv = p['von'] * p['deltas'] * tau
    ye = y + p['gam'] * tau
    yp = p['eta1'] * ye + p['eta2']
    xs = p['xs0'] * (1. - nl(yp))
    
    f = np.array([-p['alpha'] * x - H(w) * w * (x - xs) - H(mv) * mv * x,
                  -p['rp'] * (y + p['gam'] / 2 * tau)])
    
    return f


def JacobianRO2D(X, p):
    (x, y) = X
    tau = p['tauExtp'] + p['mu'] * x
    w = -p['deltas'] * tau + p['w0p']
    mv = p['von'] * p['deltas'] * tau
    ye = y + p['gam'] * tau
    yp = p['eta1'] * ye + p['eta2']
    xs = p['xs0'] * (1. - nl(yp))

    # Derivatives
    dtaudx = p['mu']
    dtaudy = 0.
    dwdx = -p['deltas'] * dtaudx
    dwdy = -p['deltas'] * dtaudy # (= 0.)
    dmvdx = p['von'] * p['deltas'] * dtaudx
    dmvdy = p['von'] * p['deltas'] * dtaudy # (= 0.)
    dyedx = p['gam'] * dtaudx
    dyedy = 1. + p['gam'] * dtaudy # (= 1.)
    dypdx = p['eta1'] * dyedx
    dypdy = p['eta1'] * dyedy
    dxsdx = -p['xs0'] * dypdx * dnl(yp)
    dxsdy = -p['xs0'] * dypdy * dnl(yp)

    J = np.array([[-p['alpha'] - H(w) * dwdx * (x - xs) - H(w) * w * (1. - dxsdx) \
                   - H(mv) * dmvdx * x - H(mv) * mv,
                   - H(w) * dwdy * (x - xs) + H(w) * w * dxsdy - H(mv) * dmvdy * x],
                  [-p['rp'] * p['gam'] / 2 * dtaudx,
                   -p['rp'] * (1. + p['gam'] / 2 * dtaudy)]])
    
    return J


def JacFieldRO2D(dX, p, X):
    return np.dot(JacobianRO2D(X, p), dX)


def diagnose(X, p, pdim):
    # Get adimensional variables
    x = X[:, 0]
    y = X[:, 1]
    tau = p['tauExtp'] + p['mu'] * x
    w = -p['deltas'] * tau + p['w0p']
    mv = p['von'] * p['deltas'] * tau
    ye = y + p['gam'] * tau
    yp = p['eta1'] * ye + p['eta2']
    xs = p['xs0'] * (1. - nl(yp))

    # Get dimensional variables
    diagnostic = {}
    diagnostic['TE'] = x * pdim['DeltaT'] + pdim['T0']
    diagnostic['Ts'] = xs * pdim['DeltaT'] + pdim['T0']
    diagnostic['hW'] = y * pdim['Hm']
    diagnostic['hE'] = ye * pdim['Hm']
    diagnostic['tau'] = tau * pdim['tau0']
    diagnostic['w'] = w * pdim['Hm'] * pdim['c0'] / pdim['L']
    diagnostic['v'] = -mv * pdim['Ly'] * pdim['c0'] / pdim['L'] / 2

    return diagnostic

    
def diagnoseMulti(X, pMulti, pdim):
    n = X.shape[0]

    # Allocate
    diagnostic = {}
    diagnostic['TE'] = np.empty((n,))
    diagnostic['Ts'] = np.empty((n,))
    diagnostic['hW'] = np.empty((n,))
    diagnostic['hE'] = np.empty((n,))
    diagnostic['tau'] = np.empty((n,))
    diagnostic['w'] = np.empty((n,))
    diagnostic['v'] = np.empty((n,))
    for k in np.arange(n):
        p = pMulti[k]

        # Get adimensional variables
        x = X[k, 0]
        y = X[k, 1]
        tau = p['tauExtp'] + p['mu'] * x
        w = -p['deltas'] * tau + p['w0p']
        mv = p['von'] * p['deltas'] * tau
        ye = y + p['gam'] * tau
        yp = p['eta1'] * ye + p['eta2']
        xs = p['xs0'] * (1. - nl(yp))

        # Get dimensional variables
        diagnostic['TE'][k] = x * pdim['DeltaT'] + pdim['T0']
        diagnostic['Ts'][k] = xs * pdim['DeltaT'] + pdim['T0']
        diagnostic['hW'][k] = y * pdim['Hm']
        diagnostic['hE'][k] = ye * pdim['Hm']
        diagnostic['tau'][k] = tau * pdim['tau0']
        diagnostic['w'][k] = w * pdim['Hm'] * pdim['c0'] / pdim['L']
        diagnostic['v'][k] = -mv * pdim['Ly'] * pdim['c0'] / pdim['L'] / 2

    return diagnostic

    
# Reference variables
pdim = {}
pdim['T0'] = 30. # (K) Radiative equilibrium temperature
pdim['Ts0'] = 24. # (K) Thermocline reference temperature
pdim['DeltaT'] = 1. # (K) Reference temperature difference
pdim['Hm'] = 50. # (m) Mixed-layer depth
pdim['Hs'] = 50. # (m) Steepness of the tanh
pdim['h0'] = 25. # (m) Offset of the tanh
pdim['L'] = 1.5 * 10**7 # (m) Width of the basin
pdim['Ly'] = 1. * 10**6 # (m) Meridional length
pdim['epsT'] = 1. / (150 * 24*60*60) # (s^-1) SST damping rate
pdim['c0'] = 2. # (m s^-1) Velocity of the first baroclinic Kelvin mode
pdim['aM'] = 1.3*10**(-8) # (s^-1) Rayleigh friction coefficient
pdim['tau0'] = 2.667 * 10**(-7) # (m s^-2) Reference wind stress
pdim['r'] = 1. / (500 * 24*60*60) # (s^-1) Dynamical adjustment timescale
# of the western equatorial thermocline depth by the zonally integrated
# Sverdrup meridional mass transport resulting from wind-forced Rossby waves
pdim['b'] = 4.874 * 0.9# (s) Efficiency of wind stress in driving the thermocline

pdim['tauExt'] = -0.2 * pdim['tau0'] # (m s^-2) External wind stress
pdim['w0'] = 0. # (m s^-1) Upwelling due to mixing and/or stronger clim resp
minTE = pdim['T0'] - 2 * (pdim['T0'] - pdim['Ts0'])
maxTE = pdim['T0']
minhW = -10.
maxhW = 60.

# Adimensional parameters
p = {}
p['alpha'] = pdim['epsT'] * pdim['L'] / pdim['c0']
p['eta1'] = pdim['Hm'] / pdim['Hs']
p['eta2'] = pdim['h0'] / pdim['Hs']
p['rp'] = pdim['r'] * pdim['L'] / pdim['c0']
p['gam'] = pdim['b'] * pdim['L'] * pdim['tau0'] / pdim['Hm']
p['xs0'] = (pdim['Ts0'] - pdim['T0']) / pdim['DeltaT']
p['tauExtp'] = pdim['tauExt'] / pdim['tau0']
p['w0p'] = pdim['w0'] * pdim['L'] / (pdim['Hm'] * pdim['c0'])

p['von'] = 1.
p['deltas'] = 1.
#p['mu'] = .4

# Config model
dim = 2
day2sec = 24 * 60 * 60
year2day = 365
tdim = pdim['L'] / pdim['c0']

# Config simulation
spinup = 50. * year2day * day2sec / tdim # 50 years
dt = 1. * day2sec / tdim  # 1 day
tdimYear = dt * tdim / day2sec / year2day
ntSpinup = int(spinup / dt + 0.1)

#muRng = np.arange(0., 0.361, 0.03)
muRng = np.arange(0.5, 0.8, 0.01)
#muRng = np.array([0.478])
#muRng = np.array([0.])

# Define diffusion matrix
sigma1 = 0.
sigma2 = 2. / 15 * 10
Sigma = np.matrix([[-alpha * sigma1, 0],
                   [gam * sigma1, sigma2]])
Q = np.dot(Sigma, Sigma.T)

TEFP = np.empty((muRng.shape[0],))
phiRng = np.empty((muRng.shape[0],))
for imu in np.arange(muRng.shape[0]):
    p['mu'] = muRng[imu]
    print 'mu = ', p['mu']

    # Get initial state from converged simulation
    M0 = np.eye(dim)
    x0s = np.array([-1., 0.])
    xt = propagate(x0s, fieldRO2D, p, stepRK4, dt, ntSpinup)
    #xt = xt[:40]
    time = np.arange(xt.shape[0]) * tdimYear
    diagnostic = diagnose(xt, p, pdim)
    xtCut = xt[xt.shape[0]/2:]
    timeCut = np.arange(xtCut.shape[0]) * tdimYear
    x0 = xt[-1]
    (T, dist) = getPeriod(xtCut, step=1)
    print 'dist = ', dist
    if np.abs(T) < 1.e-8:
        print 'Fixed point'
        
    elif dist > 1.e-5:
        print 'Not closed'
        FloquetExp = np.nan
        Phi = np.nan
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(diagnostic['TE'], diagnostic['hW'], '-k')
        ax.scatter(diagnostic['TE'][0], diagnostic['hW'][0], s=40, c='k')
        ax.set_xlim(minTE, maxTE)
        ax.set_ylim(minhW, maxhW)
        ax.set_xlabel(r'$TE$', fontsize='xx-large')
        ax.set_ylabel(r'$hW$', fontsize='xx-large')
        fig.savefig('orbit_mu%04d.eps' % (int(p['mu'] * 100+0.1),),
                    dpi=300, bbox_tight=True)

        
        # fig = plt.figure()
        # ax1 = fig.add_subplot(511)
        # ax1.plot(time, diagnostic['TE'], '-r')
        # ax1.plot(time, diagnostic['Ts'], '-b')
        # ax1.set_ylabel(r'TE, TS', fontsize='xx-large')
        # ax2 = fig.add_subplot(512)
        # ax2.plot(time, diagnostic['hW'], '-r')
        # ax2.plot(time, diagnostic['hE'], '-b')
        # ax2.set_ylabel(r'hW, hE', fontsize='xx-large')
        # ax3 = fig.add_subplot(513)
        # ax3.plot(time, diagnostic['tau'], '-k')
        # ax3.set_ylabel(r'$\tau$', fontsize='xx-large')
        # ax4 = fig.add_subplot(514)
        # ax4.plot(time, diagnostic['w'], '-b')
        # ax4.set_ylabel(r'$w$', fontsize='xx-large')
        # ax5 = fig.add_subplot(515)
        # ax5.plot(time, diagnostic['v'], '-k')
        # ax5.set_ylabel(r'$v$', fontsize='xx-large')
        # ax5.set_xlabel(r'$t$', fontsize='xx-large')
    
        
    else:
        print 'Period = ', timeCut[T]
        
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

        FloquetExp = np.log(eigVal) / (T * tdimYear)
        print 'Exp 0 = ', FloquetExp[0]
        print 'Exp 1 = ', FloquetExp[1]
        
        # Get correlation matrix and phase diffusion coefficient
        CT = np.zeros((dim, dim))
        detMT = np.abs(np.linalg.det(MT))
        print 'det MT = ', detMT
        if detMT > 1.e-8:
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
            print 'Singular matrix at time ', t * tdim
            Phi = np.nan
            
        colors = ['r', 'k']
        scale = 0.1
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(diagnostic['TE'], diagnostic['hW'], '-k')
        ax.scatter(diagnostic['TE'][0], diagnostic['hW'][0], s=40, c='k')
        ax.set_xlim(minTE, maxTE)
        ax.set_ylim(minhW, maxhW)
        ax.set_xlabel(r'$TE$', fontsize='xx-large')
        ax.set_ylabel(r'$hW$', fontsize='xx-large')
        fig.savefig('orbit_mu%04d.eps' % (int(p['mu'] * 100+0.1),),
                    dpi=300, bbox_tight=True)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(muRng, TEFP)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(muRng, phiRng)
