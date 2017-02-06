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
pdim['b'] = 4.4 # (s) Efficiency of wind stress in driving the thermocline

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

muRng = np.arange(0., 1.2001, 0.01)

FPs = np.empty((muRng.shape[0], dim))
diags = []
eigVals = np.empty((muRng.shape[0], dim), dtype=complex)
Js = np.empty((muRng.shape[0], dim, dim))

imu = 0
p['mu'] = muRng[imu]
pMulti = [p.copy()]
FPs[imu] = np.array([-0.7787037, -0.28935185])
Js[imu] = JacobianRO2D(FPs[imu], p)
(w, v) = np.linalg.eig(Js[imu])
isort = np.argsort(-w.real)
eigVals[imu] = w[isort]

for imu in np.arange(1, muRng.shape[0]):
    p['mu'] = muRng[imu]
    pMulti.append(p.copy())

    info = scipy.optimize.fsolve(fieldRO2D, FPs[imu-1], p, JacobianRO2D,
                                 full_output=True)
    if not info[2]:
        print info[3]
        break
    FPs[imu] = info[0]
    Js[imu] = JacobianRO2D(FPs[imu], p)
    (w, v) = np.linalg.eig(Js[imu])
    isort = np.argsort(-w.real)
    eigVals[imu] = w[isort]

stable = eigVals[:, 0].real < 0
diagnostic = diagnoseMulti(FPs, pMulti, pdim)

lw = 2
fig = plt.figure(figsize=[8, 10])
ax1 = fig.add_subplot(511)
ax1.plot(muRng, pdim['T0'] * np.ones((muRng.shape[0],)), '-r')
ax1.plot(muRng, pdim['Ts0'] * np.ones((muRng.shape[0],)), '-b')
ax1.plot(muRng, diagnostic['TE'], '-r', linewidth=lw)
ax1.plot(muRng, diagnostic['Ts'], '-b', linewidth=lw)
ylim = ax1.get_ylim()
for s in np.arange(1, stable.shape[0]):
    if (stable[s-1] ^ stable[s]):
        ax1.plot([muRng[s], muRng[s]], ylim, '--k')
ax1.set_ylabel(r'$TE, TS$', fontsize='xx-large')
ax2 = fig.add_subplot(512)
ax2.plot(muRng, diagnostic['hW'], '-r', linewidth=lw)
ax2.plot(muRng, diagnostic['hE'], '-b', linewidth=lw)
ax2.set_ylabel(r'$hW, hE$', fontsize='xx-large')
ylim = ax2.get_ylim()
for s in np.arange(1, stable.shape[0]):
    if (stable[s-1] ^ stable[s]):
        ax2.plot([muRng[s], muRng[s]], ylim, '--k')
ax3 = fig.add_subplot(513)
ax3.plot(muRng, pdim['tauExt'] * np.ones((muRng.shape[0],)), '-k')
ax3.plot(muRng, diagnostic['tau'], '-k', linewidth=lw)
ax3.set_ylabel(r'$\tau$', fontsize='xx-large')
ylim = ax3.get_ylim()
for s in np.arange(1, stable.shape[0]):
    if (stable[s-1] ^ stable[s]):
        ax3.plot([muRng[s], muRng[s]], ylim, '--k')
ax4 = fig.add_subplot(514)
ax4.plot(muRng, diagnostic['w'], '-b', linewidth=lw)
ax4.set_ylabel(r'$w$', fontsize='xx-large')
ylim = ax4.get_ylim()
for s in np.arange(1, stable.shape[0]):
    if (stable[s-1] ^ stable[s]):
        ax4.plot([muRng[s], muRng[s]], ylim, '--k')
ax5 = fig.add_subplot(515)
ax5.plot(muRng, diagnostic['v'], '-k', linewidth=lw)
ax5.set_ylabel(r'$v$', fontsize='xx-large')
ax5.set_xlabel(r'$\mu$', fontsize='xx-large')
ylim = ax5.get_ylim()
for s in np.arange(1, stable.shape[0]):
    if (stable[s-1] ^ stable[s]):
        ax5.plot([muRng[s], muRng[s]], ylim, '--k')
fig.savefig('FP0.eps', dpi=300, bbox_tight=True)

fig = plt.figure()
ax1 = fig.add_subplot(211)
for d in np.arange(dim):
    ax1.plot(muRng, eigVals[:, d].real, linewidth=lw)
ax1.plot(muRng, np.zeros((muRng.shape[0],)), '-k')
ax1.set_ylabel(r'$\Re(\lambda_i)$', fontsize='xx-large')
ax1.set_xlim(muRng[0], muRng[-1])
ax1.set_ylim(-5., 1.)
ax2 = fig.add_subplot(212)
for d in np.arange(dim):
    ax2.plot(muRng, eigVals[:, d].imag, linewidth=lw)
ax2.set_ylabel(r'$\Im(\lambda_i)$', fontsize='xx-large')
ax2.set_xlabel(r'$\mu$', fontsize='xx-large')
ax2.set_xlim(muRng[0], muRng[-1])
ax2.set_ylim(-2., 2.)
fig.savefig('Exp0.eps', dpi=300, bbox_tight=True)
