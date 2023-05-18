import torch
import torch.nn as nn

import numpy as np

pi = 3.14
pi2 = 2*pi

def tonp(t):
    return t.detach().cpu().numpy()

def fromnp(x):
    return torch.from_numpy(x).float().cuda()

def rmse(yhat, y):
    if isinstance(yhat, np.ndarray):
        f = np.mean
    else:
        f = torch.mean
    return f((y-yhat)**2)**0.5

class AngleBasis(nn.Module):
    def __init__(self, n, d, use_jitter=True):
        super(AngleBasis, self).__init__()
        self.n = n
        self.d = d
        self.use_jitter = use_jitter
        self.thetas = nn.Parameter((pi*torch.rand(self.n,d)+pi/2).float().cuda())
        self.jitter = nn.Parameter(torch.ones(self.n,d).float().cuda())
        
    def project(self):
        if not self.use_jitter:
            return 
        with torch.no_grad():
            self.jitter[self.jitter < 0] = 0
            self.jitter[self.jitter > 1] = 1
        
    def phases(self):
        t0 = self.thetas.unsqueeze(2)
        t1 = self.thetas.unsqueeze(1)
        return t0-t1
    
    def jit(self):
        j0 = self.jitter.unsqueeze(2)
        j1 = self.jitter.unsqueeze(1)
        return j0*j1
    
    def ps(self):
        t = self.phases()
        p = torch.cos(t)
        if self.use_jitter:
            j = self.jit()
            p = j*p
        return p
    
    def dump(self):
        return dict(n=self.n, 
            use_jitter=self.use_jitter, 
            use_angle=True, 
            thetas=tonp(self.thetas), 
            jitter=tonp(self.jitter))
    
    def psum(self):
        return torch.mean(self.ps(), axis=0)
    
    def pvec(self):
        a,b = np.triu_indices(self.d,1)
        p = self.psum()
        return p[a,b]

def fc2ab(p, n, use_angle=True, use_jitter=True):
    if use_jitter and not use_angle:
    # Jitter only is low rank approximation
        w, v = np.linalg.eig(p)
        w[n:] = 0
        w = np.sqrt(np.real(w[:n]))
        v = np.real(v[:,:n])
        j = np.einsum('ab,b->ba', v, w)
        return dict(n=n, use_jitter=True, use_angle=False, thetas=0, jitter=j)
    # Angle + jitter or only jitter
    ab = fit_basis(p, n, use_jitter=use_jitter)
    return ab.dump()

def mat2vec(p):
    a,b = np.triu_indices(264,1)
    return p[a,b]

def fit_basis(p, n, use_jitter, nepochs=5000, pperiod=1000, verbose=False):
    d = p.shape[0]
    ab = AngleBasis(n, d, use_jitter=use_jitter)
    optim = torch.optim.Adam(ab.parameters(), lr=0.01, weight_decay=0)
    p = fromnp(mat2vec(p))
    for e in range(nepochs):
        optim.zero_grad()
        xhat = ab.pvec()
        loss = rmse(xhat, p)
        loss.backward()
        optim.step()
        ab.project()
        if verbose:
            if e == nepochs-1 or e % pperiod == 1:
                print(f'{e} {float(loss)}')
    if verbose:
        print('Complete')
    return ab

def ab2ps(abdict):
    if abdict['use_jitter'] and abdict['use_angle']:
        thetas = abdict['thetas']
        jitter = abdict['jitter']
        t0 = np.expand_dims(thetas, 1)
        t1 = np.expand_dims(thetas, 2)
        j0 = np.expand_dims(jitter, 1)
        j1 = np.expand_dims(jitter, 2)
        ps = np.cos(t0-t1)*(j0*j1)
    elif abdict['use_jitter']:
        jitter = abdict['jitter']
        j0 = np.expand_dims(jitter, 1)
        j1 = np.expand_dims(jitter, 2)
        ps = j0*j1
    else:
        thetas = abdict['thetas']
        t0 = np.expand_dims(thetas, 1)
        t1 = np.expand_dims(thetas, 2)
        ps = np.cos(t0-t1)
    return ps

def ab2fc(abdict):
    if abdict['use_jitter'] and not abdict['use_angle']:
        return np.sum(ab2ps(abdict), axis=0)
    else:
        return np.mean(ab2ps(abdict), axis=0)

def ab2fcvec(abdict):
    a,b = np.triu_indices(264,1)
    return ab2fc(abdict)[a,b]
