from scipy.io import loadmat



data  = loadmat('WingForce.mat')
coefs = data['coefs']
knt   = data['knt'].flatten()

data  = loadmat('WingSection.mat')
coefs = data['coefs']
knt1  = data['knt1'].flatten()
knt2  = data['knt2'].flatten()

