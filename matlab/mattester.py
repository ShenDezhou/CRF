from scipy.io import loadmat

te = loadmat('te.mat')
print(te)
print(te['e'].shape)
print(te['t'].shape)
