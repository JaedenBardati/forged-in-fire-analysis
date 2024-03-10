import analysis as an
import numpy as np
import yt
import matplotlib.pyplot as plt

step = 334
snap = an.load_fifs_box(step=step, width='1 pc')
center = snap.BH_pos
length=an.parse_unit('1 pc')

N = 500
integrated_col_dens = []
thetas, phis = [], []
for i in range(N):
    #get randomly angled ray
    phi = 2*np.pi*np.random.uniform(0, 1)
    phis.append(phi)
    theta = np.arccos(np.random.uniform(-1, 1))
    thetas.append(theta)
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    start = ([x, y, z]*length).in_units('pc') + center.in_units('pc') # inner parsec radius
    end = center.in_units('pc')
    an.LOGGER.info('getting ray %d/%d...' %(i+1, N))
    ray = snap.ds.r[start:end]
    an.LOGGER.info(' > getting density %d/%d...' %(i+1, N))

    # integrate intensity
    col_dens = (ray[('Dust', 'density')]*length).sum()   # temp, should really integrate over length somehow!! (replace with something like ray[('Dust', 'radius')]? note this probably doesnt exist yet)
    an.LOGGER.info(' > {} {} --> {}'.format(theta, phi, col_dens))
    integrated_col_dens.append(col_dens)


plt.figure()
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(phis, thetas, c=integrated_col_dens, s=35, cmap=cm)
plt.colorbar(sc)
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\theta$')
plt.xlim([0, 2*np.pi])
plt.ylim([0, np.pi])
plt.savefig('integrated_col_dens.png')

import pandas as pd
df = pd.DataFrame({'phis':phis, 'thetas': thetas, 'integrated_col_dens':integrated_col_dens})
df.to_csv('integrated_col_dens.csv')

print('Done now.')

