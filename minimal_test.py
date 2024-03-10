import yt

filepath = "/scratch3/01799/phopkins/fire3_suite_done/m13h206_m3e5/m13h206_m3e5_hyperref_5_done/output__collected/snapshot_334.hdf5"
print('Attempting to load snapshot...')

ds = yt.load(filepath)

print('All done.')