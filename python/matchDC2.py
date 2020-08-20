import numpy as np
import glob
import fitsio
import sys
sys.path.insert(0, '/global/homes/j/jsanch87/gcr-catalogs')
import GCRCatalogs
print(GCRCatalogs.__file__)
import GCR
from sklearn.neighbors import KDTree
import astropy.table
import healpy as hp
import os
import argparse
from match_utils import *
import astropy.units as u
import matplotlib.pyplot as plt

GCRCatalogs.set_root_dir('/global/cfs/projectdirs/lsst/shared') 

parser = argparse.ArgumentParser()

parser.add_argument('--init', dest='init', type=int, default=0, help='Tract to start')
parser.add_argument('--end', dest='end', type=int, default=9999, help='Tract to end')
parser.add_argument('--data-catalog', dest='data_catalog', type=str, default='dc2_object_run2.2i_dr3',
                   help='(data) GCR catalog name to query')
parser.add_argument('--truth-catalog', dest='truth_catalog', type=str, default='cosmoDC2_v1.1.4_image',
                   help='(truth) GCR catalog name to query')
parser.add_argument('--star-catalog', dest='star_catalog', type=str, default='dc2_truth_run2.2i_star_truth_summary',
                   help='(truth) stellar catalog name to query')
parser.add_argument('--filter-band', dest='band', type=str, default='r', 
                   help='Band to use for the matching (ignored if spatial_closest_mag is not used)')
parser.add_argument('--max-deltamag', dest='max_deltamag', type=float, default=1.0,
                   help='Maximum magnitude difference allowed for an object to be considered a match')
parser.add_argument('--search-radius', dest='max_radius', type=float, default=5.0,
                   help='Search radius for matching in units of number of pixels (pixel size = 0.2 arcsec)')
parser.add_argument('--debug', dest='debug', default=False, action='store_true',
                   help='Show debugging information')
parser.add_argument('--outdir', dest='outdir', default='/global/cscratch1/sd/jsanch87/dr6c_matches_wshear',
                   help='Output directory for matched files')

args = parser.parse_args()
object_cat = GCRCatalogs.load_catalog(args.data_catalog)
galaxy_true = GCRCatalogs.load_catalog(args.truth_catalog)
star_true = GCRCatalogs.load_catalog(args.star_catalog)

# Add magnitude quantity modifier and others:

bands =  ['u', 'g', 'r', 'i', 'z', 'y']

if 'truth' not in args.truth_catalog:
    galaxy_true.add_derived_quantity('is_pointsource', lambda x: (0*x).astype(np.bool), f'ra')
    galaxy_true.add_derived_quantity('is_variable', lambda x: (0*x).astype(np.bool), f'ra')
    galaxy_true.add_derived_quantity('id', lambda x: x, 'galaxy_id')

for band in bands:
    if 'truth' in args.truth_catalog:
        galaxy_true.add_derived_quantity(f'mag_{band}', lambda x: (x*u.nJy).to(u.ABmag).value, f'flux_{band}')
    star_true.add_derived_quantity(f'mag_{band}', lambda x: (x*u.nJy).to(u.ABmag).value, f'flux_{band}')
col_mags = [f'mag_{band}_cModel' for band in bands]
col_mags_true = [f'mag_{band}' for band in bands]
columns = ['ra', 'dec', 'objectId'] + col_mags
shear_cols = ['ellipticity_1_true', 'ellipticity_2_true', 'shear_1', 'shear_2']
columns_true = ['id','ra', 'dec', 'redshift', 'is_pointsource', 'is_variable'] + col_mags_true
tracts = np.array(object_cat.available_tracts)

print('Tracts available', len(tracts))

# We first query all the stars in the DC2 footprint with magnitude < 30

data_stars_true = star_true.get_quantities(columns_true, filters=[f'mag_{args.band} < 30'])
tab_star = astropy.table.Table(data_stars_true)
for col in shear_cols:
    tab_star[col] = np.zeros(len(tab_star))

# We now create a "bad truth object" to point to in case that there are no matches in the truth catalog"

bad_obj = dict()
for col in columns_true:
    bad_obj[col] = [-99]
bad_obj = astropy.table.Table(bad_obj)
padding = 0.1 # Padding in ra, dec to match

#if 'dr1b' in args.data_catalog:
#    tracts = tracts[tracts!=2897] # Patch to avoid a faulty tract at DC2 run 2.1i DR1B

if args.end > len(tracts-1):
    args.end = len(tracts-1)

for i, tract in enumerate(tracts[args.init:args.end]):
    nameout = os.path.join(args.outdir, f'matched_ids_{args.data_catalog}_{tract}.fits.gz')
    if os.path.isfile(nameout):
        continue
    else:
        print('Generating match for tract:', tract, i)
        data_meas = object_cat.get_quantities(columns, native_filters= [f'tract == {tract}']) #tract 2897 missing/corrupt
        print('Got', len(data_meas['ra']), 'objects in the object catalog')
        # There are some nuances in the boundaries of tracts that we are ignoring here
        max_ra = np.nanmax(data_meas['ra'])+padding
        min_ra = np.nanmin(data_meas['ra'])-padding
        max_dec = np.nanmax(data_meas['dec'])+padding
        min_dec = np.nanmin(data_meas['dec'])-padding
        vertices = hp.ang2vec(np.array([min_ra, max_ra, max_ra, min_ra]), np.array([min_dec, min_dec, max_dec, max_dec]), lonlat=True)
        print('vertices: ', min_ra, max_ra, min_dec, max_dec)
        ipix = hp.query_polygon(32, vertices, inclusive=True)
        print('healpixels to query', ipix)
        if 'truth' in args.truth_catalog:
            native_filter = f'(healpix == {ipix[0]})' # The native quantity for the truth catalogs is called healpix instead of healpix_pixel
        else:
            native_filter = f'(healpix_pixel == {ipix[0]})'
        for ipx in ipix:
            if 'truth' in args.truth_catalog:
                native_filter=native_filter+f' | (healpix == {ipx})'
            else:
                native_filter=native_filter+f' | (healpix_pixel == {ipx})'
        data_galaxies_true = galaxy_true.get_quantities(columns_true+shear_cols,
            filters=[f'ra >= {min_ra}',f'ra <={max_ra}', f'dec >= {min_dec}', f'dec <= {max_dec}',
                f'mag_{args.band} < 30'], native_filters=native_filter)
        print('Got', len(data_galaxies_true['ra']), 'true galaxies')
        _tab_gal = astropy.table.Table(data_galaxies_true)
        _sel_star = (tab_star['ra'] >= min_ra-padding) & (tab_star['ra'] <= max_ra+padding) & (tab_star['dec'] >= min_dec-padding) & (tab_star['dec'] <= max_dec+padding)
        _tab_star = tab_star[_sel_star]
        truth_table = astropy.table.vstack(_tab_gal, _tab_star)
        if args.debug:
            plt.figure()
            plt.scatter(truth_table['ra'], truth_table['dec'])
            plt.scatter(data_meas['ra'], data_meas['dec'])
        dist, ids, matched, n_neigh_obj, n_neigh_truth = spatial_closest_mag_1band_3d(data_meas['ra'], data_meas['dec'],
                                                                                   data_meas[f'mag_{args.band}_cModel'],
                                                                                   truth_table['ra'],
                                                                                   truth_table['dec'],
                                                                                   truth_table[f'mag_{args.band}'],
                                                                                   np.arange(len(truth_table)),
                                                                                   max_deltamag=args.max_deltamag, rmax=args.max_radius)
        ## We return the distance in arcseconds
        truth_table = astropy.table.vstack(truth_table, bad_obj)
        ids[ids==-99] = -1 # We assign the bad matches to the "bad object" since it's the last

        tab_out = astropy.table.Table([truth_table['id'][ids], data_meas['objectId'], matched, truth_table['is_pointsource'][ids], 
                           truth_table['ra'][ids], truth_table['dec'][ids], truth_table['mag_u'][ids], truth_table['mag_g'][ids], 
                           truth_table['mag_r'][ids], truth_table['mag_i'][ids], truth_table['mag_z'][ids], truth_table['mag_y'][ids], 
                           truth_table['redshift'][ids], dist, n_neigh_truth, n_neigh_obj, truth_table['shear_1'][ids],
                           truth_table['shear_2'][ids], truth_table['ellipticity_1_true'][ids], truth_table['ellipticity_2_true'][ids]],
                           names=('truthId','objectId','is_matched','is_star', 'ra', 'dec',
                                 'mag_u_lsst', 'mag_g_lsst','mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst',
                                 'redshift_true', 'dist', 'n_neigh_truth', 'n_neigh_object', 'shear_1', 'shear_2', 
                                 'ellipticity_1_true', 'ellipticity_2_true'))
        if args.debug:
            plt.figure()
            plt.scatter(truth_table['ra'][ids][matched], data_meas['ra'][matched])
            plt.figure()
            plt.scatter(truth_table['dec'][ids][matched], data_meas['dec'][matched])
            plt.figure()
            plt.hist(truth_table['redshift'][ids][matched])
            plt.show()
        tab_out.write(nameout, overwrite=False)

