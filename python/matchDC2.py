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
from match_utils import spatial_closest_mag_1band
import astropy.units as u
GCRCatalogs.set_root_dir('/global/cfs/projectdirs/lsst/shared') 

parser = argparse.ArgumentParser()
parser.add_argument('--init', dest='init', type=int, default=0, help='Tract to start')
parser.add_argument('--end', dest='end', type=int, default=9999, help='Tract to end')
parser.add_argument('--data-catalog', dest='data_catalog', type=str, default='dc2_object_run2.2i_dr3',
                   help='(data) GCR catalog name to query')
parser.add_argument('--truth-catalog', dest='truth_catalog', type=str, default='dc2_truth_run2.2i_galaxy_truth_summary',
                   help='(truth) GCR catalog name to query')
parser.add_argument('--star-catalog', dest='star_catalog', type=str, default='dc2_truth_run2.2i_star_truth_summary',
                   help='(truth) stellar catalog name to query')
parser.add_argument('--filter-band', dest='band', type=str, default='r', 
                   help='Band to use for the matching (ignored if spatial_closest_mag is not used)')
parser.add_argument('--max-deltamag', dest='max_deltamag', type=float, default=1.0,
                   help='Maximum magnitude difference allowed for an object to be considered a match')
parser.add_argument('--search-radius', dest='max_radius', type=float, default=5.0,
                   help='Search radius for matching in units of number of pixels (pixel size = 0.2 arcsec)')
args = parser.parse_args()
object_cat = GCRCatalogs.load_catalog(args.data_catalog)
galaxy_true = GCRCatalogs.load_catalog(args.truth_catalog)
star_true = GCRCatalogs.load_catalog(args.star_catalog)
# Add magnitude quantity modifier:
bands =  ['u', 'g', 'r', 'i', 'z', 'y']
for band in bands:
    galaxy_true.add_derived_quantity(f'mag_{band}', lambda x: (x*u.nJy).to(u.ABmag).value, f'flux_{band}')
    star_true.add_derived_quantity(f'mag_{band}', lambda x: (x*u.nJy).to(u.ABmag).value, f'flux_{band}')
col_mags = [f'mag_{band}_cModel' for band in bands]
col_mags_true = [f'mag_{band}' for band in bands]
columns = ['ra', 'dec', 'objectId'] + col_mags
columns_true = ['id','ra', 'dec', 'redshift', 'is_pointsource', 'is_variable'] + col_mags_true
tracts = np.array(object_cat.available_tracts)
print('Tracts available', len(tracts))
# We first query all the stars in the DC2 footprint with magnitude < 30
data_stars_true = star_true.get_quantities(columns_true, filters=[f'mag_{args.band} < 30'])

# We now create a "bad truth object" to point to in case that there are no matches in the truth catalog"
bad_obj = dict()
for col in columns_true:
    bad_obj[col] = [-99]
bad_obj = astropy.table.Table(bad_obj)

if 'dr1b' in args.data_catalog:
    tracts = tracts[tracts!=2897] # Patch to avoid a faulty tract at DC2 run 2.1i DR1B
if args.end > len(tracts-1):
    args.end = len(tracts-1)
for tract in tracts[args.init:args.end]:
    if os.path.isfile(f'/global/cscratch1/sd/jsanch87/DC2_Run2.2i/matched_ids_{args.data_catalog}_{tract}.fits.gz'):
        continue
    else:
        print('Generating match for tract:', tract)
        data_meas = object_cat.get_quantities(columns, native_filters= [f'tract == {tract}']) #tract 2897 missing/corrupt
        print('Got', len(data_meas['ra']), 'objects in the object catalog')
        # There are some nuances in the boundaries of tracts that we are ignoring here
        max_ra = np.nanmax(data_meas['ra'])
        min_ra = np.nanmin(data_meas['ra'])
        max_dec = np.nanmax(data_meas['dec'])
        min_dec = np.nanmin(data_meas['dec'])
        vertices = hp.ang2vec(np.array([min_ra, max_ra, max_ra, min_ra]), np.array([min_dec, min_dec, max_dec, max_dec]), lonlat=True)
        print('vertices: ', min_ra, max_ra, min_dec, max_dec)
        ipix = hp.query_polygon(32, vertices, inclusive=True)
        print('healpixels to query', ipix)
        native_filter = f'(healpix == {ipix[0]})'
        for ipx in ipix:
            native_filter=native_filter+f' | (healpix == {ipx})'
        data_galaxies_true = galaxy_true.get_quantities(columns_true,
            filters=[f'ra >= {min_ra}',f'ra <={max_ra}', f'dec >= {min_dec}', f'dec <= {max_dec}',
                f'mag_{args.band} < 30'], native_filters=native_filter)
        print('Got', len(data_galaxies_true['ra']), 'true galaxies')
        _tab_gal = astropy.table.Table(data_galaxies_true)
        _tab_star = astropy.table.Table(data_stars_true)
        _sel_star = (_tab_star['ra'] >= min_ra) & (_tab_star['ra'] <= max_ra) & (_tab_star['dec'] >= min_dec) & (_tab_star['dec'] <= max_dec)
        _tab_star = _tab_star[_sel_star]
        truth_table = astropy.table.vstack(_tab_gal, _tab_star) 
        dist, ids, matched, n_neigh = spatial_closest_mag_1band(data_meas['ra'],
                                               data_meas['dec'],
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
                           truth_table['redshift'][ids], dist, n_neigh],
                          names=('truthId','objectId','is_matched','is_star', 'ra', 'dec',
                                 'mag_u_lsst', 'mag_g_lsst','mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst', 'redshift_true', 'dist', 'n_neigh'))
        tab_out.write(f'/global/cscratch1/sd/jsanch87/DC2_Run2.2i/matched_ids_{args.data_catalog}_{tract}.fits.gz', overwrite=True)

