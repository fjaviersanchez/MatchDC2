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
import FoFCatalogMatching
GCRCatalogs.set_root_dir('/global/cfs/projectdirs/lsst/shared') 

def get_1to1_group(results, n_truth, n_object, data_meas, truth_table, redshift_max=1.2):
    one_to_one_group_mask = np.in1d(results['group_id'], np.flatnonzero((n_truth == 1) & (n_object == 1)))
    truth_idx = results['row_index'][one_to_one_group_mask & truth_mask]
    object_idx = results['row_index'][one_to_one_group_mask & object_mask]
    mask = truth_table['redshift'][truth_idx] < redshift_max
    truth_1_to_1 = truth_table[truth_idx][mask]
    object_1_to_1 = astropy.table.Table(data_meas)[object_idx][mask]
    tab_out = astropy.table.hstack(truth_1_to_1, object_1_to_1, table_names=['truth', 'object'])
    return tab_out

def get_multipleto1_group(results, n_truth, n_object, data_meas, truth_table):
    multiple_to_one_group = np.in1d(results['group_id'], np.flatnonzero((n_truth > 1) & (n_object == 1)))
    truth_idx = results['row_index'][multiple_to_one_group & truth_mask]
    object_idx = results['row_index'][multiple_to_one_group & object_mask]
    truth_multiple_to_1 = truth_table[truth_idx]
    truth_multiple_to_1['group_id'] = results['group_id'][multiple_to_one_group & truth_mask]
    object_multiple_to_1 = astropy.table.Table(data_meas)[object_idx]
    object_multiple_to_1['group_id'] = results['group_id'][multiple_to_one_group & object_mask]
    return truth_multiple_to_1, object_multiple_to_1

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
    if os.path.isfile(f'/global/cscratch1/sd/jsanch87/DC2_Run2.2i/FoFMatching/matched_1to1_{args.data_catalog}_{tract}.fits.gz'):
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
        results = FoFCatalogMatching.match(
            catalog_dict={'truth': truth_table, 'object': data_meas},
            linking_lengths=1.0,
            catalog_len_getter=lambda x: len(x['ra']))
        truth_mask = results['catalog_key'] == 'truth'
        object_mask = ~truth_mask
        # then np.bincount will give up the number of id occurrences (like historgram but with integer input)
        n_groups = results['group_id'].max() + 1
        n_truth = np.bincount(results['group_id'][truth_mask], minlength=n_groups)
        n_object = np.bincount(results['group_id'][object_mask], minlength=n_groups)
        one_to_one_tab = get_1to1_group(results, n_truth, n_object, data_meas, truth_table)
        one_to_one_tab.write(f'/global/cscratch1/sd/jsanch87/DC2_Run2.2i/FoFMatching/matched_1to1_{args.data_catalog}_{tract}.fits.gz', overwrite=True)
        truth_multiple_to_1, object_multiple_to_1 = get_multipleto1_group(results, n_truth, n_object, data_meas, truth_table)
        print('Multiple to 1 entries truth', len(truth_multiple_to_1))
        print('Multiple to 1 entries object:', len(object_multiple_to_1))
        truth_multiple_to_1.write(f'/global/cscratch1/sd/jsanch87/DC2_Run2.2i/FoFMatching/matched_truth_multipleto1_{args.data_catalog}_{tract}.fits.gz', overwrite=True)
        object_multiple_to_1.write(f'/global/cscratch1/sd/jsanch87/DC2_Run2.2i/FoFMatching/matched_object_multipleto1_{args.data_catalog}_{tract}.fits.gz', overwrite=True)
