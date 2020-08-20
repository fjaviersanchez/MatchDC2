import numpy as np
from sklearn.neighbors import KDTree, DistanceMetric, BallTree
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
    
def spatial_closest_mag_1band(ra_data,dec_data,mag_data,
                              ra_true,dec_true,mag_true,true_id,
                              rmax=3,max_deltamag=1.):
    """
    Function to return the closest match in magnitude within a user-defined radius within certain
    magnitude difference.
    
    ***Caveats***: This method uses small angle approximation sin(theta)
    ~ theta for the declination axis. This should be fine to find the closest
    neighbor. This method does not use any weighting.
    
    Args:
    -----
    
    ra_data: Right ascension of the measured objects (degrees).
    dec_data: Declination of the measured objects (degrees).
    mag_data: Measured magnitude of the objects.
    ra_true: Right ascension of the true catalog (degrees).
    dec_true: Declination of the true catalog (degrees).
    mag_true: True magnitude of the true catalog.
    true_id: Array of IDs in the true catalog.
    rmax: Maximum distance in number of pixels to perform the query.
    max_deltamag: Maximum magnitude difference for the match to be good.
    
    Returns:
    --------
    
    dist: Distance to the closest neighbor in the true catalog. If inputs are
    in degrees, the returned distance is in arcseconds.
    true_id: ID in the true catalog for the closest match.
    matched: True if matched, False if not matched.
    n_neigh_x: Number of neighbors in the search radius in the true catalog
    n_neigh_y: Number of neighbors in the search radius in the object catalog
    """

    X = np.zeros((len(ra_true),2))
    X[:,0] = ra_true
    X[:,1] = dec_true
    tree = KDTree(X, metric='euclidean')
    rmax = rmax*0.2/3600
    Y = np.zeros((len(ra_data),2))
    Y[:,0] = ra_data
    Y[:,1] = dec_data
    ind, dist= tree.query_radius(Y, r=rmax, return_distance=True)
    tree_obj = KDTree(Y, metric='euclidean')
    n_neigh_y = tree_obj.query_radius(Y, r=rmax, count_only=True)
    matched = np.zeros(len(ind),dtype=bool)
    ids = np.zeros(len(ind),dtype=true_id.dtype)
    dist_out = np.zeros(len(ind))
    n_neigh_x = np.zeros(len(ind), dtype=np.int32)
    for i, ilist in enumerate(ind):
        if len(ilist)>0:
            dmag = np.fabs(mag_true[ilist]-mag_data[i])
            good_ind = np.argmin(dmag)
            ids[i]=true_id[ilist[good_ind]]
            dist_out[i]=dist[i][good_ind]
            n_neigh_x[i] = len(ilist)
            if np.min(dmag)<max_deltamag:
                matched[i]=True
            else:
                matched[i]=False
        else:
            ids[i]=-99
            matched[i]=False
            dist_out[i]=-99.
    return dist_out*3600., ids, matched, n_neigh_y.flatten(), n_neigh_x.flatten()

def spatial_closest_mag_1band_3d(ra_data, dec_data, mag_data,
                              ra_true, dec_true, mag_true, true_id,
                              rmax=3, max_deltamag=1.):
    """
    Function to return the closest match in magnitude within a user-defined radius within certain
    magnitude difference.
    
    ***Caveats***: This method uses small angle approximation sin(theta)
    ~ theta for the declination axis. This should be fine to find the closest
    neighbor. This method does not use any weighting.
    
    Args:
    -----
    
    ra_data: Right ascension of the measured objects (degrees).
    dec_data: Declination of the measured objects (degrees).
    mag_data: Measured magnitude of the objects.
    ra_true: Right ascension of the true catalog (degrees).
    dec_true: Declination of the true catalog (degrees).
    mag_true: True magnitude of the true catalog.
    true_id: Array of IDs in the true catalog.
    rmax: Maximum distance in number of pixels to perform the query.
    max_deltamag: Maximum magnitude difference for the match to be good.
    
    Returns:
    --------
    
    dist: Distance to the closest neighbor in the true catalog. If inputs are
    in degrees, the returned distance is in arcseconds.
    true_id: ID in the true catalog for the closest match.
    matched: True if matched, False if not matched.
    n_neigh_x: Number of neighbors in the search radius in the true catalog
    n_neigh_y: Number of neighbors in the search radius in the object catalog
    """
    rmax = np.radians(rmax*0.2/3600)
    X = np.zeros((len(ra_true),2))
    X[:,0] = np.radians(dec_true) # Haversine distance assumes first coordinate to be the latitude
    X[:,1] = np.radians(ra_true)
    tree = BallTree(X, metric='haversine')
    Y = np.zeros((len(ra_data),2))
    Y[:,0] = np.radians(dec_data)
    Y[:,1] = np.radians(ra_data)
    # Center search at Y and look for points in X
    ind, dist= tree.query_radius(Y, r=rmax, return_distance=True)
    tree_obj = BallTree(Y, metric='haversine')
    # Center search at Y and look for points in Y
    n_neigh_y = tree_obj.query_radius(Y, r=rmax, count_only=True)
    matched = np.zeros(len(ind),dtype=bool)
    ids = np.zeros(len(ind),dtype=true_id.dtype)
    dist_out = np.zeros(len(ind))
    n_neigh_x = np.zeros(len(ind), dtype=np.int32)
    for i, ilist in enumerate(ind):
        if len(ilist)>0:
            dmag = np.fabs(mag_true[ilist]-mag_data[i])
            good_ind = np.argmin(dmag)
            ids[i]=true_id[ilist[good_ind]]
            dist_out[i]=dist[i][good_ind]
            n_neigh_x[i] = len(ilist)
            if np.min(dmag)<max_deltamag:
                matched[i]=True
            else:
                matched[i]=False
        else:
            ids[i]=-99
            matched[i]=False
            dist_out[i]=-99.
    return dist_out*3600., ids, matched, n_neigh_y.flatten(), n_neigh_x.flatten()
