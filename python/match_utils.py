import numpy as np
from sklearn.neighbors import KDTree


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
    n_neigh: Number of neighbors in the search radius
    """
    X = np.zeros((len(ra_true),2))
    X[:,0] = ra_true
    X[:,1] = dec_true
    tree = KDTree(X,metric='euclidean')
    Y = np.zeros((len(ra_data),2))
    Y[:,0] = ra_data
    Y[:,1] = dec_data
    ind,dist= tree.query_radius(Y,r=rmax*0.2/3600,return_distance=True)
    matched = np.zeros(len(ind),dtype=bool)
    ids = np.zeros(len(ind),dtype=true_id.dtype)
    dist_out = np.zeros(len(ind))
    n_neigh = np.zeros(len(ind), dtype=np.int32)
    for i, ilist in enumerate(ind):
        if len(ilist)>0:
            dmag = np.fabs(mag_true[ilist]-mag_data[i])
            good_ind = np.argmin(dmag)
            ids[i]=true_id[ilist[good_ind]]
            dist_out[i]=dist[i][good_ind]
            n_neigh[i] = len(ilist)
            if np.min(dmag)<max_deltamag:
                matched[i]=True
            else:
                matched[i]=False
        else:
            ids[i]=-99
            matched[i]=False
            dist_out[i]=-99.
    return dist_out*3600., ids, matched, n_neigh


