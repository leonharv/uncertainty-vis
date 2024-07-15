import numpy as np
import tensorflow as tf
from scipy import sparse
from scipy.interpolate import CubicSpline
from skimage import transform
from error_propagation_radon_transform import utils

import concurrent.futures

import argparse
import sys
import os
import logging
logger = logging.getLogger(__name__)

def sliceRSP(rsp, z=0, scale=1):
    phantom = np.pad(rsp[:,:,z], ((30*scale,30*scale), (100*scale, 100*scale)))
    RSP_shape = phantom.shape[:2]
    return phantom, RSP_shape

def get_mlp(spline, phi, spotx=0, shape=(130, 1026), chord_length=True):
    '''
    Generate a MLP.
    '''
    MLP_prototype = sparse.lil_matrix(shape)
    
    rotation = np.array([
        [np.cos(-phi), -np.sin(-phi)],
        [np.sin(-phi), np.cos(-phi)]
    ]).T
      
    z_start = -81
    dz = ( 2 * 81 )/shape[1]/2
    box_edge = 96 # x direction
    
    # draw the spline
    i = np.arange(-600,shape[1]+1600)
    z_mlp = i * dz + z_start
    x_mlp = spline(z_mlp)
    # rotate
    point_mlp = (rotation @ np.array([z_mlp, x_mlp])).T
    
    i_rotated = np.int64((point_mlp[:,0] - z_start) / dz/2)
    j_rotated = np.int64((point_mlp[:,1] + box_edge) * shape[0] / (2 * box_edge) )
    
    valid_i = (i_rotated >= 0) & (i_rotated < shape[1])
    valid_j = (j_rotated >= 0) & (j_rotated < shape[0])
    
    valid = valid_i & valid_j
    
    MLP_prototype[ j_rotated[valid], i_rotated[valid] ] = 1
            
    if chord_length:
        points = np.array([z_mlp[valid], x_mlp[valid]]).transpose()
        dx_dz = spline(points[:,0], 1)
        
        length = np.sqrt( dz**2 + dx_dz**2).sum()
                
        return MLP_prototype / length
    else:
        return MLP_prototype
    
def worker_spotx_future(idx, cs, phi, spotx, target_shape, chord_length):
    '''
    Worker function for concurrent executions
    '''
    return get_mlp(cs, phi, spotx, shape=target_shape, chord_length=chord_length).reshape((1,-1))

def bulkMLP_concurrent(num_angle, num_offset, num_spotx, chord_length, target_shape, max_workers=32):
    '''
    Generate many MLP based on the defined parameters.

    Parameters
    ----------
    num_angle: int
        How many angles should be generated in the range of [0,180).
    num_offset: int
        How many offsets (bending of the MLP) should be generated. If 1, no curved MLP will be generated.
    num_spotx: int
        How many parrel beams should be generated.
    chord_length: bool
        If True, the chord length will be used as values for the MLP, otherwise only 1 is set.
    target_shape: array_like
        The shape of the MLP.
    max_workers: int
        How many workers should be generated in parallel.

    Returns
    -------
    sparse.lil_matrix
        A matrix containing all MLP.
    '''
    angles = -np.linspace(0,180,num_angle) * np.pi / 180
    if num_offset == 1:
        offsets = [0]
    else:
        offsets = np.linspace(-15, 15, num_offset)
    spotxs = np.linspace(-95, 95, num_spotx)

    logger.debug('Allocating MLP matrix...')
    MLP_angles_offsets_spotx = sparse.lil_matrix((len(angles) * len(offsets) * len(spotxs), np.prod(target_shape)))
    logger.debug('Finished allocating MLP matrix.')

    future_to_idx = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers = max_workers) as executor:
        for odx,spotx in enumerate(spotxs):
            for xdx, offset in enumerate(offsets):
            
                points = np.array([
                    [-500,spotx], # beam source
                    [-400,spotx],
                    [-300,spotx],
                    [-200,spotx],
                    [-100,spotx],
                    [-81,spotx], # enter phantom

                    [81,offset + spotx] # leave phantom
                ])

                cs = CubicSpline(points[:,0], points[:,1])

                for adx,phi in enumerate(angles):
                    idx = odx * len(angles)*len(offsets) + xdx * len(angles) + adx

                    future = executor.submit(worker_spotx_future, idx, cs, phi, spotx, target_shape, chord_length)
                    future_to_idx[future] = idx
                
    for future in concurrent.futures.as_completed(future_to_idx):
        idx = future_to_idx[future]
        try:
            MLP_angles_offsets_spotx[idx,:] = future.result()
        except Exception as exc:
                print('%r generated an exception: %s' % (idx, exc))

    return MLP_angles_offsets_spotx

def randomizeMLPs(MLP, num_angle, num_offset, num_spotx, std_intervall=3):
    i1, _, i3 = np.indices((num_spotx, num_offset, num_angle))

    rng = np.random.default_rng()
    # mean of 15 and 99% are within 3 * std = 15
    i2 = rng.normal(num_offset // 2, (num_offset // 2) / std_intervall, (num_spotx, num_offset, num_angle)).astype(np.int64)
    i2[ i2 < 0 ] = 0
    i2[ i2 > num_offset - 1 ] = num_offset - 1

    idx = np.ravel_multi_index((i1, i2, i3), (num_spotx, num_offset, num_angle))

    return MLP[idx.flatten(), :]

def calcWEPL(RSP, MLP):
    wepl = MLP @ RSP.flatten()
    return wepl

def exponential(value, gamma):
    return np.exp( - gamma * value)

def main(num_angle=179, num_offset=1, num_spotx=190, chord_length=True, filter_name='ramp'):
    rsp = np.load('../../Data/simple_pCT/Phantoms/Head/RSP.npy')
    _, RSP_shape = sliceRSP(rsp, 0)

    width = 10
    chords = 'exact' if chord_length else 'map'
    max_workers = 16

    logger.info('''Parameters are:
                num_angle = {:d}
                num_offset = {:d}
                num_spotx = {:d}
                chord_length = {:s}
                filter_name = {:s}'''.format(
                    num_angle,
                    num_offset,
                    num_spotx,
                    chords,
                    filter_name
                ))
    
    logger.info('Start generating MLPs...')
    MLP_angles_offsets_spotx = bulkMLP_concurrent(num_angle, num_offset, num_spotx, chord_length, RSP_shape, max_workers)
    logger.info('Finished generating MLPs.')

    logger.info('Saving MLPs.')
    if chord_length:
        sparse.save_npz('../../Data/simple_pCT/MLP/MLP_angles{:d}_offset{:d}_spotx{:d}_exact_{:d}_{:d}.npz'.format(num_angle, num_offset, num_spotx, RSP_shape[0], RSP_shape[1]), MLP_angles_offsets_spotx.tocsc())
    else:
        sparse.save_npz('MLP_angles{:d}_offset{:d}_spotx{:d}_map_{:d}_{:d}.npz'.format(num_angle, num_offset, num_spotx, RSP_shape[0], RSP_shape[1]), MLP_angles_offsets_spotx.tocsc())

    logger.info('Loading jacobian.')
    jacobian = np.load('../../Data/simple_pCT/Jacobian/J_angles{:d}_offset{:d}_spotx{:d}_{:s}_{:s}_{:d}_{:d}.npy'.format(num_angle, num_offset, num_spotx, chords, filter_name, RSP_shape[0], RSP_shape[1]))
    l,m,n,o = jacobian.shape
    jacobian_reshaped = tf.reshape(jacobian, (l*m, n*o))

    theta = np.linspace(0., 180., num_angle, endpoint=False)
    steps = 8

    reconstructed = np.empty((num_spotx,num_spotx,rsp.shape[2]//steps))
    variance_out = np.empty((num_spotx,num_spotx,rsp.shape[2]//steps))
    logger.info('Start pipeline...')
    for z in range(0,rsp.shape[2],steps):
        logger.info('Start iteration {:d} of {:d} ...'.format(z//steps, rsp.shape[2]//steps))
        x, _ = sliceRSP(rsp, z)

        logger.info('Randomize MLPs...')
        MLPs_randomized = randomizeMLPs(MLP_angles_offsets_spotx, num_angle, num_offset, num_spotx)

        logger.info('Calculate WEPL...')
        wepl = calcWEPL(x, MLPs_randomized)
        wepl = np.reshape(wepl, (num_spotx, num_offset, num_angle))

        logger.info('Calculate Std...')
        wepl_mean = np.mean(wepl, axis=1)
        wepl_std = np.std(wepl, axis=1)

        logger.info('Build Covariance...')
        Sigma_in = utils.build_covariance_y(wepl_std**2, function=exponential, width=width)

        logger.info('Propagate error...')
        Sigma_out = jacobian_reshaped @ Sigma_in @ np.transpose(jacobian_reshaped)
        variance_out[:,:,z//steps] = tf.reshape(tf.abs(tf.linalg.tensor_diag_part(Sigma_out)), (num_spotx,num_spotx))

        logger.info('Reconstruct...')
        reconstructed[:,:,z//steps] = transform.iradon(wepl_mean, theta=theta, filter_name=filter_name, circle=True)

        # if not os.path.exists('../../Data/simple_pCT/Sigma/{:d}'.format(num_angle)):
        #     os.makedirs('../../Data/simple_pCT/Sigma/{:d}'.format(num_angle))

        # np.save('../../Data/simple_pCT/Sigma/{:d}/Sigma_raedler_angles{:d}_offset{:d}_spotx{:d}_{:s}_{:s}_{:d}_{:d}_z{:d}.npy'.format(num_angle, num_angle, num_offset, num_spotx, chords, filter_name, RSP_shape[0], RSP_shape[1], z), Sigma_out)
        
    logger.info('saving variance.')
    np.save('../../Data/simple_pCT/Variance/Variance_raedler_angles{:d}_offset{:d}_spotx{:d}_{:s}_{:s}_{:d}_{:d}.npy'.format(num_angle, num_offset, num_spotx, chords, filter_name, RSP_shape[0], RSP_shape[1]), variance_out)

    logger.info('Saving reconstruction.')
    np.save('../../Data/simple_pCT/Reconstruction/Head/{:d}_{:d}/3D/RSP_angles{:d}_offset{:d}_spotx{:d}_{:s}_{:s}.npy'.format(RSP_shape[0], RSP_shape[1], num_angle, num_offset, num_spotx, chords, filter_name), reconstructed)

    logger.info('Finished pipeline.')

if __name__ == '__main__':
    logging.basicConfig(filename='one-shot-creation-uncertain-wepls.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser(description='Create MLPs, WEPLs, Jacobian, Reconstruction and Sigma in one shot.')
    parser.add_argument('num_angle', default=178, type=int, help='Number of angles')
    parser.add_argument('num_offset', default=1, type=int, help='Number of offsets')
    parser.add_argument('num_spotx', default=130, type=int, help='Number of spotx')
    parser.add_argument('filter_name', default='ramp', choices=['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann'], help='Filter')

    args = parser.parse_args()

    logger.info('Started')
    main(num_angle=args.num_angle, num_offset=args.num_offset, num_spotx=args.num_spotx, filter_name=args.filter_name)
    logger.info('Finished')