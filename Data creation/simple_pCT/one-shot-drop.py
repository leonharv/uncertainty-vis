import numpy as np
from scipy import sparse
from scipy.interpolate import CubicSpline
from skimage import transform
from skimage.transform import resize

import concurrent.futures

import argparse
import sys
import os
import logging
logger = logging.getLogger(__name__)

def sliceRSP(rsp, z=0, scale=1, target_shape=None):
    phantom = np.pad(rsp[:,:,z], ((30*scale,30*scale), (100*scale, 100*scale)))
    RSP_shape = phantom.shape[:2]

    if target_shape:
        phantom = resize( phantom, target_shape, anti_aliasing=True )

    return phantom, RSP_shape

def randomizeMLPs(MLP, num_angle, num_offset, num_spotx, numberOfSamples, std_intervall=3):
    i1, _, i3 = np.indices((num_spotx, numberOfSamples, num_angle))

    rng = np.random.default_rng()
    # mean of 15 and 99% are within 3 * std = 15
    i2 = rng.normal(num_offset // 2, (num_offset // 2) / std_intervall, (num_spotx, numberOfSamples, num_angle)).astype(np.int64)
    i2[ i2 < 0 ] = 0
    i2[ i2 > num_offset - 1 ] = num_offset - 1

    idx = np.ravel_multi_index((i1, i2, i3), (num_spotx, num_offset, num_angle))

    return MLP[idx.flatten(), :]

def calcWEPL(RSP, MLP):
    wepl = MLP @ RSP.flatten()
    return wepl

def getObjectHull(RSP_shape):
    obj = np.zeros(RSP_shape, dtype='bool')
    for i in range(RSP_shape[0]):
        for j in range(RSP_shape[1]):
            # a circle is used as a valid volume
            vec = np.array([i, j])
            center = np.array(RSP_shape) // 2 
            
            obj[i,j] = True if (i - center[0])**2 / center[0]**2 + (j - center[1])**2 / center[1]**2 <= 1 else False

    return obj.flatten()

def getStraightMLPs(MLPs, num_angle, num_offset, num_spotx):
    i1, _, i3 = np.indices((num_spotx, 1, num_angle))

    # the middle mlp (num_offset // 2) is always a straight line
    idx = np.ravel_multi_index((i1, num_offset//2, i3), (num_spotx, num_offset, num_angle))
    return MLPs[idx.flatten(), :]

def prepareDROP(MLPs):
    A_norm = sparse.linalg.norm(MLPs, axis=1)**2
    valid = A_norm > 1e-9
    A_norm = A_norm[valid]

    S_array = np.array([MLPs[:,i].count_nonzero() for i in range(MLPs.shape[1])])
    S_array[ S_array <= 0 ] = 1
    S_array = 1 / S_array

    return A_norm, S_array, valid

def DROP(A, b, A_norm, S_lamb, obj, mlp_shape, num_iterations):
    x = np.random.random(np.prod(mlp_shape))

    for iter in range(num_iterations):
        Ax = A @ x
        residuals = b - Ax
        factor = residuals / A_norm
        x += S_lamb * (A.T @ factor)
        x[~obj] = 0

    return x

def main(num_angle=179, num_offset=1, num_spotx=190, numberOfSamples=50, chord_length=True, mlp_shape=(130, 130), lamb=0.5, num_iterations=2000):
    rsp = np.load('../../Data/simple_pCT/Phantoms/Head/RSP.npy')

    chords = 'exact' if chord_length else 'map'

    logger.info(f'''Parameters are:
                num_angle = {num_angle}
                num_offset = {num_offset}
                num_spotx = {num_spotx}
                chord_length = {chords}
                mlp_shape = {mlp_shape}
                numberOfSamples = {numberOfSamples}
                lamb = {lamb}
                num_iterations = {num_iterations}''')
    
    # logger.info('Start generating MLPs...')
    # MLP_angles_offsets_spotx = bulkMLP_concurrent(num_angle, num_offset, num_spotx, chord_length, mlp_shape, max_workers)
    # logger.info('Finished generating MLPs.')

    # logger.info('Saving MLPs.')
    # if chord_length:
    #     sparse.save_npz('../../Data/simple_pCT/MLP/MLP_angles{:d}_offset{:d}_spotx{:d}_exact_{:d}_{:d}.npz'.format(num_angle, num_offset, num_spotx, mlp_shape[0], mlp_shape[1]), MLP_angles_offsets_spotx.tocsc())
    # else:
    #     sparse.save_npz('../../Data/simple_pCT/MLP/MLP_angles{:d}_offset{:d}_spotx{:d}_map_{:d}_{:d}.npz'.format(num_angle, num_offset, num_spotx, mlp_shape[0], mlp_shape[1]), MLP_angles_offsets_spotx.tocsc())

    logger.info('Loading MLPs.')
    MLP_angles_offsets_spotx = sparse.load_npz(f'../../Data/simple_pCT/MLP/MLP_angles{num_angle}_offset{num_offset}_spotx{num_spotx}_{chords}_{mlp_shape[0]}_{mlp_shape[1]}.npz')

    logger.info('Calculate object hull.')
    obj = getObjectHull(mlp_shape)

    logger.info('Create straight MLPs.')
    MLPs_straight = getStraightMLPs(MLP_angles_offsets_spotx, num_angle, num_offset, num_spotx)

    logger.info('Prepare DROP constants.')
    A_norm, S_array, valid = prepareDROP(MLPs_straight)
    S_lamb = S_array * lamb
    A = MLPs_straight[valid,:]

    steps = 8
    reconstructed = np.empty((num_spotx,num_spotx,rsp.shape[2]//steps))
    variance_out = np.empty((num_spotx,num_spotx,rsp.shape[2]//steps))

    logger.info('Start pipeline...')
    for z in range(0,rsp.shape[2]//steps):
        logger.info('Process slice {:d} of {:d} ...'.format(z, rsp.shape[2]//steps))
        x, _ = sliceRSP(rsp, z*steps, 1, mlp_shape)

        logger.info('Randomize MLPs...')
        MLPs_randomized = randomizeMLPs(MLP_angles_offsets_spotx, num_angle, num_offset, num_spotx, numberOfSamples)

        logger.info('Calculate WEPL...')
        wepl = calcWEPL(x, MLPs_randomized)
        wepl = np.reshape(wepl, (num_spotx, numberOfSamples, num_angle))

        logger.info('Starting DROP ensemble...')
        reconstructions = np.empty((numberOfSamples,np.prod(mlp_shape)))
        for idx in range(numberOfSamples):
            logger.info(f'Process enssemble {idx} of {numberOfSamples}...')
            b = wepl[:,idx,:].flatten()[valid]

            reconstructions[idx,:] = DROP(A, b, A_norm, S_lamb, obj, mlp_shape, num_iterations)

        logger.info('Finished DROP ensemble.')
        reconstructed[:,:,z] = np.mean(reconstructions, axis=0).reshape(mlp_shape)
        variance_out[:,:,z] = np.var(reconstructions, axis=0).reshape(mlp_shape)

    logger.info('saving variance.')
    np.save('../../Data/simple_pCT/Variance/Variance_ensemble_raedler_angles{:d}_offset{:d}_spotx{:d}_{:s}_{:s}_{:d}_{:d}.npy'.format(num_angle, num_offset, num_spotx, chords, filter_name, mlp_shape[0], mlp_shape[1]), variance_out)

    logger.info('Saving reconstruction.')
    np.save('../../Data/simple_pCT/Reconstruction/Head/{:d}_{:d}/3D/RSP_ensemble_angles{:d}_offset{:d}_spotx{:d}_{:s}_{:s}.npy'.format(mlp_shape[0], mlp_shape[1], num_angle, 1, num_spotx, chords, filter_name), reconstructed)

    logger.info('Finished pipeline.')

if __name__ == '__main__':
    logging.basicConfig(filename='one-shot-creation-uncertain-wepls.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser(description='Create MLPs, WEPLs, Jacobian, Reconstruction and Sigma in one shot.')
    parser.add_argument('--num_angle', default=178, type=int, help='Number of angles')
    parser.add_argument('--num_offset', default=1, type=int, help='Number of offsets')
    parser.add_argument('--num_spotx', default=130, type=int, help='Number of spotx')
    parser.add_argument('--num_samples', default=50, type=int, help='Number of samples taken for the ensemble')
    parser.add_argument('--chord_length', action='store_true', help='The chord length will be used instead of storing a 1 in the MLP.')
    parser.add_argument('--mlp_height', default=130, type=int, help='Height of the MLPs')
    parser.add_argument('--mlp_width', default=130, type=int, help='Width of the MLPs')
    parser.add_argument('--lamb', default=0.5, type=float, help='Convergence rate')
    parser.add_argument('--num_iterations', default=2000, type=int, help='Number of iterations for the DROP algorithm')

    args = parser.parse_args()

    logger.info('Started')
    main(num_angle=args.num_angle, num_offset=args.num_offset, num_spotx=args.num_spotx, numberOfSamples=args.num_samples, chord_length=args.chord_length, mlp_shape=(args.mlp_height, args.mlp_width), lamb=args.lamb, num_iterations=args.num_iterations)
    logger.info('Finished')