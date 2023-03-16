import numpy as np
import torch
from e3nn import o3
import healpy as hp


def compute_trace(rotA, rotB):
    '''
    rotA, rotB are tensors of shape (*,3,3)
    returns Tr(rotA @ rotB.T)
    '''
    prod = torch.matmul(rotA, rotB.transpose(-1, -2))
    trace = prod.diagonal(dim1=-1, dim2=-2).sum(-1)
    return trace


def rotation_error(rotA, rotB):
    '''
    rotA, rotB are tensors of shape (*,3,3)
    returns error in radians, tensor of shape (*)
    '''
    trace = compute_trace(rotA, rotB)
    return torch.arccos(torch.clamp( (trace - 1)/2, -1, 1))


def nearest_rotmat(src, target):
    '''return index of target that is nearest to each element in src
    uses negative trace of the dot product to avoid arccos operation

    :src: tensor of shape (B, 3, 3)
    :target: tensor of shape (*, 3, 3)
    '''
    trace = compute_trace(src.unsqueeze(1), target.unsqueeze(0))

    return torch.max(trace, dim=1)[1]


def s2_near_identity_grid(max_beta=np.pi / 8, n_alpha=8, n_beta=3):
    """
    :return: rings around the north pole
    size of the kernel = n_alpha * n_beta
    """
    beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
    alpha = torch.linspace(0, 2 * np.pi, n_alpha + 1)[:-1]
    a, b = torch.meshgrid(alpha, beta, indexing="ij")
    b = b.flatten()
    a = a.flatten()
    return torch.stack((a, b))


def so3_near_identity_grid(max_beta=np.pi / 8, max_gamma=2 * np.pi, n_alpha=8, n_beta=3, n_gamma=None):
    """
    :return: rings of rotations around the identity, all points (rotations) in
    a ring are at the same distance from the identity
    size of the kernel = n_alpha * n_beta * n_gamma
    """
    if n_gamma is None:
        n_gamma = n_alpha  # similar to regular representations
    beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
    alpha = torch.linspace(0, 2 * np.pi, n_alpha)[:-1]
    pre_gamma = torch.linspace(-max_gamma, max_gamma, n_gamma)
    A, B, preC = torch.meshgrid(alpha, beta, pre_gamma, indexing="ij")
    C = preC - A
    A = A.flatten()
    B = B.flatten()
    C = C.flatten()
    return torch.stack((A, B, C))


def s2_healpix_grid(rec_level: int=0, max_beta: float=np.pi/6):
    """Returns healpix grid up to a max_beta
    """
    n_side = 2**rec_level
    npix = hp.nside2npix(n_side)
    m = hp.query_disc(nside=n_side, vec=(0,0,1), radius=max_beta)
    # print(f'nside: {nside} -> npix: {npix} -> n_in_disc: {len(m)}')
    beta, alpha = hp.pix2ang(n_side, m)
    alpha = torch.from_numpy(alpha)
    beta = torch.from_numpy(beta)
    return torch.stack((alpha, beta)).float()


def so3_healpix_grid(rec_level: int=3):
    """Returns healpix grid over so3
    https://github.com/google-research/google-research/blob/4808a726f4b126ea38d49cdd152a6bb5d42efdf0/implicit_pdf/models.py#L272

    alpha: 0-2pi around Y
    beta: 0-pi around X
    gamma: 0-2pi around Y

    rec_level | num_points | bin width (deg)
    ----------------------------------------
         0    |         72 |    60
         1    |        576 |    30
         2    |       4608 |    15
         3    |      36864 |    7.5
         4    |     294912 |    3.75
         5    |    2359296 |    1.875

    :return: tensor of shape (3,npix)
    """
    n_side = 2**rec_level
    npix = hp.nside2npix(n_side)
    beta, alpha = hp.pix2ang(n_side, torch.arange(npix))
    gamma = torch.linspace(0, 2*np.pi, 6*n_side + 1)[:-1]

    alpha = alpha.repeat(len(gamma))
    beta = beta.repeat(len(gamma))
    gamma = torch.repeat_interleave(gamma, npix)
    return torch.stack((alpha, beta, gamma)).float()


def s2_irreps(lmax):
    return o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])


def so3_irreps(lmax):
    return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])

def wigner_D(l, alpha, beta, gamma):
    '''o3 version but handles cuda tensors'''
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    alpha = alpha[..., None, None] % (2 * np.pi)
    beta = beta[..., None, None] % (2 * np.pi)
    gamma = gamma[..., None, None] % (2 * np.pi)
    X = o3.so3_generators(l).to(alpha.device)
    return torch.matrix_exp(alpha * X[1]) @ torch.matrix_exp(beta * X[0]) @ torch.matrix_exp(gamma * X[1])


def flat_wigner(lmax, alpha, beta, gamma):
    return torch.cat([(2 * l + 1) ** 0.5 * wigner_D(l, alpha, beta, gamma).flatten(-2) for l in range(lmax + 1)], dim=-1)


def rotate_s2(s2_signal, alpha=0, beta=0, gamma=0):
    '''alpha beta gamma in radians'''
    lmax = int(s2_signal.shape[-1]**0.5) - 1
    irreps = s2_irreps(lmax)
    alpha = torch.tensor(alpha, dtype=torch.float32)
    beta = torch.tensor(beta, dtype=torch.float32)
    gamma = torch.tensor(gamma, dtype=torch.float32)
    return torch.einsum("ij,...j->...i",
                        irreps.D_from_angles(alpha, beta, gamma),
                        s2_signal)

if __name__ == "__main__":
    # test gradient ascent
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    from scipy.spatial.transform import Rotation

    lmax = 8
    eval_xyx = so3_healpix_grid(rec_level=3)
    eval_wigners = flat_wigner(lmax, *eval_xyx).transpose(0,1)
    eval_rotmats = o3.angles_to_matrix(*eval_xyx)

    def grid_max(signal):
        return eval_rotmats[torch.max(signal.squeeze(1), dim=1)[1]]

    def ascent_max(fourier, signal, lr=1e-3, n_iters=20):
        rot = eval_rotmats[torch.max(signal.squeeze(1), dim=1)[1]]
        a,b,g = o3.matrix_to_angles(rot)
        a.requires_grad = True
        b.requires_grad = True
        g.requires_grad = True

        for _ in range(n_iters):
            wigners = flat_wigner(lmax, a,b,g).transpose(0,1)
            val = torch.matmul(fourier, wigners)
            da, db, dg = torch.autograd.grad(val.mean(), (a, b, g))
            a = a + lr * da
            b = b + lr * db
            g = g + lr * dg
        rot = o3.angles_to_matrix(a, b, g)
        return rot

    B = 64
    grid_err = []
    ascent_err = []
    for _ in range(10):
        rots = torch.from_numpy(Rotation.random(500).as_matrix()).float()
        angles = o3.matrix_to_angles(rots)
        D = flat_wigner(lmax, *angles)

        # convert
        weights = torch.zeros((B,1,len(rots)),dtype=torch.float32)
        weights[torch.arange(B),0,torch.arange(B)] = 100
        # print('D', D.shape)
        # print('weights', weights.shape)
        fourier = torch.einsum('ni,xyn->xyi', D, weights) / D.shape[0]**0.5
        signal = torch.matmul(fourier, eval_wigners)

        gt = rots[torch.arange(B)]
        grid_max_rot = grid_max(signal)
        ascent_max_rot = ascent_max(fourier, signal)

        grid_err.append(rotation_error(gt, grid_max_rot))
        ascent_err.append(rotation_error(gt, ascent_max_rot))

    grid_err = torch.cat(grid_err)
    ascent_err = torch.cat(ascent_err)
    print('grid_error', torch.median(grid_err).item()*180/np.pi)
    print('ga_error', torch.median(ascent_err).item()*180/np.pi)
