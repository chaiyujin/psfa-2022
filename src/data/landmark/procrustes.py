from typing import Union

import numpy as np
import torch


def procrustes_align(
    src_pts: Union[np.ndarray, torch.Tensor], dst_pts: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:

    return Procrustes.align(src_pts, dst_pts)


class Procrustes(object):
    def __call__(self, src_pts, dst_pts):
        return Procrustes.align(src_pts, dst_pts)

    @staticmethod
    def align(src_pts, dst_pts):
        assert src_pts.ndim == dst_pts.ndim, f"src_pts and dst_pts have different dim: {src_pts.ndim}, {dst_pts.ndim}"
        if src_pts.ndim == 2:
            if isinstance(src_pts, np.ndarray) and isinstance(dst_pts, np.ndarray):
                return Procrustes.align_npy(src_pts, dst_pts)
            elif torch.is_tensor(src_pts) and torch.is_tensor(dst_pts):
                return Procrustes.align_torch(src_pts, dst_pts)
            else:
                raise ValueError("src_pts, dst_pts have different type!")
        elif src_pts.ndim == 3:
            # print('batch_align_torch', src_pts.shape, dst_pts.shape)
            assert torch.is_tensor(src_pts) and torch.is_tensor(dst_pts)
            if dst_pts.shape[0] == 1 and src_pts.shape[0] != 1:
                dst_pts = dst_pts.repeat(src_pts.shape[0], 1, 1)
            return Procrustes.batch_align_torch(src_pts, dst_pts)
        else:
            raise ValueError("src_pts, dst_pts should be (N_Points, Dim) or (N_Batch, N_Points, Dim)!")

    @staticmethod
    def align_npy(S1: np.ndarray, S2: np.ndarray):
        """
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        """
        assert S1.ndim == 2 and S2.ndim == 2

        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.T
            S2 = S2.T
            transposed = True
        assert S2.shape == S1.shape
        assert S1.shape[-2] in [2, 3]

        # 1. Remove mean.
        mu1 = S1.mean(axis=1, keepdims=True)
        mu2 = S2.mean(axis=1, keepdims=True)
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = np.sum(X1**2)

        # 3. The outer product of X1 and X2.
        K = X1.dot(X2.T)

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, Vh = np.linalg.svd(K)
        V = Vh.T
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = np.eye(U.shape[0])
        Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
        # Construct R.
        R = V.dot(Z.dot(U.T))

        # 5. Recover scale.
        scale = np.trace(R.dot(K)) / var1

        # 6. Recover translation.
        t = mu2 - scale * (R.dot(mu1))

        # 7. Error:
        S1_hat = scale * R.dot(S1) + t

        if transposed:
            S1_hat = S1_hat.T

        return S1_hat

    @staticmethod
    def align_torch(S1: torch.Tensor, S2: torch.Tensor):
        """
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        """
        assert S1.ndim == 2 and S2.ndim == 2

        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.T
            S2 = S2.T
            transposed = True
        assert S2.shape == S1.shape
        assert S1.shape[-2] in [2, 3]

        # 1. Remove mean.
        mu1 = S1.mean(dim=1, keepdim=True)
        mu2 = S2.mean(dim=1, keepdim=True)
        X1 = S1 - mu1
        X2 = S2 - mu2

        # print('X1', X1.shape)

        # 2. Compute variance of X1 used for scale.
        var1 = torch.sum(X1**2)

        # print('var', var1.shape)

        # 3. The outer product of X1 and X2.
        K = X1.mm(X2.T)

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, V = torch.svd(K)
        # V = Vh.T
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[0], device=S1.device)
        Z[-1, -1] *= torch.sign(torch.det(U @ V.T))
        # Construct R.
        R = V.mm(Z.mm(U.T))

        # print('R', X1.shape)

        # 5. Recover scale.
        scale = torch.trace(R.mm(K)) / var1
        # print(R.shape, mu1.shape)
        # 6. Recover translation.
        t = mu2 - scale * (R.mm(mu1))
        # print(t.shape)

        # 7. Error:
        S1_hat = scale * R.mm(S1) + t

        if transposed:
            S1_hat = S1_hat.T

        return S1_hat.contiguous()

    @staticmethod
    def batch_align_torch(S1: torch.Tensor, S2: torch.Tensor):
        """
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        """
        assert S1.ndim == 3 and S2.ndim == 3

        transposed = False
        if S1.shape[1] != 3 and S1.shape[1] != 2:
            S1 = S1.permute(0, 2, 1)
            S2 = S2.permute(0, 2, 1)
            transposed = True
        assert S2.shape == S1.shape
        assert S1.shape[-2] in [2, 3]

        # 1. Remove mean.
        mu1 = S1.mean(dim=-1, keepdim=True)
        mu2 = S2.mean(dim=-1, keepdim=True)

        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = torch.sum(X1**2, dim=1).sum(dim=1)

        # 3. The outer product of X1 and X2.
        K = X1.bmm(X2.permute(0, 2, 1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0], 1, 1)
        Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

        # 5. Recover scale.
        scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

        # 6. Recover translation.
        t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

        # 7. Error:
        S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

        if transposed:
            S1_hat = S1_hat.permute(0, 2, 1)

        return S1_hat.contiguous()
