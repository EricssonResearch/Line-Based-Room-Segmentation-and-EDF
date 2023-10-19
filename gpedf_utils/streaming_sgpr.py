""" This file contains code from the GitHub repository https://github.com/wjmaddox/online_gp """

from gpytorch.models import ApproximateGP
from gpytorch import variational, means, kernels, likelihoods, distributions, lazy
import torch
from copy import deepcopy
from gpytorch.utils.cholesky import psd_safe_cholesky
import numpy as np

torch.set_default_dtype(torch.double)

class StreamingSGPR(ApproximateGP):
    """
    https://github.com/thangbui/streaming_sparse_gp/blob/b46e6e4a9257937f7ca26ac06099f5365c8b50d8/code/osgpr.py
    """
    def __init__(
            self,
            inducing_points,
            old_strat=None,
            old_kernel=None,
            old_C_matrix=None,
            covar_module=None,
            mean_module=None,
            likelihood=None,
            learn_inducing_locations=True,
            jitter=1e-4,
            inducing_thresh=0.5e-6
    ):
        data_dim = -2 if inducing_points.dim() > 1 else -1
        variational_distribution = variational.CholeskyVariationalDistribution(
            inducing_points.size(data_dim)
        )
        variational_strategy = variational.UnwhitenedVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)


        if mean_module is None:
            self.mean_module = means.ZeroMean() 
        else:
            self.mean_module = mean_module

        if covar_module is None:
            self.covar_module = kernels.MaternKernel()
        else:
            self.covar_module = covar_module

        if likelihood is None:
            self.likelihood = likelihoods.GaussianLikelihood()
        else:
            self.likelihood = likelihood

        self._old_strat = old_strat
        self._old_kernel = old_kernel
        self._old_C_matrix = old_C_matrix
        self._data_dim = data_dim
        self._jitter = jitter
        self.Kaa_old_inv_Kab = None
        self.inducing_thresh = inducing_thresh

    def forward(self, inputs):
        mean = self.mean_module(inputs)
        covar = self.covar_module(inputs)
        return distributions.MultivariateNormal(mean, covar)

    def current_C_matrix(self, x):
        sigma2 = self.likelihood.noise
        z_b = self.variational_strategy.inducing_points
        Kbf = self.covar_module(z_b, x).evaluate()
        C1 = Kbf @ Kbf.transpose(-1, -2) / sigma2

        if self._old_C_matrix is None:
            C2 = torch.zeros_like(C1)
        else:
            assert self._old_strat is not None
            assert self._old_kernel is not None
            z_a = self._old_strat.inducing_points.detach()
            Kaa_old = self._old_kernel(z_a).add_jitter(self._jitter).detach()
            C_old = self._old_C_matrix.detach()
            Kab = self.covar_module(z_a, z_b).evaluate()
            self.Kaa_old_inv_Kab = Kaa_old.inv_matmul(Kab)
            C2 = self.Kaa_old_inv_Kab.transpose(-1, -2) @ C_old.matmul(self.Kaa_old_inv_Kab)

        C = C1 + C2
        L = psd_safe_cholesky(C, upper=False, jitter=self._jitter)
        L = lazy.TriangularLazyTensor(L, upper=False)
        return lazy.CholLazyTensor(L, upper=False)

    def current_c_vec(self, x, y):
        sigma2 = self.likelihood.noise
        z_b = self.variational_strategy.inducing_points
        Kbf = self.covar_module(z_b, x).evaluate()
        c1 = Kbf @ y / sigma2

        if self._old_C_matrix is None:
            c2 = torch.zeros_like(c1)
            c3 = torch.zeros_like(c1)
        else:
            assert self._old_strat is not None
            assert self._old_kernel is not None
            z_a = self._old_strat.inducing_points.detach()
            ma = self._old_strat.variational_distribution.mean.detach().unsqueeze(-1)
            Kaa_old = self._old_kernel(z_a).add_jitter(self._jitter).detach()
            C_old = self._old_C_matrix.detach()
            Kab = self.covar_module(z_a, z_b).evaluate()
            Kaa_old_inv_ma = Kaa_old.inv_matmul(ma)
            Kba_Kaa_old_inv = self.Kaa_old_inv_Kab.transpose(-1, -2)
            c2 = Kab.transpose(-1, -2) @ Kaa_old_inv_ma
            c3 = Kba_Kaa_old_inv @ C_old.matmul(Kaa_old_inv_ma)

        return c1 + c2 + c3

    def _update_variational_moments(self, x, y):
        C = self.current_C_matrix(x)
        c = self.current_c_vec(x, y)
        z_b = self.variational_strategy.inducing_points
        Kbb = self.covar_module(z_b).evaluate()
        L = psd_safe_cholesky(Kbb + C.evaluate(), upper=False, jitter=self._jitter)
        m_b = Kbb @ torch.cholesky_solve(c, L, upper=False)
        S_b = Kbb @ torch.cholesky_solve(Kbb, L, upper=False)
        return m_b, S_b

    def update_variational_distribution(self, x_new, y_new):
        m_b, S_b = self._update_variational_moments(x_new, y_new)
        
        q_mean = self.variational_strategy._variational_distribution.variational_mean
        q_mean.data.copy_(m_b.squeeze(-1))

        upper_new_covar = psd_safe_cholesky(S_b, jitter=self._jitter)
        upper_q_covar = self.variational_strategy._variational_distribution.chol_variational_covar
        upper_q_covar.copy_(upper_new_covar)
        self.variational_strategy.variational_params_initialized.fill_(1)
    
    def select_new_inducing_points(self, x_new):
        """
        Adaptive Inducing Point Selection for new data points.
        
        Args:
        x_new_batch: new samples (batch of new data points)
        rho: acceptance threshold, 0 < rho < 1
        """
        # calculate the kernel function between x_new_batch and all inducing points
        # assuming covar_module is used as the kernel function
        if self.variational_strategy.inducing_points.numel() > 0:
            with torch.no_grad():
                z_old = self.variational_strategy.inducing_points.clone().detach()
                if type(self.mean_module) == LineSegMean and len(self.mean_module.room_graph.nodes[self.mean_module.room_label]["lines"]) > 0:
                    d_old = torch.from_numpy(self.mean_module.lineseg_dist(z_old))
                    selected_old_indices = torch.where(d_old > 0.1)[0]
                    z_old = z_old[selected_old_indices]
                if len(z_old) > 0:
                    perturbation = torch.empty_like(z_old)
                    
                    perturbation.uniform_(-1e-4, 1e-4)
                    z_old += perturbation
                    z_mask = torch.ones(z_old.size(0), dtype=bool)
                    z_old = z_old[z_mask]
                    kernel_values = self.covar_module(z_old, x_new).evaluate()
                    # get the maximum value for each new data point
                    d = torch.max(kernel_values, dim=0).values
                    # find the indices of new data points that have d < rho
                    selected_new_indices = torch.where(d < self.inducing_thresh)[0]
                    # if there are any selected data points, add them to the inducing points
                    x_new = x_new[selected_new_indices]
                if len(x_new) > 0:
                    z_new = torch.cat([z_old, x_new], dim=0)
                else:
                    z_new = z_old          
        else:
            z_new = x_new
        return z_new
            
        
    def get_fantasy_model(self, x_new, y_new, z_new=None, **kwargs):
        if z_new is None:
            z_new = self.select_new_inducing_points(x_new=x_new, **kwargs)

        fantasy_model = type(self)(
            inducing_points=z_new,
            likelihood=self.likelihood,
            mean_module=self.mean_module,
            covar_module=deepcopy(self.covar_module),
            old_strat=self.variational_strategy,
            old_kernel=self.covar_module,
            old_C_matrix=self.current_C_matrix(x_new),
            jitter=self._jitter,
            inducing_thresh=self.inducing_thresh
        )
        with torch.no_grad():
            fantasy_model.update_variational_distribution(x_new, y_new)
        return fantasy_model
    
    
class LineSegMean(means.Mean):
    def __init__(self, room_label, room_graph, line_graph, lamb=100):
        super().__init__()
        self.lamb = lamb
        self.room_label = room_label
        self.room_graph = room_graph
        self.line_graph = line_graph
        
    def lineseg_dist(self, x):
        endpoints = np.array([self.line_graph.nodes[line]["line"].endpoints for line in self.room_graph.nodes[self.room_label]["lines"]])
        # Make sure x and endpoints are 2D arrays
        x = np.atleast_2d(x)
        # Reshape x and endpoints to enable broadcasting
        x = x[:, np.newaxis, :] # Shape: (n_points, 1, 2)
        endpoints = endpoints[np.newaxis, :, :, :] # Shape: (1, n_segments, 2, 2)
        # Get the endpoints of the segments
        a = endpoints[:, :, 0, :] # Shape: (1, n_segments, 2)
        b = endpoints[:, :, 1, :] # Shape: (1, n_segments, 2)
        # vector from a to b
        ab = b - a # Shape: (1, n_segments, 2)
        # squared length of ab
        ab2 = np.sum(ab**2, axis=-1) # Shape: (1, n_segments)
        # vector from a to x
        ax = x - a # Shape: (n_points, n_segments, 2)
        # dot product of ab and ax
        abax = np.sum(ab * ax, axis=-1) # Shape: (n_points, n_segments)
        # normalized distance along ab
        t = np.clip(abax / ab2, 0.0, 1.0) # Shape: (n_points, n_segments)
        # vector from x to closest point on ab
        xt = ax - t[..., np.newaxis] * ab # Shape: (n_points, n_segments, 2)
        # squared distance from x to closest point on ab
        xt2 = np.sum(xt**2, axis=-1) # Shape: (n_points, n_segments)
        # Find the minimum distance along the second axis
        min_dists = np.sqrt(np.min(xt2, axis=1)) # Shape: (n_points,)
        return min_dists
    
    def forward(self, x):
        if x.size(0) == 0 or len(self.room_graph.nodes[self.room_label]["lines"]) == 0:
            return torch.empty(0)
        x_detached = x.detach()
        min_dist = self.lineseg_dist(x_detached)
        res = torch.exp(-torch.Tensor(min_dist)*self.lamb)
        return res


class CircularMean(means.Mean):
    def __init__(self, r=0, lengthscale=500):
        super().__init__()
        self.r2 = r**2
        self.lengthscale = lengthscale

    def forward(self, x):
        res = torch.exp(-(x[:,0]**2 + x[:,1]**2-self.r2)*self.lengthscale)#-x.matmul(self.weights).squeeze(-1)
        return res
    