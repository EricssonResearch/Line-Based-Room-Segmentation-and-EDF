""" This file contains code from the GitHub repository https://github.com/wjmaddox/online_gp """

from gpedf_utils.streaming_sgpr import StreamingSGPR
import torch
import gpytorch

torch.set_default_dtype(torch.double)

class OnlineSGPRegression(torch.nn.Module):
    def __init__(
            self,
            covar_module=None,
            inducing_points=None,
            mean_module = None,
            learn_inducing_locations=True,
            jitter=1e-4, 
            inducing_thresh=0.5e-6):
        
        super().__init__()

        if covar_module is not None:
            self.gp = StreamingSGPR(inducing_points, learn_inducing_locations=learn_inducing_locations,
                                covar_module=covar_module, mean_module=mean_module, 
                                jitter=jitter, inducing_thresh=inducing_thresh)
            if torch.cuda.is_available():
                self.gp = self.gp.cuda()
        else:
            self.gp = None
            
    def predict(self, inputs):
        with gpytorch.settings.fast_pred_var():
            pred_dist = self.gp.likelihood(self.gp(inputs))
        return pred_dist.mean 

    def update(self, inputs, targets):
        self.gp = self.gp.get_fantasy_model(inputs, targets)