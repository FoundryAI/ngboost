import numpy as np
import jax.numpy as jnp
from jax import grad as jaxgrad

class MLE:

    @staticmethod
    def loss(forecast, Y, sample_weight=None):
        return np.average(forecast.nll(Y.squeeze()), weights=sample_weight)

    @staticmethod
    def grad(forecast, Y, natural=True, n_mc_fisher=100):
        if 'D_nll' in dir(forecast):
            grad_fn = lambda Y: forecast.D_nll(Y)
        else:
            g = jaxgrad(lambda θ,y: jnp.sum(forecast.__class__.nll_jax(θ,y)))
            grad_fn = lambda Y: g(forecast.params_, Y).T
        grad = grad_fn(Y)
        if natural:
            if 'fisher_info' in dir(forecast):
                fisher = forecast.fisher_info()
            else:
                grads = np.stack([grad_fn(Y) for Y in forecast.sample(n_mc_fisher).T])
                fisher = np.mean(np.einsum('sik,sij->sijk', grads, grads), axis=0)
            grad = np.linalg.solve(fisher, grad)
        return grad


class CRPS:

    @staticmethod
    def loss(forecast, Y, sample_weight=None):
        return np.average(forecast.crps(Y.squeeze()), weights=sample_weight)

    @staticmethod
    def grad(forecast, Y, natural=True):
        grad = forecast.D_crps(Y)
        if natural:
            metric = forecast.crps_metric()
            grad = np.linalg.solve(metric, grad)
        return grad
