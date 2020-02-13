from ngboost.distns import RegressionDistn
from ngboost.scores import LogScore, CRPScore
from statsmodels.discrete.discrete_model import NegativeBinomial as dist


class NegativeBinomialLogScore(LogScore):
    """
    The log scoring rule is the same as negative log-likelihood: -log(P̂(y)),
    also known as the maximum likelihood estimator. This rule currently uses
    the inherited method for calculating the Riemannian metric.
    """

    def score(self, Y):
        """The value of the score at the current parameters, given the data Y"""
        return self.dist.score(Y)

    def d_score(self, Y):
        """The derivative of the score at the current parameters, given the data Y"""
        pass

    def metric(self):
        """The value of the Riemannian metric at the current parameters"""
        return super().metric()


class NegativeBinomial(RegressionDistn):
    """
    Implements the negative binomial distribution for NGBoost.

    The negative binomial distribution has two parameters:
        n: the number of successes, and
        p: the probability of a single success.
    See scipy.stats.nbinom for more information.
    This distribution currently has LogScore implemented for it.
    """

    n_params = 2
    scores = [NegativeBinomialLogScore]

    def __init__(self, params):
        super().__init__(params)
        self.n = params[0]
        self.p = params[1]
        self.dist = dist([self.n, self.p])

    def fit(Y):
        return self.dist.fit(Y)

    def sample(self, m):
        pass

    def __getattr__(self, name):
        """Gives us NegativeBinomial.mean() from ???."""
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {'n': self.n, 'p': self.p}
