# Author: Linus Bleistein <linus.bleistein@inria.fr>
#
# License: BSD 3-Clause

from abc import abstractmethod

import numpy as np
from sklearn.metrics import check_scoring
from sklearn.utils.metadata_routing import _MetadataRequester

from .utils import check_X_y_domain, extract_source_indices

# xxx(okachaiev): maybe it would be easier to reuse _BaseScorer?
# xxx(okachaiev): add proper __repr__/__str__
# xxx(okachaiev): support clone()


class _BaseDomainAwareScorer(_MetadataRequester):
    __metadata_request__score = {"sample_domain": True}

    @abstractmethod
    def _score(self, estimator, X, y, sample_domain=None, **params):
        pass

    def __call__(self, estimator, X, y=None, sample_domain=None, **params):
        return self._score(estimator, X, y, sample_domain=sample_domain, **params)


class SupervisedScorer(_BaseDomainAwareScorer):
    """Compute score on supervised dataset.

    Parameters
    ----------
    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If None, the provided estimator object's `score` method is used.
    greater_is_better : bool, default=True
        Whether `scorer` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `scorer`.
    """

    __metadata_request__score = {"target_labels": True}

    def __init__(self, scoring=None, greater_is_better=True):
        super().__init__()
        self.scoring = scoring
        self._sign = 1 if greater_is_better else -1

    def _score(
        self, estimator, X, y=None, sample_domain=None, target_labels=None, **params
    ):
        scorer = check_scoring(estimator, self.scoring)

        X, y, sample_domain = check_X_y_domain(X, y, sample_domain, allow_nd=True)
        source_idx = extract_source_indices(sample_domain)

        return self._sign * scorer(
            estimator,
            X[~source_idx],
            target_labels[~source_idx],
            sample_domain=sample_domain[~source_idx],
            **params,
        )


class DemographicParityDifferenceScorer(_BaseDomainAwareScorer):
    """Compute Demographic Parity score as the absolute difference.

    Parameters
    ----------
    weight_estimator : estimator, default=None
        An estimator to compute the weights for the samples.
        If None, the weights are set to 1.
    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, s)``.
    greater_is_better : bool, default=True
        Whether `scorer` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `scorer`.

    Returns
    -------
    dp_score : float
        The Demographic Parity score, which is the absolute difference
        between the average predicted probabilities for the two groups and 1.
        A score of 0 indicates perfect demographic parity, while higher values
        indicate greater disparity between the groups.
    """

    def __init__(
        self,
        weight_estimator=None,
        scoring=None,
        greater_is_better=False,
    ):
        super().__init__()
        self.weight_estimator = weight_estimator
        self.scoring = scoring
        self._sign = 1 if greater_is_better else -1

    def _score(self, estimator, X, s):
        proba_group_1 = estimator.predict_proba(X[s == 1])[0]
        proba_group_0 = estimator.predict_proba(X[s == 0])[0]

        difference = np.mean(proba_group_1, axis=0) - np.mean(proba_group_0, axis=0)

        return self._sign * np.abs(difference)


class DemographicParityRatioScorer(_BaseDomainAwareScorer):
    """Compute Demographic Parity score.

    Parameters
    ----------
    weight_estimator : estimator, default=None
        An estimator to compute the weights for the samples.
        If None, the weights are set to 1.
    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, s)``.
    greater_is_better : bool, default=True
        Whether `scorer` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `scorer`.

    Returns
    -------
    dp_score : float
        The Demographic Parity score, which is the absolute difference
        between the average predicted probabilities for the two groups and 1.
        A score of 0 indicates perfect demographic parity, while higher values
        indicate greater disparity between the groups.
    """

    def __init__(
        self,
        weight_estimator=None,
        scoring=None,
        greater_is_better=False,
    ):
        super().__init__()
        self.weight_estimator = weight_estimator
        self.scoring = scoring
        self._sign = 1 if greater_is_better else -1

    def _score(self, estimator, X, s):
        proba_group_1 = estimator.predict_proba(X[s == 1])[0]
        proba_group_0 = estimator.predict_proba(X[s == 0])[0]

        ratio = np.mean(proba_group_1, axis=0) / np.mean(proba_group_0, axis=0)

        return self._sign * np.abs(ratio - 1)


class TPParityDifferenceScorer(_BaseDomainAwareScorer):
    """Compute True Positive Parity score as the absolute difference.

    Parameters
    ----------
    weight_estimator : estimator, default=None
        An estimator to compute the weights for the samples.
        If None, the weights are set to 1.
    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    greater_is_better : bool, default=True
        Whether `scorer` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `scorer`.
    """

    def __init__(self, weight_estimator=None, scoring=None, greater_is_better=False):
        super().__init__()
        self.weight_estimator = weight_estimator
        self.scoring = scoring
        self._sign = 1 if greater_is_better else -1

    def _score(self, estimator, X, y, s):
        positives_group_1 = X[s == 1, y == 1]
        positives_group_0 = X[s == 0, y == 1]

        # Get the probabilities of the positive class
        proba_group_1 = estimator.predict_proba(positives_group_1)[1]
        proba_group_0 = estimator.predict_proba(positives_group_0)[1]

        difference = np.mean(proba_group_1, axis=0) - np.mean(proba_group_0, axis=0)

        return self._sign * np.abs(difference)


class TPParityRatioScorer(_BaseDomainAwareScorer):
    """Compute True Positive Parity score as the ratio.

    Parameters
    ----------
    weight_estimator : estimator, default=None
        An estimator to compute the weights for the samples.
        If None, the weights are set to 1.
    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    greater_is_better : bool, default=True
        Whether `scorer` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `scorer`.
    """

    def __init__(self, weight_estimator=None, scoring=None, greater_is_better=False):
        super().__init__()
        self.weight_estimator = weight_estimator
        self.scoring = scoring
        self._sign = 1 if greater_is_better else -1

    def _score(self, estimator, X, y, s):
        positives_group_1 = X[s == 1, y == 1]
        positives_group_0 = X[s == 0, y == 1]

        # Get the probabilities of the positive class
        proba_group_1 = estimator.predict_proba(positives_group_1)[1]
        proba_group_0 = estimator.predict_proba(positives_group_0)[1]

        ratio = np.mean(proba_group_1, axis=0) / np.mean(proba_group_0, axis=0)

        return self._sign * np.abs(ratio - 1)


class FNParityDifferenceScorer(_BaseDomainAwareScorer):
    """
    Compute False Negative Parity score as the absolute difference.

    Parameters
    ----------
    weight_estimator : estimator, default=None
        An estimator to compute the weights for the samples.
        If None, the weights are set to 1.
    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    greater_is_better : bool, default=False
        Whether `scorer` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `scorer`.

    Returns
    -------
    fnp_score : float
        The False Negative Parity score, which is the absolute difference
        between the average predicted probabilities of the negative class for
        the two groups. A score of 0 indicates perfect false negative parity,
        while higher values indicate greater disparity between the groups.
    """

    def __init__(self, weight_estimator=None, scoring=None, greater_is_better=False):
        super().__init__()
        self._sign = 1 if greater_is_better else -1
        self.scoring = scoring
        self.weight_estimator = weight_estimator

    def _score(self, estimator, X, y, s):
        positive_group_1 = X[s == 1, y == 1]
        positive_group_0 = X[s == 0, y == 1]

        # Get the probabilities of the negative class
        proba_group_1 = estimator.predict_proba(positive_group_1)[0]
        proba_group_0 = estimator.predict_proba(positive_group_0)[0]

        diff = np.mean(proba_group_1, axis=0) - np.mean(proba_group_0, axis=0)

        return self._sign * np.abs(diff)


class FNParityRatioScorer(_BaseDomainAwareScorer):
    """Compute False Negative Parity score as the ratio.

    Parameters
    ----------
    weight_estimator : estimator, default=None
        An estimator to compute the weights for the samples.
        If None, the weights are set to 1.
    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    greater_is_better : bool, default=True
        Whether `scorer` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `scorer`.

    Returns
    -------
    fnp_score : float
        The False Negative Parity score, which is the ratio of the average
        predicted probabilities of the negative class for the two groups.
        A score of 1 indicates perfect false negative parity, while higher or
    """

    def __init__(self, weight_estimator=None, scoring=None, greater_is_better=False):
        super().__init__()
        self.weight_estimator = weight_estimator
        self.scoring = scoring
        self._sign = 1 if greater_is_better else -1

    def _score(self, estimator, X, y, s):
        positive_group_1 = X[s == 1, y == 1]
        positive_group_0 = X[s == 0, y == 1]

        # Get the probabilities of the negative class
        proba_group_1 = estimator.predict_proba(positive_group_1)[0]
        proba_group_0 = estimator.predict_proba(positive_group_0)[0]

        ratio = np.mean(proba_group_1, axis=0) / np.mean(proba_group_0, axis=0)

        return self._sign * np.abs(ratio - 1)
