import numpy as np
import soft_rank_isotonic
from scipy import special


def isotonic_l2(input_s, input_w=None):
    if input_w is None:
        input_w = np.arange(len(input_s))[::-1] + 1
    input_w = input_w.astype(input_s.dtype)
    solution = np.zeros_like(input_s)
    soft_rank_isotonic.isotonic_l2(input_s - input_w, solution)
    return solution


def isotonic_kl(input_s, input_w=None):
    if input_w is None:
        input_w = np.arange(len(input_s))[::-1] + 1
    input_w = input_w.astype(input_s.dtype)
    solution = np.zeros(len(input_s)).astype(input_s.dtype)
    soft_rank_isotonic.isotonic_kl(input_s, input_w.astype(input_s.dtype), solution)
    return solution


def _partition(solution, eps=1e-9):
    if len(solution) == 0:
        return []

    sizes = [1]

    for i in range(1, len(solution)):
        if abs(solution[i] - solution[i - 1]) > eps:
            sizes.append(0)
        sizes[-1] += 1

    return sizes


def _check_regularization(regularization):
    if regularization not in ("l2", "kl"):
        raise ValueError("'regularization' should be either 'l2' or 'kl' " "but got %s." % str(regularization))


class _Differentiable(object):

    def jacobian(self):
        identity = np.eye(self.size)
        return np.array([self.jvp(identity[i]) for i in range(len(identity))]).T

    @property
    def size(self):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def jvp(self, vector):
        raise NotImplementedError

    def vjp(self, vector):
        raise NotImplementedError


class Isotonic(_Differentiable):

    def __init__(self, input_s, input_w, regularization="l2"):
        self.input_s = input_s
        self.input_w = input_w
        _check_regularization(regularization)
        self.regularization = regularization
        self.solution_ = None

    @property
    def size(self):
        return len(self.input_s)

    def compute(self):

        if self.regularization == "l2":
            self.solution_ = isotonic_l2(self.input_s, self.input_w)
        else:
            self.solution_ = isotonic_kl(self.input_s, self.input_w)
        return self.solution_

    def _check_computed(self):
        if self.solution_ is None:
            raise RuntimeError("Need to run compute() first.")

    def jvp(self, vector):
        self._check_computed()
        start = 0
        return_value = np.zeros_like(self.solution_)
        for size in _partition(self.solution_):
            end = start + size
            if self.regularization == "l2":
                val = np.mean(vector[start:end])
            else:
                val = np.dot(special.softmax(self.input_s[start:end]), vector[start:end])
            return_value[start:end] = val
            start = end
        return return_value

    def vjp(self, vector):
        start = 0
        return_value = np.zeros_like(self.solution_)
        for size in _partition(self.solution_):
            end = start + size
            if self.regularization == "l2":
                val = 1.0 / size
            else:
                val = special.softmax(self.input_s[start:end])
            return_value[start:end] = val * np.sum(vector[start:end])
            start = end
        return return_value


def _inv_permutation(permutation):
    inv_permutation = np.zeros(len(permutation), dtype=int)
    inv_permutation[permutation] = np.arange(len(permutation))
    return inv_permutation


class Projection(_Differentiable):

    def __init__(self, input_theta, input_w=None, regularization="l2"):
        if input_w is None:
            input_w = np.arange(len(input_theta))[::-1] + 1
        self.input_theta = np.asarray(input_theta)
        self.input_w = np.asarray(input_w)
        _check_regularization(regularization)
        self.regularization = regularization
        self.isotonic = None

    def _check_computed(self):
        if self.isotonic_ is None:
            raise ValueError("Need to run compute() first.")

    @property
    def size(self):
        return len(self.input_theta)

    def compute(self):
        self.permutation = np.argsort(self.input_theta)[::-1]
        input_s = self.input_theta[self.permutation]

        self.isotonic_ = Isotonic(input_s, self.input_w, self.regularization)
        dual_sol = self.isotonic_.compute()
        primal_sol = input_s - dual_sol

        self.inv_permutation = _inv_permutation(self.permutation)
        return primal_sol[self.inv_permutation]

    def jvp(self, vector):
        self._check_computed()
        ret = vector.copy()
        ret -= self.isotonic_.jvp(vector[self.permutation])[self.inv_permutation]
        return ret

    def vjp(self, vector):
        self._check_computed()
        ret = vector.copy()
        ret -= self.isotonic_.vjp(vector[self.permutation])[self.inv_permutation]
        return ret


def _check_direction(direction):
    if direction not in ("ASCENDING", "DESCENDING"):
        raise ValueError("direction should be either 'ASCENDING' or 'DESCENDING'")


class SoftRank(_Differentiable):

    def __init__(self, values, direction="ASCENDING", regularization_strength=1.0, regularization="l2"):
        self.values = np.asarray(values)
        self.input_w = np.arange(len(values))[::-1] + 1
        _check_direction(direction)
        sign = 1 if direction == "ASCENDING" else -1
        self.scale = sign / regularization_strength
        _check_regularization(regularization)
        self.regularization = regularization
        self.projection_ = None

    @property
    def size(self):
        return len(self.values)

    def _check_computed(self):
        if self.projection_ is None:
            raise ValueError("Need to run compute() first.")

    def compute(self):
        if self.regularization == "kl":
            self.projection_ = Projection(self.values * self.scale, np.log(self.input_w), regularization=self.regularization)
            self.factor = np.exp(self.projection_.compute())
            return self.factor
        else:
            self.projection_ = Projection(self.values * self.scale, self.input_w, regularization=self.regularization)
            self.factor = 1.0
            return self.projection_.compute()

    def jvp(self, vector):
        self._check_computed()
        return self.factor * self.projection_.jvp(vector) * self.scale

    def vjp(self, vector):
        self._check_computed()
        return self.projection_.vjp(self.factor * vector) * self.scale


class SoftSort(_Differentiable):

    def __init__(self, values, direction="ASCENDING", regularization_strength=1.0, regularization="l2"):
        self.values = np.asarray(values)
        _check_direction(direction)
        self.sign = 1 if direction == "DESCENDING" else -1
        self.regularization_strength = regularization_strength
        _check_regularization(regularization)
        self.regularization = regularization
        self.isotonic_ = None

    @property
    def size(self):
        return len(self.values)

    def _check_computed(self):
        if self.isotonic_ is None:
            raise ValueError("Need to run compute() first.")

    def compute(self):
        size = len(self.values)
        input_w = np.arange(1, size + 1)[::-1] / self.regularization_strength
        values = self.sign * self.values
        self.permutation_ = np.argsort(values)[::-1]
        s = values[self.permutation_]

        self.isotonic_ = Isotonic(input_w, s, regularization=self.regularization)
        res = self.isotonic_.compute()

        self.isotonic_.s = s
        return self.sign * (input_w - res)

    def jvp(self, vector):
        self._check_computed()
        return self.isotonic_.jvp(vector[self.permutation_])

    def vjp(self, vector):
        self._check_computed()
        inv_permutation = _inv_permutation(self.permutation_)
        return self.isotonic_.vjp(vector)[inv_permutation]


class Sort(_Differentiable):

    def __init__(self, values, direction="ASCENDING"):
        _check_direction(direction)
        self.values = np.asarray(values)
        self.sign = 1 if direction == "DESCENDING" else -1
        self.permutation_ = None

    @property
    def size(self):
        return len(self.values)

    def _check_computed(self):
        if self.permutation_ is None:
            raise ValueError("Need to run compute() first.")

    def compute(self):
        self.permutation_ = np.argsort(self.sign * self.values)[::-1]
        return self.values[self.permutation_]

    def jvp(self, vector):
        self._check_computed()
        return vector[self.permutation_]

    def vjp(self, vector):
        self._check_computed()
        inv_permutation = _inv_permutation(self.permutation_)
        return vector[inv_permutation]


def soft_rank(values, direction="ASCENDING", regularization_strength=1.0, regularization="l2"):
    return SoftRank(values, regularization_strength=regularization_strength, direction=direction, regularization=regularization).compute()


def soft_sort(values, direction="ASCENDING", regularization_strength=1.0, regularization="l2"):
    return SoftSort(values, regularization_strength=regularization_strength, direction=direction, regularization=regularization).compute()


def sort(values, direction="ASCENDING"):
    return Sort(values, direction=direction).compute()


def rank(values, direction="ASCENDING"):
    permutation = np.argsort(values)
    if direction == "DESCENDING":
        permutation = permutation[::-1]
    return _inv_permutation(permutation) + 1
