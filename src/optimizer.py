import numpy as np
from typing import Dict, Tuple
from .spaces import MultiverseHierarchy
from .goals import GoalHierarchy

Index = Tuple[int, ...]


def gradient_descent_step(
    hierarchy: MultiverseHierarchy,
    goals: GoalHierarchy,
    lambdas: Dict[int, float],
    eta: float = 1e-2
) -> MultiverseHierarchy:
    """
    Один шаг градиентного спуска по состояниям Psi^{(a)} для минимизации J_multiverse.
    Здесь мы используем численное приближение градиента по каждому Psi^{(a)}.
    Это медленно, но прозрачно для экспериментов на малых размерах.
    """
    new_h = hierarchy.copy()
    eps = 1e-4

    from .foam import multiverse_functional

    base_J = multiverse_functional(hierarchy, goals, lambdas)

    for a, psi in hierarchy.states.items():
        d = psi.size
        grad = np.zeros_like(psi, dtype=np.complex128)

        for k in range(d):
            e_k = np.zeros_like(psi, dtype=np.complex128)
            e_k[k] = 1.0

            psi_plus = psi + eps * e_k
            psi_minus = psi - eps * e_k

            h_plus = hierarchy.copy()
            h_minus = hierarchy.copy()
            h_plus.states[a] = psi_plus
            h_minus.states[a] = psi_minus

            J_plus = multiverse_functional(h_plus, goals, lambdas)
            J_minus = multiverse_functional(h_minus, goals, lambdas)

            grad_k = (J_plus - J_minus) / (2 * eps)
            grad[k] = grad_k

        new_h.states[a] = psi - eta * grad
        # при желании нормировать
        norm = np.linalg.norm(new_h.states[a])
        if norm > 0:
            new_h.states[a] /= norm

    return new_h
