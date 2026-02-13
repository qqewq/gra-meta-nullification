import numpy as np
from typing import Dict, Tuple
from .spaces import MultiverseHierarchy
from .goals import GoalHierarchy, GoalProjector

Index = Tuple[int, ...]


def foam_level(
    hierarchy: MultiverseHierarchy,
    goals: GoalHierarchy,
    level: int
) -> float:
    """
    Пенa уровня l:
    Phi^{(l)} = sum_{a != b, dim(a)=dim(b)=l} | <Psi^{(a)} | P_{G_l} | Psi^{(b)}> |^2
    Здесь предполагается один общий проектор для всех индексов данного уровня,
    но можно обобщить до P_{G_l}^{(a)}.
    """
    indices = hierarchy.level_indices(level)
    if not indices:
        return 0.0

    # пока берём один проектор (например для первого индекса уровня)
    P = goals.get(indices[0]).P  # type: ignore

    phi = 0.0
    for i, a in enumerate(indices):
        for j, b in enumerate(indices):
            if i == j:
                continue
            psi_a = hierarchy.states[a]
            psi_b = hierarchy.states[b]
            v = np.vdot(psi_a, P @ psi_b)
            phi += np.abs(v) ** 2
    return float(phi)


def multiverse_functional(
    hierarchy: MultiverseHierarchy,
    goals: GoalHierarchy,
    lambdas: Dict[int, float]
) -> float:
    """
    J_multiverse = sum_l Lambda_l * sum_{dim(a)=l} J^{(l)}(Psi^{(a)})
    В минимальном прототипе J^{(l)} = Phi^{(l)}.
    """
    J = 0.0
    for l in hierarchy.levels():
        phi_l = foam_level(hierarchy, goals, l)
        J += lambdas.get(l, 1.0) * phi_l
    return float(J)
