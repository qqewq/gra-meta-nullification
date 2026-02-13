import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

Index = Tuple[int, ...]


@dataclass
class ReflectionOperator:
    """
    Оператор отражения R, реализованный матрицей R (гермитова инволюция).
    Для дуальности |A>,|B> это может быть аналогом sigma_x в базисе {A,B}.
    """
    R: np.ndarray  # shape: (d, d)

    def apply(self, v: np.ndarray) -> np.ndarray:
        return self.R @ v

    def commutes_with(self, P: np.ndarray, atol: float = 1e-8) -> bool:
        return np.allclose(self.R @ P, P @ self.R, atol=atol)


@dataclass
class GoalProjector:
    """
    Проектор цели G_l^{(a)} в локальном пространстве H^{(a)}.
    """
    P: np.ndarray  # shape: (d, d)

    def project(self, v: np.ndarray) -> np.ndarray:
        return self.P @ v


class GoalHierarchy:
    """
    Хранилище проекторов целей для мультииндексов.
    """

    def __init__(self, projectors: Dict[Index, GoalProjector]):
        self.projectors = projectors

    def get(self, a: Index) -> GoalProjector:
        return self.projectors[a]
