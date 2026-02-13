import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List


Index = Tuple[int, ...]


@dataclass
class Duality:
    """Пара противоположных состояний |A>, |B> в локальном пространстве."""
    A: np.ndarray  # shape: (d,)
    B: np.ndarray  # shape: (d,)

    @property
    def plus(self) -> np.ndarray:
        """|+> = (|A> + |B>)/sqrt(2)."""
        v = self.A + self.B
        return v / np.linalg.norm(v)

    @property
    def minus(self) -> np.ndarray:
        """|-> = (|A> - |B>)/sqrt(2)."""
        v = self.A - self.B
        return v / np.linalg.norm(v)


class MultiverseHierarchy:
    """
    Многоуровневая иерархия состояний Psi^{(a)} с мультииндексами a.
    Здесь мы храним только одно состояние на мультииндекс, без явного тензорного произведения.
    """

    def __init__(self, states: Dict[Index, np.ndarray]):
        """
        states: словарь {(a0,...,al) -> вектор состояния (ndarray)}.
        """
        self.states: Dict[Index, np.ndarray] = states

    def levels(self) -> List[int]:
        return sorted({len(a) - 1 for a in self.states.keys()})

    def level_indices(self, l: int) -> List[Index]:
        return [a for a in self.states.keys() if len(a) - 1 == l]

    def copy(self) -> "MultiverseHierarchy":
        return MultiverseHierarchy({a: v.copy() for a, v in self.states.items()})
