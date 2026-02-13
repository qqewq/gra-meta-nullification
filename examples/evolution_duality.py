import numpy as np

from src.spaces import MultiverseHierarchy, Duality
from src.goals import GoalHierarchy, GoalProjector, ReflectionOperator
from src.foam import multiverse_functional
from src.optimizer import gradient_descent_step


def evolution_duality_demo(steps: int = 50, eta: float = 1e-1):
    """
    Пример: |E>,|L> как дарвинизм и ламаркизм.
    Цель инвариантна относительно замены E<->L и выделяет симметричную суперпозицию.
    """
    # |E> = (1,0), |L> = (0,1)
    E = np.array([1.0, 0.0], dtype=np.complex128)
    L = np.array([0.0, 1.0], dtype=np.complex128)

    dual = Duality(A=E, B=L)

    # отражение R: меняет E<->L
    R = np.array([[0.0, 1.0],
                  [1.0, 0.0]], dtype=np.complex128)
    refl = ReflectionOperator(R=R)

    plus = dual.plus.reshape(-1, 1)
    P_plus = plus @ plus.conj().T
    goal = GoalProjector(P=P_plus)

    states = {
        (0,): E.copy()
    }
    hierarchy = MultiverseHierarchy(states=states)
    goals = GoalHierarchy(projectors={(0,): goal})
    lambdas = {0: 1.0}

    print("=== Evolution Duality Demo ===")
    print("Initial state:", hierarchy.states[(0,)])
    print("Initial J:", multiverse_functional(hierarchy, goals, lambdas))

    for step in range(steps):
        hierarchy = gradient_descent_step(hierarchy, goals, lambdas, eta=eta)

    psi_final = hierarchy.states[(0,)]
    print("Final state:", psi_final)
    print("Norm:", np.linalg.norm(psi_final))
    print("Overlap with |+>:", np.abs(np.vdot(dual.plus, psi_final)))


if __name__ == "__main__":
    evolution_duality_demo()
