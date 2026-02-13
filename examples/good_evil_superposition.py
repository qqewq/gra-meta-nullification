import numpy as np

from src.spaces import MultiverseHierarchy, Duality
from src.goals import GoalHierarchy, GoalProjector, ReflectionOperator
from src.foam import multiverse_functional
from src.optimizer import gradient_descent_step


def good_evil_superposition_demo(steps: int = 50, eta: float = 1e-1):
    """
    Демонстрация: |D>,|Z> в C^2, отражение sigma_x и проектор на |+><+|.
    Показываем, как состояние стягивается к суперпозиции добра и зла.
    """
    # |D> = (1,0), |Z> = (0,1)
    D = np.array([1.0, 0.0], dtype=np.complex128)
    Z = np.array([0.0, 1.0], dtype=np.complex128)

    dual = Duality(A=D, B=Z)

    # оператор отражения R = sigma_x
    R = np.array([[0.0, 1.0],
                  [1.0, 0.0]], dtype=np.complex128)
    refl = ReflectionOperator(R=R)

    # проектор на |+><+|
    plus = dual.plus.reshape(-1, 1)
    P_plus = plus @ plus.conj().T
    goal = GoalProjector(P=P_plus)

    # один мультииндекс на уровне 0
    states = {
        (0,): np.array([1.0, 0.0], dtype=np.complex128)  # стартуем в чистом |D>
    }
    hierarchy = MultiverseHierarchy(states=states)
    goals = GoalHierarchy(projectors={(0,): goal})
    lambdas = {0: 1.0}

    print("=== Good/Evil Superposition Demo ===")
    print("Initial state:", hierarchy.states[(0,)])
    print("Initial J:", multiverse_functional(hierarchy, goals, lambdas))

    for step in range(steps):
        hierarchy = gradient_descent_step(hierarchy, goals, lambdas, eta=eta)

    psi_final = hierarchy.states[(0,)]
    print("Final state:", psi_final)
    print("Norm:", np.linalg.norm(psi_final))
    print("Overlap with |+>:", np.abs(np.vdot(dual.plus, psi_final)))


if __name__ == "__main__":
    good_evil_superposition_demo()
