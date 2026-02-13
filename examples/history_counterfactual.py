import numpy as np

from src.spaces import MultiverseHierarchy, Duality
from src.goals import GoalHierarchy, GoalProjector, ReflectionOperator
from src.foam import multiverse_functional
from src.optimizer import gradient_descent_step


def history_counterfactual_demo(steps: int = 50, eta: float = 1e-1):
    """
    Пример: |H>,|A> как история и контрфакт, цель инвариантна относительно замены H<->A.
    Здесь также используем проектор на |+> = (|H>+|A>)/sqrt(2).
    """
    # |H> = (1,0), |A> = (0,1)
    H = np.array([1.0, 0.0], dtype=np.complex128)
    A = np.array([0.0, 1.0], dtype=np.complex128)

    dual = Duality(A=H, B=A)

    # отражение R: меняет H<->A
    R = np.array([[0.0, 1.0],
                  [1.0, 0.0]], dtype=np.complex128)
    refl = ReflectionOperator(R=R)

    # проектор цели на |+><+|
    plus = dual.plus.reshape(-1, 1)
    P_plus = plus @ plus.conj().T
    goal = GoalProjector(P=P_plus)

    # стартуем в чистой истории |H>
    states = {
        (0,): H.copy()
    }
    hierarchy = MultiverseHierarchy(states=states)
    goals = GoalHierarchy(projectors={(0,): goal})
    lambdas = {0: 1.0}

    print("=== History/Counterfactual Demo ===")
    print("Initial state:", hierarchy.states[(0,)])
    print("Initial J:", multiverse_functional(hierarchy, goals, lambdas))

    for step in range(steps):
        hierarchy = gradient_descent_step(hierarchy, goals, lambdas, eta=eta)

    psi_final = hierarchy.states[(0,)]
    print("Final state:", psi_final)
    print("Norm:", np.linalg.norm(psi_final))
    print("Overlap with |+>:", np.abs(np.vdot(dual.plus, psi_final)))


if __name__ == "__main__":
    history_counterfactual_demo()
