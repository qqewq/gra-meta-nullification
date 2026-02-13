import numpy as np
from .spaces import MultiverseHierarchy, Duality
from .goals import GoalHierarchy, GoalProjector, ReflectionOperator
from .foam import multiverse_functional
from .optimizer import gradient_descent_step


def example_good_evil():
    """
    Пример: пространство C^2 с базисом |D>,|Z>,
    оператор отражения sigma_x и проектор на |+><+|.
    Показываем, как состояние стягивается к суперпозиции добра и зла.
    """
    # базис |D>=(1,0), |Z>=(0,1)
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

    # уровень 0: один мультииндекс (0,)
    states = {
        (0,): np.array([1.0, 0.0], dtype=np.complex128)  # начнем с чистого "добра"
    }
    hierarchy = MultiverseHierarchy(states=states)
    goals = GoalHierarchy(projectors={(0,): goal})
    lambdas = {0: 1.0}

    print("Initial J:", multiverse_functional(hierarchy, goals, lambdas))

    for step in range(50):
        hierarchy = gradient_descent_step(hierarchy, goals, lambdas, eta=1e-1)

    psi_final = hierarchy.states[(0,)]
    print("Final state:", psi_final)
    print("Norm:", np.linalg.norm(psi_final))
    print("Overlap with |+>:", np.abs(np.vdot(dual.plus, psi_final)))


if __name__ == "__main__":
    example_good_evil()
