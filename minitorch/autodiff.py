from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vlist_minus = list(vals)
    vlist_plus = list(vals)
    vlist_minus[arg] -= epsilon / 2
    vlist_plus[arg] += epsilon / 2

    return (f(*vlist_plus) - f(*vlist_minus)) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    to_visit = [variable]
    visited = set()
    topo = []

    while len(to_visit) > 0:
        v = to_visit.pop(0)
        if v.unique_id in visited:
            continue

        topo.append(v)
        visited.add(v.unique_id)
        to_visit.extend(v.parents)

    return topo


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    nodes = topological_sort(variable)

    # zero out all grads first
    for n in nodes:
        n.derivative = None

    # start from the end node and propagate derivatives
    derivatives = {variable.unique_id: deriv}
    for n in nodes:
        if n.is_leaf():
            continue

        assert n.unique_id in derivatives
        for v, d in n.chain_rule(derivatives[n.unique_id]):
            if v.is_leaf():
                v.accumulate_derivative(d)
            else:
                cur = derivatives.get(v.unique_id, 0.0)
                derivatives[v.unique_id] = cur + d


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
