"""Causal DAG declared over CelebA conditioning attributes (TODO item 2)."""
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class CausalGraph:
    attributes: List[str]
    edges: List[Tuple[str, str]] = field(default_factory=list)

    def __post_init__(self):
        self.edges = [tuple(e) for e in self.edges]
        self._children: Dict[str, List[str]] = {a: [] for a in self.attributes}
        self._parents: Dict[str, List[str]] = {a: [] for a in self.attributes}
        for parent, child in self.edges:
            if parent not in self._children or child not in self._children:
                raise ValueError(f"edge {(parent, child)} references an attribute not in {self.attributes}")
            self._children[parent].append(child)
            self._parents[child].append(parent)
        self._topo = self._topological_order()

    def parents(self, node: str) -> List[str]:
        return list(self._parents[node])

    def children(self, node: str) -> List[str]:
        return list(self._children[node])

    def descendants(self, node: str) -> Set[str]:
        """Transitive descendants of ``node``, excluding ``node`` itself."""
        seen: Set[str] = set()
        stack = list(self._children[node])
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(self._children[n])
        return seen

    def topological_order(self) -> List[str]:
        return list(self._topo)

    def _topological_order(self) -> List[str]:
        indeg = {a: len(self._parents[a]) for a in self.attributes}
        queue = [a for a in self.attributes if indeg[a] == 0]
        order = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for c in self._children[n]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        if len(order) != len(self.attributes):
            raise ValueError(f"causal graph over {self.attributes} with edges {self.edges} has a cycle")
        return order

    @classmethod
    def from_dict(cls, raw: dict) -> "CausalGraph":
        return cls(list(raw["attributes"]), [tuple(e) for e in raw.get("edges", [])])
