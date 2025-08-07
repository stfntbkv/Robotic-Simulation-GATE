"""
Re-worked Prolog wrapper for AutoGPT-P
✓ compatible with PySWIP ≥ 0.3
✓ implements _predicate_to_prolog_string
✓ minor stylistic/typing clean-ups
"""

from __future__ import annotations

import os
import string
from itertools import chain
from typing import List, Dict, Sequence

from pyswip import Prolog

from object_detection.detection_memory_segment import DetectedObject, ObjectRelation
from pddl.core import Predicate, LogicElement, And, Not
from pddl.plan import Plan

from autogpt_p.helpers.dnf import goal_to_dnf
from autogpt_p.helpers.name_helpers import digits_to_letters

# --------------------------------------------------------------------------- #
# Paths to the static Prolog “environment”
# --------------------------------------------------------------------------- #
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_CURRENT_DIR)
_ROOT_DIR = os.path.dirname(_PACKAGE_ROOT)

_DATA_DIR = os.path.join(_ROOT_DIR, "data", "prolog_environment")
_ENV_FILE = os.path.join(_DATA_DIR, "execution.txt")

_EXECUTION_PREDICATE = "execute"

# --------------------------------------------------------------------------- #
# Helper classes & functions
# --------------------------------------------------------------------------- #


def _prolog_vars(n: int) -> List[str]:
    """Return first *n* capital letters – used as anonymous Prolog variables."""
    return list(string.ascii_uppercase[:n])


class PrologPredicate:
    """Lightweight representation of a (possibly ground) Prolog predicate."""

    def __init__(self, name: str, arity: int, parameters: Sequence[str] | None = None):
        self.name = name
        self.arity = arity
        self.parameters: List[str] = list(parameters) if parameters else []

    # ------------------------------------------------------------------ #
    # Conversions
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pddl(cls, pred: Predicate) -> "PrologPredicate":
        return cls(pred.name, len(pred.variables),
                   [digits_to_letters(v.name) for v in pred.variables])

    @classmethod
    def from_object_relation(cls, rel: ObjectRelation) -> "PrologPredicate":
        params = [f"{digits_to_letters(o.class_name)}{o.id}" for o in rel.related_objects]
        return cls(rel.relation_name, len(params), params)

    @classmethod
    def from_string(cls, s: str) -> "PrologPredicate":
        s = s.replace(" ", "")
        head, body = s.split("(", 1)
        params = body.rstrip(")").split(",") if body.rstrip(")") else []
        return cls(head, len(params), params)

    # ------------------------------------------------------------------ #
    # Pretty printing
    # ------------------------------------------------------------------ #

    def clear_pattern(self) -> str:
        """Pattern used with retractall/1: name(A,B,...)"""
        return f"{self.name}({','.join(_prolog_vars(self.arity))})"

    def __str__(self) -> str:  # pragma: no cover
        if not self.parameters:
            return self.clear_pattern()
        return f"{self.name}({','.join(self.parameters)})"


def _element_str(pred: PrologPredicate, add: bool) -> str:
    tag = "add" if add else "del"
    return f"{tag}({pred})"


class PrologAddDelList:
    """Encapsulate a list of additions/deletions for the planner."""

    def __init__(self) -> None:
        self.predicates: List[PrologPredicate] = []
        self.add_or_del: List[bool] = []

    def append_add(self, pred: PrologPredicate) -> None:
        self.append(pred, True)

    def append_del(self, pred: PrologPredicate) -> None:
        self.append(pred, False)

    def append(self, pred: PrologPredicate, add: bool = True) -> None:
        self.predicates.append(pred)
        self.add_or_del.append(add)

    def __str__(self) -> str:  # pragma: no cover
        elems = (_element_str(p, a) for p, a in zip(self.predicates, self.add_or_del))
        return f"[{','.join(elems)}]"


# --------------------------------------------------------------------------- #
# Main wrapper
# --------------------------------------------------------------------------- #


class PrologWrapper:
    """Interface between Python and the Prolog execution environment."""

    def __init__(self) -> None:
        self.prolog = Prolog()
        self.added_predicates: List[PrologPredicate] = []

    # -------------  low-level helpers ---------------------------------- #

    def _assert(self, statement: str) -> None:
        self.prolog.assertz(statement)
        self.added_predicates.append(
            PrologPredicate.from_string(statement.split(":-")[0])
        )

    def _list_to_pl(self, items: Sequence[str | PrologPredicate]) -> str:
        return "[" + ",".join(map(str, items)) + "]"

    def _predicate_to_pl_string(self, pred: Predicate) -> str:
        params = ",".join(digits_to_letters(v.name) for v in pred.variables)
        return f"{pred.name}({params})"

    # -------------  public API ----------------------------------------- #

    # state management --------------------------------------------------- #
    def clear_all(self) -> None:
        for pred in self.added_predicates:
            self.prolog.retractall(pred.clear_pattern())
        self.added_predicates.clear()

    # knowledge base I/O ------------------------------------------------- #
    def add_statement_from_string(self, statement: str) -> None:
        self._assert(statement)

    def query_bool_from_string(self, query: str) -> bool:
        return bool(list(self.prolog.query(query)))

    # planning ----------------------------------------------------------- #
    def check_goal_state(self, relations: List[ObjectRelation], goal: LogicElement) -> bool:
        """Return True iff *goal* is satisfied in *relations*."""
        self.clear_all()
        for rel in relations:
            self.add_object_relation(rel)

        for conjunct in goal_to_dnf(goal):
            conj_query = ",".join(self._predicate_to_pl_string(p) for p in conjunct)
            if self.query_bool_from_string(conj_query):
                return True
        return False

    def get_result_from_plan(
        self,
        object_list: List[DetectedObject],
        start_state: List[ObjectRelation],
        plan: Plan,
    ) -> List[ObjectRelation]:
        """Return the world‐state after executing *plan* in Prolog."""
        # initialise KB
        self.clear_all()
        self._setup_plan_environment()

        # map of Prolog-atom → DetectedObject
        all_objs = set(chain.from_iterable(r.related_objects for r in start_state)) | set(object_list)
        obj_map: Dict[str, DetectedObject] = {
            f"{digits_to_letters(o.class_name)}{o.id}": o for o in all_objs
        }

        # encode initial state
        init_state = [PrologPredicate.from_object_relation(r) for r in start_state]

        # encode plan effects
        effect_lists = [self._effect_to_pl_list(a.get_action_effects()) for a in plan.actions]
        plan_effects_pl = self._list_to_pl(map(str, effect_lists))

        query = f"{_EXECUTION_PREDICATE}({self._list_to_pl(init_state)},{plan_effects_pl},X)"
        # PySWIP ≥ 0.3 -> use keyword arg
        result = list(self.prolog.query(query, maxresult=1))[0]
        final_state_pl = [PrologPredicate.from_string(s) for s in result["X"]]

        return [self._relation_from_pl(p, obj_map) for p in final_state_pl]

    def validate_plan(
        self,
        object_list: List[DetectedObject],
        start_state: List[ObjectRelation],
        plan: Plan,
        goal: LogicElement,
    ) -> bool:
        return self.check_goal_state(
            self.get_result_from_plan(object_list, start_state, plan), goal
        )

    # predicate helpers -------------------------------------------------- #
    def add_object_relation(self, relation: ObjectRelation) -> None:
        atoms = ",".join(o.class_name for o in relation.related_objects)
        self._assert(f"{relation.relation_name}({atoms})")

    def add_pddl_predicate(self, pred: Predicate) -> None:
        atoms = ",".join(digits_to_letters(v.write_no_type()) for v in pred.variables)
        self._assert(f"{pred.name}({atoms})")

    # internal helpers --------------------------------------------------- #
    def _setup_plan_environment(self) -> None:
        with open(_ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                self.add_statement_from_string(line.strip())

    def _relation_from_pl(self, pred: PrologPredicate, obj_map: Dict[str, DetectedObject]) -> ObjectRelation:
        return ObjectRelation(pred.name, [obj_map[p] for p in pred.parameters])

    def _effect_to_pl_list(self, effect: LogicElement) -> PrologAddDelList:
        out = PrologAddDelList()
        if isinstance(effect, And):
            for elem in effect.logic_elements:
                if isinstance(elem, Predicate):
                    out.append_add(PrologPredicate.from_pddl(elem))
                elif isinstance(elem, Not) and isinstance(elem.logic_elements[0], Predicate):
                    out.append_del(PrologPredicate.from_pddl(elem.logic_elements[0]))
        return out
