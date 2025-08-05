# autogpt_p/planning/autogpt_planner.py
from __future__ import annotations         # built-in generics work on Py < 3.9
import os
import signal
from typing import List

from object_affordance_mapping.object_affordance_mapping import (
    ObjectAffordanceMappingDatabase,
)
from object_detection.detection_memory_segment import DetectedObject, ObjectRelation

from autogpt_p.llm.llm_interface import NoGoalException, LLMInterface
from autogpt_p.execution.pddl_scenario import define_domain, define_problem
from autogpt_p.planning.planner import FastDownwardPlanner
from autogpt_p.planning.goal_validator import (
    MissingGoal,
    UnknownPredicate,
    UnknownObject,
    GoalValidator,
    PredicateLimitation,
    TypingError,
)
from autogpt_p.planning.validation_error_handler import ValidationErrorHandler
from pddl.core import Predicate
from pddl.plan import Plan
from pddl.problem import PredicateParsingException, ObjectParsingException


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def timeout_handler(signum, frame):
    raise TimeoutError("Function call timeout")


def _timeout_function(f, timeout, *args, **kwargs):
    """Run *f* with Unix alarm-based timeout."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        return f(*args, **kwargs)
    finally:
        signal.alarm(0)  # always disable alarm


def get_limitations():
    """Domain-invariant predicate constraints used by GoalValidator."""
    return [
        PredicateLimitation(
            "an object can only be on one other object",
            "forall(on(X,Y), not(on_other(X,Y,Z)))",
        ),
        PredicateLimitation(
            "an object can only be in one other object",
            "forall(in(X,Y), (not(in_other(X,Y,Z)), not(on_other(X,Y,Z))))",
        ),
        PredicateLimitation(
            "an object held by an actor cannot be elsewhere",
            "forall(inhand(X,Y), (not(in_other_hand(X,Y,Z)), not(on_other(X,Y,Z)), not(in_other(X,Y,Z))))",
        ),
        PredicateLimitation(
            "an actor can only be at one location",
            "forall(at(X,Y), not(at_other(X,Y,Z)))",
        ),
        PredicateLimitation(
            "a liquid can only be in one container",
            "forall(liquid_in(X,Y), not(liquid_in_other(X,Y,Z)))",
        ),
    ]


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
class AutoGPTPlanner:
    """
    Inner feedback-loop planner driven by an LLM.  Two public planning
    variants are provided; `plan()` is an alias to the incremental variant
    for backward compatibility.
    """

    def __init__(
        self,
        chatgpt: LLMInterface,
        objects: List[DetectedObject],
        relations: List[ObjectRelation],
        locations: List[str],
        oam_db: ObjectAffordanceMappingDatabase,
        actor_skill_mapping=None,
        *,
        partial_plan: bool = False,
        max_predicates: int = 5,
        max_loops: int = 5,
        number_of_examples: int = 3,
    ):
        self.actor_skill_mapping = actor_skill_mapping
        self.chatgpt = chatgpt
        self.chatgpt_fresh = chatgpt.branch()
        self.oam_db = oam_db

        self.max_loops = max_loops
        self.max_predicates = max_predicates
        self.partial_plan = partial_plan
        self.number_of_examples = number_of_examples

        # scene-specific state
        self.problem = self.domain = None
        self.objects = self.relations = None
        self.locations = locations or []
        self.recent_plan = None
        self.feedback_loops = 0

        self.update_scene(objects, relations, locations)

    # ---------------- scene / history management ---------------- #
    def update_scene(
        self,
        objects: List[DetectedObject],
        relations: List[ObjectRelation],
        locations: list[str] | None = None,
    ):
        """Refresh PDDL domain & problem from new perception input."""
        self.locations = locations if locations is not None else self.locations
        self.objects, self.relations = objects, relations

        self.domain = define_domain(
            "robotic_planning", self.oam_db, objects, self.actor_skill_mapping, False
        )
        self.problem = define_problem(
            "test",
            self.domain,
            objects,
            relations,
            self.locations,
            self.actor_skill_mapping,
        )
        self.feedback_loops = 0

    def reset_history(self):
        """Start a new conversation branch with the LLM (keeps system prompts)."""
        self.chatgpt, self.chatgpt_fresh = self.chatgpt_fresh, self.chatgpt.branch()

    # ---------------- public API ---------------- #
    def plan(self, user_task: str):
        """
        **Compatibility wrapper** expected by evaluation scripts.
        Uses the incremental-memory planner under the hood.
        """
        return self.plan_with_incremental_goal_memory(
            user_task, self.number_of_examples
        )

    # --------------------------------------------------------------------- #
    # Incremental goal-memory planning (preferred)
    # --------------------------------------------------------------------- #
    def plan_with_incremental_goal_memory(
        self, user_task: str, number_of_examples: int
    ) -> Plan:
        """
        Feedback loop that iteratively queries the LLM for a goal, validates
        it, and plans with Fast-Downward until a consistent plan is found or
        the loop budget is exhausted.
        """
        found_goal = False
        error_handler: ValidationErrorHandler | None = None
        plan = Plan([])
        plan.costs = -1

        predicates = [Predicate(p.name, p.variables, p.comment, p.definition)
                      for p in self.domain.predicates]

        while self.feedback_loops < self.max_loops and not found_goal:
            self._make_generic(False)
            self.feedback_loops += 1

            # 1) ask LLM (with or without previous error context)
            try:
                if error_handler:
                    goal_string = error_handler.correct_error()
                elif self.partial_plan:
                    goal_string = self.chatgpt.ask_for_partial_goal_in_context_learning(
                        user_task, self.domain, self.problem,
                        self.max_predicates, number_of_examples
                    )
                else:
                    goal_string = self.chatgpt.ask_for_goal_in_context_learning(
                        user_task, self.domain, self.problem, number_of_examples
                    )
                print("GOAL:", goal_string)
            except NoGoalException:
                error_handler = ValidationErrorHandler(MissingGoal(), self.chatgpt)
                print(error_handler.validation_error.error_message)
                continue

            # 2) build PDDL problem & validate
            self._make_generic(True)

            if "?" in goal_string:
                error_handler = ValidationErrorHandler(TypingError(), self.chatgpt)
                print(error_handler.validation_error.error_message)
                continue

            try:
                problem = define_problem(
                    "test",
                    self.domain,
                    self.objects,
                    self.relations,
                    self.locations,
                    self.actor_skill_mapping,
                    [],
                    goal_string,
                )
            except PredicateParsingException as e:
                error_handler = ValidationErrorHandler(
                    UnknownPredicate(e.predicate_name), self.chatgpt
                )
                print(error_handler.validation_error.error_message)
                continue
            except ObjectParsingException as e:
                error_handler = ValidationErrorHandler(
                    UnknownObject(e.object_name), self.chatgpt
                )
                print(error_handler.validation_error.error_message)
                continue

            error = GoalValidator(get_limitations(), predicates, problem.goal).validate()
            if error:
                error_handler = ValidationErrorHandler(error, self.chatgpt)
                continue

            # 3) call Fast-Downward (with timeout)
            planner = FastDownwardPlanner(self.domain)
            try:
                plan = _timeout_function(planner.solve, 120, problem)
                found_goal, self.problem = True, problem
            except TimeoutError:
                print("Plan too complex, switching to partial-plan mode")
                self.partial_plan = True
                self.max_predicates = max(1, self.max_predicates - 1)

        self.recent_plan = plan
        return plan

    # --------------------------------------------------------------------- #
    # Non-incremental variant kept for completeness
    # --------------------------------------------------------------------- #
    def plan_without_incremental_goal_memory(self, user_task: str) -> Plan:
        return self.plan_with_incremental_goal_memory(user_task, 0)

    # ---------------- internal helpers ---------------- #
    def _make_generic(self, generic: bool):
        """Regenerate a typed or untyped domain (generic==True = untyped)."""
        self.domain = define_domain(
            "robotic_planning", self.oam_db, self.objects, self.actor_skill_mapping, generic
        )
