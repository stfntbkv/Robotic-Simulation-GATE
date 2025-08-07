from abc import ABC
from typing import Optional

from autogpt_p.llm.llm_interface import LLMInterface
from autogpt_p.state_machine.auto_gpt_p_memory import Memory
from autogpt_p.tools.partial_plan import PartialPlan
from autogpt_p.tools.explore import Explore
from autogpt_p.tools.correction import Correction
from autogpt_p.tools.plan import Plan
from autogpt_p.tools.suggest_substitution import SuggestSubstitution
from autogpt_p.tools.tool import Tool
from autogpt_p.tools.tool_selector import ToolSelector, from_classes
from autogpt_p.tools.tool_selector_factory import ToolSelectorFactory


FAILURE = "FAILURE"
SELECTED_TOOL = "SELECTED TOOL:"


class State(ABC):

    def __init__(self, context):
        self.next_state = None
        self.context = context

    def next(self):
        return self.next_state

    def enter(self):
        pass

    def exit(self):
        pass

    def process_command(self, command, memory):
        pass

    def process_failure(self, error, memory):
        pass

    def abort(self):
        pass


class Idle(State):

    def __init__(self, context):
        super().__init__(context)

    def enter(self):
        print("<<<Entering Idle State>>>")
        self.context.llm.reset_history()
        self.context.current_iterations = 0
        self.context.final_plan = False
        if self.context.memory.planner:
            self.context.memory.planner.recent_plan = None

    def exit(self):
        return

    def process_command(self, command, memory):
        self.next_state = SelectTool(self.context, command, memory)

    def process_failure(self, error, memory):
        # there cannot be an error during the idle state
        self.next_state = self

    def abort(self):
        self.next_state = self


class SelectTool(State):

    def __init__(self, context, command, memory):
        super().__init__(context)
        self.command = command
        self.memory = memory

    def enter(self):
        print("<<<Entering Select Tool State>>>")
        # if max_iterations are reached break the cycle and go back to idle state
        if self.context.current_iterations == 0:
            self.memory.command_memory.append(self.command)
        if self.context.current_iterations >= self.context.max_iterations:
            self.next_state = Idle(self.context)
            print("Max number of iterations reached")
        elif self.memory.planner.recent_plan and self.memory.planner.recent_plan.is_valid():
            self.next_state = ExecutePlan(self.context, self.memory.planner.recent_plan)
        else:
            tool = determine_action(self.context.llm, self.command, self.memory,
                                    ToolSelectorFactory.get_instance().produce_tool_selector(),
                                    self.context.current_iterations == 0)
            self.context.current_iterations += 1
            if tool:
                if isinstance(tool, Correction):
                    self.context.current_iterations = self.context.max_iterations
                self.memory.tool_memory.append(tool)
                self.next_state = ExecuteTool(self.context, tool)
            else:
                # in the failure case just call partial plan once as a
                # fallback strategy as gpt-4 is not good at determining when this is needed
                if PartialPlan(self.memory) not in self.memory.tool_memory:
                    tool = PartialPlan(self.memory).get_executable([])
                    self.memory.tool_memory.append(tool)
                    self.next_state = ExecuteTool(self.context, tool)

                else:
                    self.next_state = Idle(self.context)

        self.context.transition(self.next())

    def exit(self):
        return

    def process_command(self, command, memory):
        self.next_state = SelectTool(self.context, command, memory)

    def process_failure(self, error, memory):
        self.next_state = Idle(self.context)

    def abort(self):
        self.next_state = Idle(self.context)


class ExecuteTool(State):

    def __init__(self, context, tool):
        super().__init__(context)
        self.tool = tool

    def enter(self):
        print("<<<Entering Execute Tool State>>>")
        result = self.tool.execute()
        self.tool.memory.last_result = result
        self.next_state = SelectTool(self.context, self.tool.memory.command_memory[-1], self.tool.memory)
        self.context.transition(self.next())

    def exit(self):
        self.tool.abort()

    def process_command(self, command, memory):
        self.next_state = SelectTool(self.context, command, memory)

    def process_failure(self, error, memory):
        pass

    def abort(self):
        self.next_state = Idle(self.context)


class ExecutePlan(State):

    def __init__(self, context, plan):
        super().__init__(context)
        self.plan = plan

    def enter(self):
        print("<<<Entering Execute Plan State>>>")
        self.context.executor.execute(self.plan)
        self.context.memory.update_with_plan(self.plan)
        if not self.context.memory.planner.partial_plan:
            self.next_state = Idle(self.context)
        else:
            if self.context.memory.planner:
                self.context.memory.planner.recent_plan = None
            self.next_state = SelectTool(self.context, self.context.memory.command_memory[-1], self.context.memory)
        self.context.transition(self.next_state)

    def exit(self):
        self.context.executor.abort()

    def process_command(self, command, memory):
        self.next_state = SelectTool(self.context, command, memory)

    def process_failure(self, error, memory):
        pass

    def abort(self):
        self.next_state = Idle(self.context)


def parse_response(response, tool_selector: ToolSelector) -> Optional[Tool]:
    if FAILURE in response:
        return None
    start = response.find(SELECTED_TOOL) + len(SELECTED_TOOL)
    tool_string = response[start:]
    tool = tool_selector.get_tool_in(tool_string)
    # if not formatted correctly try to find a tool name in the response and just use that
    if not tool:
        tool = tool_selector.get_tool_in(response)
    if tool:
        return tool.get_executable(tool_string.replace(tool.keyword, "").strip().split(" "))
    else:
        return tool


def get_tools(memory) -> ToolSelector:
    return from_classes(memory, [PartialPlan, Plan, Explore, SuggestSubstitution, Correction])


def determine_action(llm_instance: LLMInterface, objective: str, memory: Memory, tool_selector: ToolSelector,
                     first_time: bool):
    tool_str = "\n".join([t.__str__() for t in tool_selector.get_all_tools()])
    mem_str = memory.print_memory_for_gpt()
    objective = memory.substitution_memory.substitute_in_prompt(objective)
    previous_objectives = "\n".join(["\"" + objective + "\"" for objective in memory.command_memory[:-1]])
    if objective == "correction":
        return Correction(memory).get_executable([])
    if first_time:
        formatted_prompt = \
            f"""You are an robotic assistant that tries to help the user to achieve his plan.
You are given a representation of the scene and need to determine which tool to use to help the user achieve 
his goal the best way.

Scene-Memory:
{mem_str}


Available Tools:
{tool_str}

Previous User Requests:
{previous_objectives}

User-Request: {objective}

Lets think step by step. Remember that unexplored locations could contain objects that are relevant to the task.
Only call the planning tool if all utensils are there to fulfill the plan. For example if there is nothing for cutting
the goal "cut the apple" cannot be reached. 
Which tool would you use? End your answer with {SELECTED_TOOL}<TOOL> <PARAMETERS>. 
End your answer with {FAILURE} if no tool is valid.
        """
    else:
        formatted_prompt = \
            f"""New Scene Memory:
{mem_str}

Result of last tool execution: {memory.last_result}

Available Tools:
{tool_str}

Previous User Requests:
{previous_objectives}

User-Request: {objective}

Lets think step by step. Remember that unexplored locations could contain objects that are relevant to the task.
Only call the planning tool if all utensils are there to fulfill the plan. For example if there is nothing for cutting
the goal "cut the apple" cannot be reached. Which tool would you use? 
End your answer with {SELECTED_TOOL}<TOOL> <PARAMETERS>. 
End your answer with {FAILURE} if no tool is valid.
"""#TODO was_correction_made tool aufrufen
    response = llm_instance.prompt(formatted_prompt)
    return parse_response(response, tool_selector)
