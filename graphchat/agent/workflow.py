from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Literal, Union, Annotated,Optional ,Any
from base import Sponsor,BusinessRequirement,BusinessValue, SpecialIssue, SystemRequest
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage
from langgraph.graph import add_messages
from langgraph.types import Send 
import operator


import time
from llm import chat_model, Zhipullm, deep_chat

class RequestState(BaseModel):
    input: str
    messages: Annotated[list[BaseMessage],add_messages]
    stage: Literal["clarity","varied", "sponsor","insight", "finalize"] = "clarity"
    problem_definition: str = Optional[str]
    business_need : str = Optional[str]
    sponsors : list[Sponsor] =[]
    sponsors_insight:Annotated[dict[str,list[Union[BusinessRequirement,BusinessValue,SpecialIssue]]],operator.add] = Optional[dict[Any,list[Any]]]
    system_request : SystemRequest =[]

class SponsorState(BaseModel):
    sponsor: Sponsor |None = None
    business_need:str |None = None
    sponsors_insight:Annotated[dict[str,list[Union[BusinessRequirement,BusinessValue,SpecialIssue]]],operator.add] = Optional[dict[Any,list[Any]]]

State = StateGraph(RequestState)
parallelstate = StateGraph(SponsorState)

def problem_definition(state:State):


    #chat = 
    #return {"stage":"varied","input":chat.question,"problem_definition":chat.problem}
    pass

def business_need(state:State):
    class Need(BaseModel):
        need: str = Field(...,description="a detailed business need description")

    struct_output=Zhipullm.with_structured_output(Need)
    chat = struct_output.invoke([
        SystemMessage(content=""),
        HumanMessage(content=state.problem_definition)
    ])
    return {"stage":"sponsor","business_need":chat.need}

def condition(state:State):
    x= "some value"
    if x:
        return "problem_definition"
    else:
        return "business_need"

def sponsor_determination(state:State):
    class Sponsors(BaseModel):
        sponsors: list[Sponsor]
    """
    1. add the prompt template and values(context) to te system prompt.
    2. add the human message that the agent is supposssed to respond to.
    """
    struct_output=Zhipullm.with_structured_output(Sponsors)
    chat = struct_output.invoke([
        SystemMessage(content=""),
        HumanMessage(content="")
    ])
    return {"stage":"insight","sponsors":chat.sponsors}


def router(state:State):
    return [Send("requirements",{"sponsor":sponsor_data}) for sponsor_data in state["sponsors"] ]

def requirements_determination(state:parallelstate): 
    class Requirements(BaseModel):
        requirements : list[BusinessRequirement]
    """
    1. add the prompt template and values(context) to te system prompt.
    2. add the human message that the agent is supposssed to respond to.
    """
    struct_output=Zhipullm.with_structured_output(Requirements)
    chat = struct_output.invoke([
        SystemMessage(content=""),
        HumanMessage(content="")
    ])
    return {"sponsors_insight":{state.sponsor:chat.requirements}}

def value_determination(state:parallelstate):
    class Values(BaseModel):
        values : list[BusinessValue]
    """
    1. add the prompt template and values(context) to te system prompt.
    2. add the human message that the agent is supposssed to respond to.
    """
    struct_output=Zhipullm.with_structured_output(Values)
    chat = struct_output.invoke([
        SystemMessage(content=""),
        HumanMessage(content="")
    ])
    return {"sponsors_insight":{state.sponsor:chat.values}}

def specialissues(state:parallelstate):
    class Constraints(BaseModel):
        constraints : list[SpecialIssue]
    """
    1. add the prompt template and values(context) to te system prompt.
    2. add the human message that the agent is supposssed to respond to.
    """
    struct_output=Zhipullm.with_structured_output(Constraints)
    chat = struct_output.invoke([
        SystemMessage(content=""),
        HumanMessage(content="")
    ])
    return {"sponsors_insight":{state.sponsor:chat.constraints}}

    
State.add_node("problem_definition",problem_definition)
State.add_node("business_need",business_need)
State.add_node("sponsors",sponsor_determination)


parallelstate.add_node("requirements",requirements_determination)
parallelstate.add_node("values",value_determination)
parallelstate.add_node("constraints",specialissues)


parallelstate.add_edge(START,"requirements")
parallelstate.add_edge("requirements","values")
parallelstate.add_edge("values","constraints")
parallelstate.add_edge("constraints",END)

requrement_graph=parallelstate.compile()
State.add_node("requrement_graph",requrement_graph)

State.add_edge(START,"problem_definition")
State.add_conditional_edges("problem_definition",condition,{"problem_definition":"problem_definition","business_need":"business_need"})
State.add_edge("business_need","sponsors")

State.add_conditional_edges("sponsors",router, ["requrement_graph"])
State.add_edge("requrement_graph",END)

stage1 = State.compile()
stage1.get_graph(xray=True).draw_mermaid_png(output_file_path="stage1.png",frontmatter_config={"config":{"theme": "neo-dark"}})

import pytest