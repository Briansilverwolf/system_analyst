from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import PersistentDict
from langchain_core.tools import tool,BaseTool
from langchain.agents import create_react_agent
from typing import Annotated, Any
from langgraph.graph import add_messages
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from llm import chat_model
from langgraph.prebuilt import ToolNode

class SqliteStore:

    def __init__(self, data_file = "store_data.pkl", vector_file = "store_vectors.pkl"):   
        self.store = InMemoryStore()
        self.store._data = PersistentDict(filename=data_file)
        self.store._vectors = PersistentDict(filename=vector_file)

    def put(self,namespace,key,value):
        self.store.put(namespace,key,value)
    
    def search(self,namespace,**kwargs):
        self.store.search(namespace, **kwargs)

long_term_store = SqliteStore()
short_term_store =SqliteSaver.from_conn_string("checkpointer.sqlite")



@tool
def sum_tool(x:int, y:int):
    "sum two interger number x and y"
    return x+y

chat_model.bind_tools([sum_tool],tool_choice="auto")
tools =[sum_tool]
class chat(BaseModel):
    name: str = Field(...,description="name of the person(human) who provide advice")
    advice: str = Field(...,description= "The advice the person provide")


class Messages(BaseModel):
    input:str
    messages: Annotated[list[Any], add_messages]
    tool_name:str
    tools: list[BaseTool]
    

State = StateGraph(Messages)

def call_llm(state:State):
    chat =chat_model.invoke([
    SystemMessage(content="you are a professional, you respond to use request and use tools as per the request"),
    HumanMessage(content=state["input"])
    ]+state["messages"])
    return {"messages":AIMessage(content=chat.content)}

def condition(state:State):

    lst_message = state["messages"][-1]
    if getattr(lst_message, "tool_calls", None):
        return "tool_call"
    else: 
        return END
        
tool_call =ToolNode(tools)
 
State.add_node("call_llm",call_llm)
State.add_node("tool_call",tool_call)

State.add_edge(START,"call_llm")
State.add_conditional_edges("call_llm",condition,{"tool_call":"tool_call", END:END})
State.add_edge("tool_call","call_llm")

agent = State.compile()


agent.get_graph(xray=True).draw_mermaid_png(output_file_path="agent.png")