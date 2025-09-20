from deepagents import SubAgent
from langchain_core.tools import tool
from base import SystemRequest,FeasibilityAnalysis,RequirementDefinition,UseCaseDiagram, ActivityDiagram, UseCaseDescription
import json

@tool
def generate_systemRequest(request:SystemRequest):
    """use this tool after you have well analyzed the business need and you have all the fields that are required to create this 
        system request
    """
    with open("systemrequest.txt","w",encoding="utf-8") as f:
        f.write(str(request)) 
    return request

@tool
def generate_feasibility(feasibility:FeasibilityAnalysis):
    """
    use this tool to check if the system requestted is feasible, This tools can only be used after the system request is created 
    or when the user specifically request so.
    """
    with open("feasibility.txt","w",encoding="utf-8") as f:
        f.write(str(feasibility)) 
    return feasibility

def generate_requirement(requirements:RequirementDefinition):
    """
    
    """
    with open("feasibility.txt","w",encoding="utf-8") as f:
        f.write(str(requirements)) 
    return requirements

analyst = SubAgent(
    name="Analyst",
    description ="""
            This agent is an analyst that analyze the business need or  the business problem need and convert it to a system 
            request though analysis
            """,
    prompt="""
            You are a professional system analyst who accept the business need or  the business problem need and convert it 
            to a system request through well thought of analysis create a system request 
            """,
    tools=["generate_systemRequest","generate_feasibility","generate_requirement"]    
    )

@tool
def generate_useCaseDiagram(usecasediagram:UseCaseDiagram):
    """

    """
    with open("feasibility.txt","w",encoding="utf-8") as f:
        f.write(str(usecasediagram)) 
    return usecasediagram

@tool
def generate_activityDiagram(activitydiagram:ActivityDiagram):
    """
    
    """
    with open("feasibility.txt","w",encoding="utf-8") as f:
        f.write(str(activitydiagram)) 
    return activitydiagram

@tool
def generate_useCaseDescription(usecasedescription:UseCaseDescription):
    """
    
    """
    with open("feasibility.txt","w",encoding="utf-8") as f:
        f.write(str(usecasedescription)) 
    return usecasedescription

analyst_model =SubAgent(
    name= "model_analysis",
    description ="""
    This agent analyze the business request and generate the Functional Models, This include the 
        1. Use Case Diagram
        2. Activity Diagram
        3. Use case Description
    """,
    prompt = """
    You are a professional Functional Modeling Architecture you have The following tools to use
        1. Use Case Diagram tool
        2. Activity Diagram tool
        3. Use case Description tool
    You use the unified process to cordinate with this tools to make sure you craft the project Functional Model
    """,
    tools=["generate_useCaseDescription","generate_activityDiagram","generate_useCaseDiagram"]
)

