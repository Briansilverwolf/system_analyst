from llm import Zhipullm, chat_model, deep_chat, Zhipullm
from langchain_core.messages import SystemMessage, HumanMessage
from deepagents import create_deep_agent, SubAgent,tools
from rich.console import Console
from langgraph.checkpoint.memory import InMemorySaver
from analyst import (
    analyst,analyst_model,
    generate_systemRequest,generate_feasibility,generate_requirement,
    generate_useCaseDescription, generate_activityDiagram,generate_useCaseDiagram
    )
from rich.panel import Panel
console = Console()
from test_cli import clean_output

checkpointer = InMemorySaver()

agent = create_deep_agent(
    model=Zhipullm,
    instructions="You are an expert system designer and development, you have keen thinking when crafting systems",
    tools=[
        generate_systemRequest,
        generate_feasibility,
        generate_requirement,
        generate_useCaseDescription,
        generate_activityDiagram,
        generate_useCaseDiagram
        ],
    subagents=[analyst,analyst_model],
    config_schema={"configurable":{"thread_id":"session_1"}}
)
agent.checkpointer=checkpointer
console.print(Panel("UI/UX Design Agent", style="blue"))

while True:
    query = input("\nQuery: ").strip()
    
    if query.lower() in ['exit', 'quit', 'q']:
        break
        
    if not query:
        continue
    
    try:
        response = agent.invoke({"messages": query},config={"configurable":{"thread_id":"session_1"}})
        clean_text = clean_output(response)
        
        console.print(Panel(clean_text, title="Response", style="green"))
        
    except Exception as e:
        console.print(Panel(f"Error: {e}", style="red"))