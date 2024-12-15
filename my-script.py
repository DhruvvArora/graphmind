from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from typing import Annotated
import functools
from dotenv import load_dotenv
import os
import sys
from pydantic import BaseModel
from typing import Literal

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)  


from langchain_core.messages import HumanMessage

def agent_node(state, agent, name):
    result = agent.invoke(state)
    
    if "messages" in result and result["messages"]:
        last_message = result["messages"][-1]
        message_content = last_message.content
        return {
            "messages": [HumanMessage(content=message_content, name=name)]
        }
    else:
        print(f"{name} agent did not return a message.")
        return {"messages": []}

system_prompt = (
    "You are a medical assistant providing the information related to the medical field."
    "You are also a supervisor tasked with managing a conversation between the following agents: {members}. "
    "Each agent provides specific information related to its expertise area in response to user queries. "
    "For example, MedicineAgent should give medical advice within general advice limitations, MedicalHospitalAgent should suggest hospitals based on location, and MedicalDepartmentAgent should offer relevant departmental options."
)

members = ["GreetingAgent", "FarewellAgent", "MedicineAgent", "MedicalHospitalAgent", "MedicalDepartmentAgent"]

class routeResponse(BaseModel):
    next: Literal["FINISH", "GreetingAgent", "FarewellAgent", "MedicineAgent", "MedicalHospitalAgent", "MedicalDepartmentAgent"]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next? "
            "Or should we FINISH? Select one of: ['GreetingAgent', 'FarewellAgent', 'MedicineAgent', 'MedicalHospitalAgent', 'MedicalDepartmentAgent', 'FINISH']",
        ),
    ]
).partial(members=", ".join(members))

def UserProxyAgent(state):
    return {"messages": state["messages"]}

def OperatorAgent(state):
    supervisor_chain = prompt | llm.with_structured_output(routeResponse)
    response = supervisor_chain.invoke(state)
    
    if hasattr(response, 'next'):
        print(f"Operator selected: {response.next}")
        return {"next": response.next, "messages": state.get("messages", [])}
    else:
        print("Error: 'next' not found in operator response.")
        return {"next": "FINISH", "messages": state.get("messages", [])}  

from langchain_core.messages import HumanMessage
from langchain.tools import tool

@tool
def greeting_tool() -> str:
    """Fetches a greeting message from OpenAI."""
    response = llm.invoke({"prompt": "Greet the user warmly and offer assistance, without any disclaimers."})
    return response["choices"][0]["message"]["content"]

@tool
def farewell_tool() -> str:
    """Fetches a farewell message from OpenAI."""
    response = llm.invoke({"prompt": "Say goodbye to the user in a friendly and polite manner, without any disclaimers."})
    return response["choices"][0]["message"]["content"]

from langchain_core.prompts import ChatPromptTemplate

# Define the system message
system_instruction = "You are a confident and direct medical assistant. Provide information clearly without disclaimers or hesitation."

@tool
def medicine_tool() -> str:
    """Fetches information about general medicines for common symptoms like fever or headache."""
    # Create a prompt with the system message and the specific request
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("user", "Provide a clear and direct list of commonly recommended over-the-counter medications for treating fever. Focus solely on listing common medications for fever without disclaimers.")
    ])
    
    response = llm.invoke(prompt_template())
    return response["choices"][0]["message"]["content"]



@tool
def medical_hospital_tool() -> str:
    """Fetches information about nearby hospitals based on general location data."""
    response = llm.invoke({
        "prompt": (
            "Provide detailed information about general hospitals that offer a range of medical services. "
            "Avoid disclaimers and focus on listing the types of hospitals typically found in urban areas."
        )
    })
    return response["choices"][0]["message"]["content"]

@tool
def medical_department_tool() -> str:
    """Fetches information about medical departments available in a hospital."""
    response = llm.invoke({
        "prompt": (
            "List common medical departments in a hospital and briefly describe their primary functions. "
            "Avoid any disclaimers and focus directly on department details."
        )
    })
    return response["choices"][0]["message"]["content"]


GreetingsAgent = create_react_agent(llm, tools=[greeting_tool])
GreetingsNode = functools.partial(agent_node, agent=GreetingsAgent, name="GreetingAgent")

FarewellAgent = create_react_agent(llm, tools=[farewell_tool])
FarewellNode = functools.partial(agent_node, agent=FarewellAgent, name="FarewellAgent")

MedicineAgent = create_react_agent(llm, tools=[medicine_tool])
MedicineNode = functools.partial(agent_node, agent=MedicineAgent, name="MedicineAgent")

MedicalHospitalAgent = create_react_agent(llm, tools=[medical_hospital_tool])
MedicalHospitalNode = functools.partial(agent_node, agent=MedicalHospitalAgent, name="MedicalHospitalAgent")

MedicalDepartmentAgent = create_react_agent(llm, tools=[medical_department_tool])
MedicalDepartmentNode = functools.partial(agent_node, agent=MedicalDepartmentAgent, name="MedicalDepartmentAgent")

workflow = StateGraph(AgentState)
workflow.add_node("UserProxy", UserProxyAgent)
workflow.add_node("GreetingAgent", GreetingsNode)
workflow.add_node("FarewellAgent", FarewellNode)
workflow.add_node("Operator", OperatorAgent)
workflow.add_node("MedicineAgent", MedicineNode)
workflow.add_node("MedicalHospitalAgent", MedicalHospitalNode)
workflow.add_node("MedicalDepartmentAgent", MedicalDepartmentNode)

conditional_map = {
    "GreetingAgent": "GreetingAgent",
    "FarewellAgent": "FarewellAgent",
    "MedicineAgent": "MedicineAgent",
    "MedicalHospitalAgent": "MedicalHospitalAgent",
    "MedicalDepartmentAgent": "MedicalDepartmentAgent",
    "FINISH": END
}
workflow.add_conditional_edges("Operator", lambda x: x["next"], conditional_map)

workflow.add_edge(START, "UserProxy") 
workflow.add_edge("UserProxy", "Operator")

graph = workflow.compile()

print("I am a Medical Chatbot configured by LangGraph. (type 'exit' to end with assistance)")
user_messages = []

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        print("Exiting the chatbot. Goodbye!")
        break
    
    user_messages.append(HumanMessage(content=user_input))
    
    for state in graph.stream({"messages": user_messages}):
        if "__end__" not in state:
            for agent_name, agent_response in state.items():
                if "messages" in agent_response:
                    message_content = agent_response["messages"][-1].content
                    print(f"{agent_name}: {message_content}")
                    user_messages.append(agent_response["messages"][-1])
                else:
                    print(f"No message content available in response from {agent_name}.")
            print("--------------------------------------------------------------------")
