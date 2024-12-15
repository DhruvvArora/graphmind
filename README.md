# Medical Chatbot with LangGraph

This project is a medical chatbot designed to assist users with various inquiries, including greeting, farewell, medicine, hospital, and department-related questions. The chatbot leverages **LangGraph** to manage agent workflows and **OpenAI** for generating responses. It routes user queries to specialized agents depending on the intent, using a hierarchical setup with an operator agent to guide the conversation flow.

## Features

- **Agent-based Workflow**: A structured conversation management system using LangGraph.
- **Specialized Agents**: Separate agents for handling greetings, farewells, medicine, medical hospital, and medical department queries.
- **Operator Agent**: Directs user queries to the appropriate agent based on context.
- **State Management**: Maintains conversation state and tracks the flow between agents.
- **User Proxy**: Allows user messages to be processed and routed within the workflow.

## Requirements

- **Python 3.8+**
- **LangGraph** (for creating the chatbot workflow)
- **OpenAI API Key** (for language generation via `ChatOpenAI`)
- **Dotenv** (for managing environment variables)


## Installation Steps
1. **Install the required packages:**
   - `pip install -U langchain langchain_openai langsmith pandas langchain_experimental matplotlib langgraph langchain_core`

2. **Set your OpenAI API Key:**
   - Create a .evn file.
   - In that file, set your OpenAI API key: `OPENAI_API_KEY=your_openai_api_key`


## Specialized Agents
The system employs the following specialized agents:

1. **GreetingAgent**
   - Handles user greetings such as "Hello" or "Hi" and responds with a friendly message.

2. **MedicineAgent**
   - Manages queries related to medicine, prompting users to provide more details about their medicine-related questions.

3. **MedicalHospitalAgent**
   - Responds to questions about hospitals, asking for specific details on what the user needs to know.

4. **MedicalDepartmentAgent**
   - Engages with users asking about medical departments, prompting them to specify which department they're interested in.

5. **FarewellAgent**
   - Responds to farewell messages like "Goodbye" and bids the user a polite farewell.

6. **FallbackAgent**
   - Activated when the system cannot determine the user’s intent. It asks the user to rephrase their query.


## Workflow Structure
The workflow follows a structured state graph where each user input is processed as follows:
   - UserProxy: Receives user input and forwards it to the OperatorAgent.
   - OperatorAgent: Determines the appropriate specialized agent to handle the request.
   - Specialized Agents: Processes the request and responds with relevant information.
   - Response Display: Displays agent responses to the user.

The flow ends when the operator signals FINISH, or the user types exit.


## Usage
   - Once you have the system set up, you can run the code using the following command:
      `python my-script.py`


## IPYNB Script
The project folder also includes my-scripts.ipynb which breakdowns all source code and output of the project.