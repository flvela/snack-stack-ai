# Snack Stack AI Assisstant
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B.svg)
![LangChain](https://img.shields.io/badge/agent-LangChain-1C3C3C.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Agents-2C3E50)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Database-orange)

> A voice enabled multi-agent food delivery assistent. It accepts queries through text or voice from a user. It specializes in menu searches and order status tracking

## Table of Contents
1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [Quick Start](#-quick-start)
4. [Project Structure](#project-structure)

## Overview
A multi-agent food delivery and ordering assistent for a fictional company called SnackStack. The system performs the following operations:
1. Accepts natural language queries via text input (and optionally voice)
2. Routes queries to the correct specialist agent(s) using an LLM-powered orchestrator
3. Searches a menu catalog using semantic search (RAG with ChromaDB)
4. Looks up order status by Order ID, Tracking ID, or email
5. Asks the user for missing information when needed (Human-in-the-Loop)
6. Merges responses from multiple agents into a single friendly reply
7. Optionally supports voice input (Whisper STT) and voice output (OpenAI TTS)


## Tech Stack

| Layer              | Tool                                                     | Why it is here                                                              |
| ------------------ | -------------------------------------------------------- | --------------------------------------------------------------------------- |
|UI                  | [Streamlit](https://streamlit.io)                        | Zero-boilerplate web app. One file, top to bottom.                          |
|Agent framework     | [LangChain](https://www.langchain.com)                   | Generic way of creating AI agents and tools. Supports OpenAI, Anthropic, Google and more |
|Agent orchestration | [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)| Orchestration framework and runtime for managing, build and deploying stateful agents|
|Vector DB           | [ChromaDB](https://docs.trychroma.com/) | Open source vector database supporting semantic and metadata search

## 🚀 Quick Start

### 1. Create a virtual environment (recommended)

**Windows (PowerShell):**
```powershell
py -m venv .venv
```

**macOS/Linux:**
```bash
python3 -m venv .venv
```

### 2. Activate the virtual environment

**Windows (PowerShell):**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```bat
.venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -e .
```


### 4. Configure Model and Embeddings Provider

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**macOS/Linux:**
```bash
cp .env.example .env
```

Then edit `.env` and set model and embeddings config. For an example:
```env
MODEL_PROVIDER=anthropic
MODEL_API_KEY=your_anthropic_api_key
MODEL=anthropic/claude-haiku-4-5-20251001
EMBEDDINGS_MODEL_PROVIDER=huggingface
EMBEDDINGS_MODEL_API_KEY=your_hugging_face_api_key
EMBEDDINGS_MODEL=google/embeddinggemma-300m
```
>&#128161; Model providers are based on [LangChain providers](https://reference.langchain.com/python/langchain/chat_models/base/init_chat_model#parameters). Currently only anthropic and openai are tested. 

>&#128161; Embeddings providers are based on [LangChain providers](https://reference.langchain.com/python/langchain/embeddings/base/init_embeddings#parameters). Currently only hugging face and openai implemented.

To test your config see [unit tests](#6-Run-Tests)


### 5. Build and Install locally
#### 5.1 Regular install
```bash
python3 -m build
pip install .
```

#### 5.2 Editable install
Changes to source code reflect instantly with no re-install
```bash
python3 -m build
pip install -e .
```

### 6. Run Tests
#### 6.1 Test your configuration
This is a recommended test to ensure your .env is properly configured
```bash
pytest tests/tools/test_config.py
```

#### 6.2 Run All tests
```bash
pytest
```

### 7. Run Streamlit application
```
streamlit run src/streamlit_pages/app.py
```

## Project Structure
```
SNACK-STACK-AI/                   #entire application
  .github/workflows               #github repo actions or workflows
    python-app.yml                #python pull request action for building, testing and code coverage
  data/                           #contains sample data used by AI assisstant
    menu.json                     #sample menu items
    orders.json                   #sample orders
  src/                            #python source code
    agents/                       #python code related to LangGraph node, context schema and state
      __init__.py                 #package init file
      context_schema.py           #LangGraph runtime context schema defining shared objects used by the nodes (ie. llm, tools, etc...)
      menu_agent.py               #menu agent and prompt
      orchestrator.py             #orchestrator agent and prompt
      order_agent.py              #order agent and prompt
      state.py                    #the shared state that is passed between the LangGraph Nodes
      synthesizer.py              #synthesizer agent that puts together the output from menu and order agent
    jupyter/                      #contains jupyter notebooks
      graph_test.ipynb            #jupyter notebook (playground) for testing the snack-stack-ai app
    streamlit_pages/              #streamlit application
      __init__.py                 #package init file
      app.py                      #streamlit main page and app
      chat.py                     #streamlin chat page with AI assistant conversation and Graph activity
      common.py                   #common methods and definitions
      constants.py                #streamlit constants used by app.py
      graph_containers.py         #classes that initialize and populate streamlit containers with LangGraph graph nodes, state and events. It is reusable for other graphs
      menu.py                     #simple streamlit page displaying SnackStack menu used by AI assistant
      snack_stack_containers.py   #concrete implementation for SnackStack AI graph containers 
    testutils/                    #folder for unit testing utils and tools
      __init__.py                 #package init file
      common.py                   #common unit testing functions
    tools/                        #tools used for the agents to import sample data and config
      __init__.py                 #package init file
      common.py                   #common tools that can be used by any agent
      config.py                   #gets the embeddings and chat model from the config .env file
      menu.py                     #loads the menu sample items and provides the LangGraph menu search tool
      orders.py                   #loads the orders sample items and provides the LangGraph orders search tool
      vector_store.py             #Chroma DB store functionality for local persistence
    __init__.py                   #package init file
    assistant.py                  #defines the SnackStackAssistant
    snack_stack_graph.py          #the snack stack graph definition
  tests/                          #python unit tests using pytest
    agents/                       #agent unit tests 
      test_menu_agent.py          #menu agent tests
      test_orchestrator.py        #orchestrator agent tests
      test_order_agent.py         #order agent tests
      test_synthesizer.py         #synthesizer agent tests
    tools/                        #tools unit tests
      test_config.py              #config unit tests
      test_menu.py                #menu unit tests
      test_orders.py              #orders unit tests
      test_vector_store.py        #vector store unit tests
    test_snack_stack_graph.py     #snack stack graph tests
  .env.example                    #example config file
  .gitignore                      #git ignore file
  LICENSE                         #license information
  pyproject.toml                  #python project file
  README.md                       #this file 
```
