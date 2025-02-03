from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from typing import Dict, Any, List
import logging
from src.llm_manager import LanguageModelFactory
from core_web.django_storage import StorageManager
from src.configs import ChatStorageType, WorkflowType


# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class State(MessagesState):
    """State class that extends MessagesState to include summary"""
    summary: str



class ChatBotWorkflowBuilder:
    """Builds the conversation workflow"""
    def __init__(self, model):
        self.model = model
        self.memory_saver = MemorySaver()

    def _call_model(self, state: State) -> Dict[str, List[AIMessage]]:
        """Call the model with the current state"""
        logger.info("Calling model with current state")
        summary = state.get("summary", "")
        if summary:
            system_message = f"Summary of conversation earlier: {summary}"
            messages = [SystemMessage(content=system_message)] + state["messages"]
        else:
            messages = state["messages"]
            
        formatted_messages = [(msg.type, msg.content) for msg in messages]
        response = self.model.generate_response(formatted_messages)
        ai_message = AIMessage(content=response.content)
        logger.info(f"Model response received: {response.content}...")
        return {"messages": [ai_message]}

    def _should_continue(self, state: State):
        """Return the next node to execute."""
        message_count = len(state["messages"])
        decision = "summarize_conversation" if message_count > 6 else END
        logger.info(f"Conversation flow check: {message_count} messages in state, decision: {decision}")
        return decision

    def _summarize_conversation(self, state: State) -> Dict[str, Any]:
        """Summarize the conversation state"""
        logger.info("Summarizing conversation")
        summary = state.get("summary", "")

        if summary:
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_message = "Create a summary of the conversation above:"

        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = self.model.invoke(messages)

        logger.info(f"state messages: {state['messages']}\n")
        logger.info(f"All ids available: {[m.id for m in state['messages']]}\n")
        logger.info(f"Finally kept ids: {[m.id for m in state['messages'][-2:]]}\n")
        
        # Keep only the last two messages
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        logger.info(f"Summary generated: {response.content}...")
        logger.info(f"messages after Summary generated: {delete_messages}...")
        return {"summary": response.content, "messages": delete_messages}

    def build(self) -> StateGraph:
        """Create and return the workflow graph"""
        workflow = StateGraph(State)
        
        workflow.add_node("conversation", self._call_model)
        workflow.add_node("summarize_conversation", self._summarize_conversation)
        
        workflow.add_edge(START, "conversation")
        workflow.add_conditional_edges(
            "conversation",
            self._should_continue,
        )
        workflow.add_edge("summarize_conversation", END)
        
        return workflow.compile(checkpointer=self.memory_saver)

class BotBuilder:
    """Builder for constructing ChatBot instances"""
    def __init__(self):
        self.with_provider = None
        self.storage = None
        self.workflow = None
        self.temperature = None

    def with_model(self, provider: str, model_name: str):
        self.model = LanguageModelFactory.create_model(
            provider=provider,
            model_name=model_name
        )
        return self

    def with_storage(self, storage_type: ChatStorageType):
        self.storage = StorageManager(storage_type=storage_type)
        return self

    def with_workflow(self, workflow_type: WorkflowType):
        """Set up the workflow based on the provided type"""
        if not self.model:
            raise ValueError("Model must be initialized before creating workflow")
            
        match workflow_type:
            case WorkflowType.CHAT:
                workflow_builder = ChatBotWorkflowBuilder(self.model)
            case _:
                raise ValueError(f"Unsupported workflow type: {workflow_type}")
        
        self.workflow = workflow_builder.build()
        return self

    def with_temperature(self, temperature: float):
        self.temperature = temperature
        return self

    def _validate_components(self) -> None:
        """validate all required components before building the bot"""
        missing_components = []
        
        if not self.model:
            missing_components.append("model")

        if not self.storage:
            missing_components.append("storage")

        if not self.workflow:
            missing_components.append("workflow")

        if not isinstance(self.temperature, (int, float)):
            missing_components.append("temperature")
            
        if missing_components:
            raise ValueError(f"Missing required components: {', '.join(missing_components)}")

    def build(self) -> 'Bot':
        """validate all components and build the bot"""
        self._validate_components()

        return Bot(
            model=self.model,
            storage=self.storage,
            workflow=self.workflow,
            temperature=self.temperature
        )

class Bot:
    """Refactored ChatBot class that uses builder pattern"""
    def __init__(self, model, storage, workflow, temperature: float = 0):
        self.model = model
        self.storage = storage
        self.workflow = workflow
        self.temperature = temperature

    def chat(self, message: str, thread_id: str) -> str:
        """Process a single chat message and return the response"""
        logger.info(f"Processing chat message for thread {thread_id}")
        logger.info(f"User message: {message}")
        config = {"configurable": {"thread_id": thread_id}}
        
        existing_messages = []
        input_message = HumanMessage(content=message)
        all_messages = existing_messages + [input_message]
        
        response = self.workflow.invoke(
            {"messages": all_messages},
            config=config
        )
        
        ai_response = None
        if response and "messages" in response:
            for msg in response["messages"]:
                if isinstance(msg, AIMessage):
                    ai_response = msg
        
        if ai_response:
            logger.info(f"AI response for thread {thread_id}: {ai_response.content}")
            return ai_response.content
        
        logger.warning("Failed to generate AI response")
        return "I apologize, but I couldn't generate a response."