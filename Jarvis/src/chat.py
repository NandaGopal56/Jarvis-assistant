from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage, BaseMessage, AnyMessage
from dotenv import load_dotenv
from typing_extensions import TypedDict
import traceback
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from typing import Dict, Any, List, Annotated
import logging
from src.llm_manager import LanguageModelFactory
from core_web.django_storage import StorageManager
from src.configs import ChatStorageType, WorkflowType
from core_web.models import MessagePair
import time

logger = logging.getLogger(__name__)

class State(MessagesState):
    """State class that extends MessagesState to include summary"""
    # messages: list[BaseMessage]
    summary: str
    thread_id: int



class ChatBotWorkflowBuilder:
    """Builds the conversation workflow"""
    def __init__(self, model, storage):
        self.model = model
        self.memory_saver = MemorySaver()
        self.workflow = None
        self.storage = storage

    def _memory_state_update(self, state: State) -> Dict[str, List[AIMessage]]:
        """Update the memory state"""
        print(f"state in memory_state_update: {state}\n")

        last_human_message = state.get("messages")[-1]
        print(f"last_human_message: {last_human_message}\n")
        
        thread_id = state.get("thread_id")
            
        # Fetch messages from storage
        existing_messages = []
        summary = ""
        thread_history = self.storage.load_conversation(thread_id, limit=1)
        if thread_history:
            for msg_pair in thread_history:
                if msg_pair.user_message:
                    existing_messages.append(HumanMessage(content=msg_pair.user_message))
                if msg_pair.ai_message:
                    existing_messages.append(AIMessage(content=msg_pair.ai_message))
                if msg_pair.summary:
                    summary = msg_pair.summary


        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]] \
            + existing_messages \
            + [HumanMessage(content=last_human_message.content)]
        
        new_state = {
            "summary": "default summary",
            "messages": delete_messages
        }

        print(f"new_state returned from memory_state_update: {new_state}\n")

        print('-------------------------------------------------\n')
        return new_state

    def _call_model(self, state: State) -> Dict[str, List[AIMessage]]:
        """Call the model with the current state"""
        print(f"no of messages in state in call_model: {len(state['messages'])}\n")
        print(f"messages in state in call_model: {state}\n")

        system_prompt = (
            "You are a helpful AI assistant. "
            "Answer all questions to the best of your ability. "
            "The provided chat history includes a summary of the earlier conversation."
        )

        system_message = SystemMessage(content=system_prompt)
        
        # Add summary if it exists
        if state.get("summary", ""):
            system_message = SystemMessage(
                content=f"{system_prompt}\n\nSummary of conversation earlier: {state['summary']}"
            )


        # Combine messages in order: system message, kept messages, and last message
        question = [system_message] + state["messages"]

        print(f"question: {question}")
        print('-------------------------------------------------')

        # Generate response
        response = self.model.generate_response(question)

        print(f"response: {response}")
        print('-------------------------------------------------')
        
        return {"messages": [response]}

    def _should_continue(self, state: State):
        """Return the next node to execute."""

        message_count = len(state["messages"])
        decision = "summarize_conversation" if message_count > 2 else END

        print(f"no of messages in state in should_continue: {len(state['messages'])}")
        print(f"Conversation flow check: {message_count} messages in state, decision: {decision}")

        return decision

    def _summarize_conversation(self, state: State) -> Dict[str, Any]:
        """Summarize the conversation state"""
        # First, we summarize the conversation
        summary = state.get("summary", "")
        if summary:
            # If a summary already exists, we use a different system prompt
            # to summarize it than if one didn't
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_message = "Create a summary of the conversation above:"

        messages = state["messages"] + [HumanMessage(content=summary_message)]

        response = self.model.generate_response(messages)
        
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        print(f"delete_messages: {delete_messages}")
        return {"summary": response.content, "messages": delete_messages}

    def build(self) -> StateGraph:
        """Create and return the workflow graph"""
        workflow = StateGraph(State)
        
        workflow.add_node("memory_state_update", self._memory_state_update)
        workflow.add_node("conversation", self._call_model)
        workflow.add_node("summarize_conversation", self._summarize_conversation)
        
        workflow.add_edge(START, "memory_state_update")
        workflow.add_edge("memory_state_update", "conversation")
        workflow.add_conditional_edges(
            "conversation",
            self._should_continue,
        )
        workflow.add_edge("summarize_conversation", END)
        
        self.workflow = workflow.compile(checkpointer=self.memory_saver)
        
        return self.workflow
    
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

        if not self.storage:
            raise ValueError("Storage must be initialized before creating workflow")
            
        match workflow_type:
            case WorkflowType.CHATBOT:
                workflow_builder = ChatBotWorkflowBuilder(self.model, self.storage)
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
    """Bot class that uses builder pattern"""
    def __init__(self, model, storage, workflow, temperature: float = 0):
        self.model = model
        self.storage = storage
        self.workflow = workflow
        self.temperature = temperature

    def chat(self, message: str, thread_id: str) -> str:
        """Process a single chat message and return the response"""

        try:
            print('#####################################################################')
            
            # Fetch existing messages from storage
            # existing_messages = []
            # summary = ""
            # thread_history = self.storage.load_conversation(thread_id)
            # if thread_history:
            #     for msg_pair in thread_history:
            #         if msg_pair.user_message:
            #             existing_messages.append(HumanMessage(content=msg_pair.user_message))
            #         if msg_pair.ai_message:
            #             existing_messages.append(AIMessage(content=msg_pair.ai_message))
            #         if msg_pair.summary:
            #             summary = msg_pair.summary
            
            input_message = HumanMessage(content=message)
            # all_messages = existing_messages + [input_message]

            # print(f"existing_messages: {existing_messages}")
            # print('-------------------------------------------------')

            config = {"configurable": {"thread_id": thread_id}}
            
            # Get start time for processing
            start_time = time.time()

            # Clear any existing state first
            # self.workflow.update_state(config=config, 
            #                            values={"messages": existing_messages, "summary": summary, "thread_id": thread_id}, 
            #                            as_node="conversation"
            #                         )

            # print("Cleared existing state")
            # print(f"State after clearing: {self.workflow.get_state(config=config)}")
            # print('-------------------------------------------------')

            response = self.workflow.invoke(
                {"messages": [input_message], "thread_id": thread_id},
                config=config
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            ai_response = None
            if response and "messages" in response:
                for msg in response["messages"]:
                    if isinstance(msg, AIMessage):
                        ai_response = msg
            
            # Prepare response content and status
            response_content = ai_response.content if ai_response else "I apologize, but I couldn't generate a response."
            status = "completed" if ai_response else "error"
            error_message = "" if ai_response else "Failed to generate AI response"
            
            # Create message data for storage
            message_data = MessagePair(
                user_message=message,
                ai_message=response_content,
                summary=response.get("summary", None) if response else None,
                tokens_used={},  # You might want to add actual token counting
                model_version=self.model.model_name if hasattr(self.model, 'model_name') else "",
                status=status,
                processing_time=processing_time,
                error_message=error_message
            )
            
            # Save to storage
            storage_success = self.storage.save_message(thread_id, message_data)
            if not storage_success:
                logger.error(f"Failed to save message to storage for thread {thread_id}, storage_success: {storage_success}")
            
            return response_content
        
        except Exception as e:
            print(f"Error in chat: {e}")
            print(traceback.format_exc())
            return "I apologize, but I couldn't generate a response."