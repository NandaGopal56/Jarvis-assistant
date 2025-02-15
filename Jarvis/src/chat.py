from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from typing import Dict, Any, List
from src.llm.llm_manager import LanguageModelFactory
from src.storage.chat_storage import StorageManager
from src.configs import ChatStorageType, WorkflowType
from core_web.models import MessagePair, AIChatMessageStatus
import time
import traceback
import logging

logger = logging.getLogger(__name__)

class State(MessagesState):
    """State class that extends MessagesState to include summary"""
    summary: str
    thread_id: int



class ChatBotWorkflowBuilder:
    """Builds the conversation workflow"""
    def __init__(self, model, storage):
        self.model = model
        self.memory_saver = MemorySaver()
        self.workflow = None
        self.storage = storage

    async def _memory_state_update(self, state: State) -> Dict[str, List[AIMessage]]:
        """Update the memory state"""
        logger.info("Starting memory state update")
        logger.debug(f"Initial state: {state}")
        logger.info(f"Initial message count: {len(state.get('messages', []))}")

        last_human_message = state.get("messages")[-1]
        logger.debug(f"Processing last human message: {last_human_message}")
        
        thread_id = state.get("thread_id")
        logger.info(f"Processing thread ID: {thread_id}")
            
        # Fetch messages from storage
        existing_messages = []
        summary = ""
        thread_history = await self.storage.load_conversation(thread_id, limit=1)
        logger.info(f"Retrieved {len(thread_history)} messages from storage")

        if thread_history:
            for msg_pair in thread_history:
                if msg_pair.user_message:
                    existing_messages.append(HumanMessage(content=msg_pair.user_message))
                if msg_pair.ai_message:
                    existing_messages.append(AIMessage(content=msg_pair.ai_message))
                if msg_pair.summary:
                    summary = msg_pair.summary
                    logger.debug(f"Retrieved summary: {summary}")

        # Update state with messages
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]] \
            + existing_messages \
            + [HumanMessage(content=last_human_message.content)]
        
        new_state = {
            "summary": summary,
            "messages": delete_messages
        }

        logger.info(f"Memory state update complete. New message count: {len(delete_messages)}")
        logger.debug(f"Final state after memory update: {new_state}")
        return new_state

    async def _call_model(self, state: State) -> Dict[str, List[AIMessage]]:
        """Call the model with the current state"""
        logger.info("Starting model call")
        logger.info(f"Current message count: {len(state['messages'])}")
        logger.debug(f"Current state: {state}")

        system_prompt = (
            "You are a helpful AI assistant. "
            "Answer all questions to the best of your ability. "
            "The provided chat history includes a summary of the earlier conversation."
        )

        system_message = SystemMessage(content=system_prompt)
        
        # Add summary if it exists
        if state.get("summary", ""):
            logger.debug(f"Including summary in system message: {state['summary']}")
            system_message = SystemMessage(
                content=f"{system_prompt}\n\nSummary of conversation earlier: {state['summary']}"
            )

        # Combine messages
        question = [system_message] + state["messages"]
        logger.debug(f"Prepared question for model: {question}")

        # Generate response
        response = await self.model.generate_response(question)
        logger.info("Model response generated")
        logger.debug(f"Model response: {response}")
        
        return {"messages": [response]}

    async def _should_continue(self, state: State):
        """Return the next node to execute."""
        message_count = len(state["messages"])
        decision = "summarize_conversation" if message_count > 2 else END

        logger.info(f"Workflow decision check - Messages: {message_count}")
        logger.info(f"Decision: {decision}")
        logger.debug(f"Current state at decision point: {state}")

        return decision

    async def _summarize_conversation(self, state: State) -> Dict[str, Any]:
        """Summarize the conversation state"""
        logger.info("Starting conversation summarization")
        logger.info(f"Current message count: {len(state['messages'])}")
        logger.debug(f"State before summarization: {state}")

        summary = state.get("summary", "")
        if summary:
            logger.debug(f"Existing summary found: {summary}")
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            logger.debug("No existing summary found, creating new summary")
            summary_message = "Create a summary of the conversation above:"

        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = await self.model.generate_response(messages)
        
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        
        final_state = {"summary": response.content, "messages": delete_messages}
        logger.info("Summarization complete")
        logger.info(f"Final message count: {len(delete_messages)}")
        logger.debug(f"Final state after summarization: {final_state}")
        
        return final_state

    async def build(self) -> StateGraph:
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

    async def with_model(self, provider: str, model_name: str):
        self.model = await LanguageModelFactory.create_model(
            provider=provider,
            model_name=model_name
        )
        return self

    async def with_storage(self, storage_type: ChatStorageType):
        self.storage = StorageManager(storage_type=storage_type)
        return self

    async def with_workflow(self, workflow_type: WorkflowType):
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
        
        self.workflow = await workflow_builder.build()
        return self

    async def with_temperature(self, temperature: float):
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

    async def build(self) -> 'Bot':
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

    async def chat(self, message: str, thread_id: str) -> str:
        """Process a single chat message and return the response"""
        try:
            logger.info("Starting new chat interaction")
            logger.info(f"Thread ID: {thread_id}")
            logger.debug(f"User message: {message}")
            
            input_message = HumanMessage(content=message)
            config = {"configurable": {"thread_id": thread_id}}
            
            start_time = time.time()

            response = await self.workflow.ainvoke(
                {"messages": [input_message], "thread_id": thread_id},
                config=config
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Processing completed in {processing_time:.2f} seconds")
            
            ai_response = None
            if response and "messages" in response:
                for msg in response["messages"]:
                    if isinstance(msg, AIMessage):
                        ai_response = msg
            
            # Prepare response content and status
            status = AIChatMessageStatus.COMPLETED.value if ai_response else AIChatMessageStatus.FAILED.value
            error_message = "" if ai_response else "Failed to generate AI response"
            response_content = ai_response.content if ai_response else "I apologize, but I couldn't generate a response."
            
            # Create message data for storage
            message_data = MessagePair(
                user_message=message,
                ai_message=response_content,
                summary=response.get("summary", None) if response else None,
                tokens_used={},  #TODO: add actual token counting
                model_version=self.model.model_name if hasattr(self.model, 'model_name') else "",
                status=status,
                processing_time=processing_time,
                error_message=error_message
            )
            
            # Save to storage
            storage_success = await self.storage.save_message(thread_id, message_data)
            if not storage_success:
                logger.error(f"Failed to save message to storage for thread {thread_id}, storage_success: {storage_success}")
        
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            logger.error(traceback.format_exc())
            response_content = "I apologize, but I couldn't generate a response."

        return response_content