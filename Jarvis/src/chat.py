from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from typing import Literal, Dict, Any, List
import os
import logging
from datetime import datetime
from pathlib import Path
import json

# Load environment variables
load_dotenv()

# Set up logging
def setup_logger():
    """Configure and return a logger instance"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"chatbot_{timestamp}.log"
    
    # Configure logger
    logger = logging.getLogger("chatbot")
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    
    # Format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logger()

class State(MessagesState):
    """State class that extends MessagesState to include summary"""
    summary: str

class ChatBot:
    def __init__(self, model_name: str = "llama3-8b-8192", temperature: float = 0):
        """Initialize the chatbot with model configuration"""
        self.model = ChatGroq(model_name=model_name, temperature=temperature)
        self.memory = MemorySaver()
        self.workflow = self._create_workflow()
        self.message_history: Dict[str, List[Dict]] = {}
        # Create chat_history directory if it doesn't exist
        self.history_dir = Path("chat_history")
        self.history_dir.mkdir(exist_ok=True)
        
    def _get_history_file(self, thread_id: str) -> Path:
        """Get the JSON file path for a specific thread"""
        return self.history_dir / f"{thread_id}.json"
        
    def _load_message_history(self, thread_id: str) -> List[Dict]:
        """Load message history from JSON file"""
        history_file = self._get_history_file(thread_id)
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error reading history file for thread {thread_id}")
                return []
        return []

    def _save_message_history(self, thread_id: str, messages: List[Dict]):
        """Save message history to JSON file"""
        history_file = self._get_history_file(thread_id)
        
        # Convert messages to serializable format
        message_data = []
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage)):
                message_data.append({
                    'role': 'human' if isinstance(msg, HumanMessage) else 'ai',
                    'content': msg.content,
                    'timestamp': datetime.now().isoformat(),
                })
        
        try:
            # Load existing messages
            existing_data = self._load_message_history(thread_id)
            # Append new messages
            existing_data.extend(message_data[-2:])  # Add only the latest human-AI exchange
            
            # Save updated history
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving chat history for thread {thread_id}: {str(e)}")

    def _call_model(self, state: State) -> Dict[str, List[AIMessage]]:
        """Call the model with the current state"""
        logger.info("Calling model with current state")
        summary = state.get("summary", "")
        if summary:
            system_message = f"Summary of conversation earlier: {summary}"
            messages = [SystemMessage(content=system_message)] + state["messages"]
        else:
            messages = state["messages"]
            
        response: AIMessage = self.model.invoke(messages)
        logger.info(f"Model response received: {response.content}...")
        return {"messages": [response]}

    # We now define the logic for determining whether to end or summarize the conversation
    def _should_continue(self, state: State):
        """Return the next node to execute."""
        messages = state["messages"]
        message_count = len(messages)
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

    def _create_workflow(self) -> StateGraph:
        """Create the conversation workflow graph"""
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("conversation", self._call_model)
        workflow.add_node("summarize_conversation", self._summarize_conversation)
        
        # Add edges
        workflow.add_edge(START, "conversation")
        workflow.add_conditional_edges(
            "conversation",
            self._should_continue,
        )
        workflow.add_edge("summarize_conversation", END)
        
        return workflow.compile(checkpointer=self.memory)

    def chat(self, message: str, thread_id: str = "default") -> str:
        """Process a single chat message and return the response"""
        logger.info(f"Processing chat message for thread {thread_id}")
        logger.info(f"User message: {message}")
        config = {"configurable": {"thread_id": thread_id}}
        
        # Load existing messages for this thread
        existing_messages = []
        for msg_data in self._load_message_history(thread_id):
            if msg_data['role'] == 'human':
                existing_messages.append(HumanMessage(content=msg_data['content']))
            else:
                existing_messages.append(AIMessage(content=msg_data['content']))
        
        # Add new user message
        input_message = HumanMessage(content=message)
        
        # Combine existing messages with new message
        all_messages = existing_messages + [input_message]
        
        response = self.workflow.invoke(
            {"messages": all_messages},
            config=config
        )
        
        # Extract the AI's response
        ai_response = None
        if response and "messages" in response:
            for msg in response["messages"]:
                if isinstance(msg, AIMessage):
                    ai_response = msg
        
        if ai_response:
            logger.info(f"AI response for thread {thread_id}: {ai_response.content}")
            # Save both the user message and AI response
            self._save_message_history(
                thread_id, 
                [input_message, ai_response]
            )
            return ai_response.content
        
        logger.warning("Failed to generate AI response")
        return "I apologize, but I couldn't generate a response."

def print_message(role: str, content: str) -> None:
    """Print a formatted chat message"""
    print(f"\n{role}: {content}")

def test_memory_functionality(chatbot):
    """Test the chatbot's memory and summary functionality with predefined questions"""
    test_questions = [
        "Hey:",
        "how are you",
        "what is my name",
        "my name is John",
        "what is my name",
        "what is capital of india",
        "what is capital of delhi",
        "what question i asked you earlier",
        "what is my name"
    ]
    
    logger.info("Starting memory functionality test")
    for i, question in enumerate(test_questions, 1):
        print(f"\nTest Question {i}/{len(test_questions)}: {question}")
        response = chatbot.chat(question)
        print_message("Bot", response)

def interactive_chat(chatbot):
    """Run the interactive chat mode"""
    print("\nStarting interactive chat. Type 'quit' to exit.")
    # Main chat loop
    while True:
        # Get user input
        user_message = input("\nYou: ").strip()
        # Check for quit command
        if user_message.lower() in ['quit', 'exit', 'bye']:
            logger.info("User requested to quit the application")
            print("\nGoodbye!")
            break
            
        # Get chatbot response
        if user_message:
            logger.info(f"Processing user message: {user_message}")
            response = chatbot.chat(user_message)
            print_message("Bot", response)
        else:
            logger.warning("Empty message received from user")
            print("Please enter a message.")

def main():
    # Initialize chatbot
    logger.info("Starting chatbot application")
    chatbot = ChatBot()
    
    # Ask if user wants to run test sequence or interactive mode
    print("\nChatbot initialized. Would you like to:")
    print("1. Run test sequence")
    print("2. Start interactive chat")
    choice = input("Enter your choice (1 or 2): ").strip() or 2
    
    return chatbot, choice

if __name__ == "__main__":
    chatbot, choice = main()
    
    if choice == "1":
        test_memory_functionality(chatbot)
    else:
        interactive_chat(chatbot)