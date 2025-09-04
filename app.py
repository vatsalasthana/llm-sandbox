import os, json, yaml, re, datetime, time, uuid
import streamlit as st
from dotenv import load_dotenv
from typing import Optional, Dict, List, Any
import plotly.express as px
from model_router import call_model

# Simple ConversationMemory implementation
class ConversationMemory:
    def __init__(self, max_turns: int = 30):
        self.max_turns = max_turns
        self.messages = []
        self.metadata = {}
    
    def add_message(self, role: str, content: str, **kwargs):
        """Add a message with optional metadata"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat(),
            "metadata": kwargs
        }
        self.messages.append(message)
        
        # Keep only recent messages
        if len(self.messages) > self.max_turns * 2:  # *2 for user+assistant pairs
            self.messages = self.messages[-self.max_turns * 2:]
    
    def get_context(self) -> List[Dict]:
        """Get conversation history in OpenAI format"""
        context = []
        for msg in self.messages:
            if msg["role"] in ["user", "assistant", "system"]:
                context.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        return context
    
    def clear(self):
        """Clear all messages"""
        self.messages = []

# Simple tools implementation
def call_tool(tool_name: str, tool_input: str) -> str:
    """Execute a tool"""
    if tool_name == "calculator":
        try:
            # Simple calculator - be careful with eval in production
            result = eval(tool_input.replace(" ", ""))
            return f"Calculation result: {result}"
        except Exception as e:
            return f"Calculator error: {str(e)}"
    
    elif tool_name == "search":
        return f"Search results for '{tool_input}': [Mock search results - integrate with real search API]"
    
    elif tool_name == "weather":
        return f"Weather for {tool_input}: [Mock weather data - integrate with weather API]"
    
    else:
        return f"Unknown tool: {tool_name}"

# Tool registry
_TOOL_REGISTRY = {
    "calculator": "Perform mathematical calculations",
    "search": "Search the internet for information", 
    "weather": "Get weather information for a location"
}

# Load configuration
load_dotenv()

# Default config if file doesn't exist
DEFAULT_CONFIG = {
    "system_prompt": "You are a helpful assistant.",
    "models": [
        {"name": "GPT-4", "provider": "openai", "id": "gpt-4"},
        {"name": "GPT-3.5", "provider": "openai", "id": "gpt-3.5-turbo"},
        {"name": "Claude-3", "provider": "anthropic", "id": "claude-3-sonnet-20240229"}
    ],
    "personas": [
        {"name": "Helpful Assistant"},
        {"name": "Code Expert"},
        {"name": "Creative Writer"}
    ]
}

try:
    with open("config.yaml", "r") as f:
        CFG = yaml.safe_load(f)
        # Ensure required keys exist
        if not CFG:
            CFG = DEFAULT_CONFIG
        CFG.setdefault("system_prompt", DEFAULT_CONFIG["system_prompt"])
        CFG.setdefault("models", DEFAULT_CONFIG["models"])
        CFG.setdefault("personas", DEFAULT_CONFIG["personas"])
except FileNotFoundError:
    CFG = DEFAULT_CONFIG
    # Create default config file
    with open("config.yaml", "w") as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)

SYSTEM_PROMPT_DEFAULT = CFG.get("system_prompt", "You are a helpful assistant.")
personas = CFG.get("personas", [])
persona_values = [p["name"] for p in personas if "name" in p]

def get_model_names():
    return [m["name"] for m in CFG.get("models", []) if m and "name" in m]

def parse_tool_json(text: str):
    try:
        return json.loads(text)
    except:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                return None
    return None

class ChatHistoryManager:
    """Enhanced file-based chat history manager with visualization"""
    
    def __init__(self, data_dir: str = "chat_data"):
        self.data_dir = data_dir
        self.chats_file = os.path.join(data_dir, "chats.json")
        os.makedirs(data_dir, exist_ok=True)
        self.ensure_data_file()
    
    def ensure_data_file(self):
        """Ensure chat data file exists"""
        if not os.path.exists(self.chats_file):
            with open(self.chats_file, 'w') as f:
                json.dump({"chats": {}}, f)
    
    def load_chats(self) -> Dict:
        """Load all chats from file"""
        try:
            with open(self.chats_file, 'r') as f:
                return json.load(f)
        except:
            return {"chats": {}}
    
    def save_chats(self, data: Dict):
        """Save chats to file"""
        with open(self.chats_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def create_chat(self, metadata: Dict) -> str:
        """Create new chat and return chat ID"""
        chat_id = str(uuid.uuid4())[:8]  # Short ID for readability
        data = self.load_chats()
        
        data["chats"][chat_id] = {
            "id": chat_id,
            "metadata": metadata,
            "messages": [],
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat()
        }
        
        self.save_chats(data)
        return chat_id
    
    def add_message(self, chat_id: str, message_type: str, content: str, metadata: Dict = None):
        """Add message to chat"""
        data = self.load_chats()
        if chat_id not in data["chats"]:
            return
        
        message = {
            "id": str(uuid.uuid4())[:8],
            "type": message_type,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        data["chats"][chat_id]["messages"].append(message)
        data["chats"][chat_id]["updated_at"] = datetime.datetime.now().isoformat()
        self.save_chats(data)
    
    def get_chat(self, chat_id: str) -> Optional[Dict]:
        """Get specific chat"""
        data = self.load_chats()
        return data["chats"].get(chat_id)
    
    def update_chat_metadata(self, chat_id: str, metadata: Dict):
        """Update chat metadata"""
        data = self.load_chats()
        if chat_id in data["chats"]:
            data["chats"][chat_id]["metadata"].update(metadata)
            data["chats"][chat_id]["updated_at"] = datetime.datetime.now().isoformat()
            self.save_chats(data)
    
    def list_chats(self, limit: int = 20) -> List[Dict]:
        """List all chats sorted by update time"""
        data = self.load_chats()
        chats = list(data["chats"].values())
        
        # Sort by updated_at descending
        chats.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return chats[:limit]
    
    def get_chat_stats(self, chat_id: str) -> Dict:
        """Calculate chat statistics"""
        chat = self.get_chat(chat_id)
        if not chat:
            return {}
        
        messages = chat.get("messages", [])
        
        # Count messages by type
        user_msgs = [m for m in messages if m["type"] == "user"]
        assistant_msgs = [m for m in messages if m["type"] == "assistant"]
        tool_msgs = [m for m in messages if m["type"] == "tool"]
        
        # Calculate average response time
        response_times = []
        providers = set()
        models = set()
        
        for msg in assistant_msgs:
            metadata = msg.get("metadata", {})
            if "response_time" in metadata and isinstance(metadata["response_time"], (int, float)):
                response_times.append(metadata["response_time"])
            if "provider" in metadata:
                providers.add(metadata["provider"])
            if "model" in metadata:
                models.add(metadata["model"])
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "total_messages": len(user_msgs) + len(assistant_msgs),
            "user_messages": len(user_msgs),
            "assistant_messages": len(assistant_msgs),
            "tool_calls": len(tool_msgs),
            "avg_response_time": avg_response_time,
            "providers": list(providers),
            "models": list(models)
        }

# Initialize chat history manager
chat_history = ChatHistoryManager()

def generate_chat_name(settings: Dict) -> str:
    """Generate meaningful chat name"""
    models = settings.get("models", [])
    if not models:
        models = []
    elif isinstance(models, str):
        models = [models]
    
    # Filter out None values and ensure we have strings
    models = [str(m) for m in models if m is not None][:2]
    
    persona = settings.get("persona", "")
    
    name_parts = []
    if models:
        name_parts.append(f"Test: {', '.join(models)}")
        if len(settings.get("models", [])) > 2:
            name_parts[-1] += " +more"
    
    if persona:
        name_parts.append(str(persona))
    
    return " | ".join(name_parts) if name_parts else "Model Test"

def restore_memory_from_chat(chat_data: Dict) -> ConversationMemory:
    """Restore ConversationMemory from chat data"""
    memory = ConversationMemory(max_turns=30)
    
    for message in chat_data.get("messages", []):
        msg_type = message.get("type")
        content = message.get("content")
        metadata = message.get("metadata", {})
        
        if msg_type == "user":
            memory.add_message("user", content)
        elif msg_type == "assistant":
            memory.add_message(
                "assistant", 
                content,
                model=metadata.get("model"),
                provider=metadata.get("provider"),
                response_time=metadata.get("response_time", 0)
            )
        elif msg_type == "tool":
            memory.add_message("tool", content)
    
    return memory

def display_chat_conversation(chat_data: Dict):
    """Display a full chat conversation in a readable format"""
    metadata = chat_data.get("metadata", {})
    messages = chat_data.get("messages", [])
    
    # Chat header
    chat_name = metadata.get("name", f"Chat {chat_data['id']}")
    created_date = datetime.datetime.fromisoformat(chat_data["created_at"]).strftime("%Y-%m-%d %H:%M")
    
    st.markdown(f"""# 💬 {chat_name}
**ID:** `{chat_data['id']}`  
**Created:** {created_date}  
**Models:** {', '.join(metadata.get('models', []))}  
**System Prompt:** {metadata.get('system_prompt', 'Default')[:100]}...

---
""")
    
    # Display messages
    current_user_msg = None
    for i, message in enumerate(messages):
        msg_type = message.get("type")
        content = message.get("content", "")
        timestamp = datetime.datetime.fromisoformat(message["timestamp"]).strftime("%H:%M:%S")
        msg_metadata = message.get("metadata", {})
        
        if msg_type == "user":
            st.markdown(f"**[{timestamp}] You:** {content}")
            
        elif msg_type == "assistant":
            model_name = msg_metadata.get("model_name", msg_metadata.get("model", "Unknown"))
            provider = msg_metadata.get("provider", "")
            response_time = msg_metadata.get("response_time", 0)
            
            author_name = f"🤖 {model_name}"
            if provider:
                author_name += f" ({provider})"
            
            # Truncate very long responses for better readability
            display_content = content
            if len(content) > 1000:
                display_content = content[:1000] + "\n\n*[Response truncated for display]*"
            
            st.markdown(f"**[{timestamp}]** {display_content}")
            if response_time:
                st.caption(f"⏱️ Response Time: {response_time:.2f}s")
            
        elif msg_type == "tool":
            tool_name = msg_metadata.get("tool", "Unknown")
            st.markdown(f"**[{timestamp}] Tool Result ({tool_name}):**")
            st.code(content[:500] + ('...' if len(content) > 500 else ''))

def display_chat_history_sidebar():
    """Display chat history in the sidebar with proper metrics"""
    st.sidebar.header("📚 Chat History")
    st.sidebar.markdown("Compare model performance across different sessions")
    
    chats = chat_history.list_chats(20)
    
    if not chats:
        st.sidebar.info("No chat history yet. Start a conversation to create your first chat history!")
        return
    
    # Add search/filter options
    search_term = st.sidebar.text_input("🔍 Search chats", "")
    model_filter = st.sidebar.multiselect(
        "🤖 Filter by model", 
        options=list(set([m for chat in chats for m in chat_history.get_chat_stats(chat['id']).get('models', [])])),
        default=[]
    )
    
    # Filter chats
    filtered_chats = []
    for chat in chats:
        stats = chat_history.get_chat_stats(chat['id'])
        chat_name = chat.get('metadata', {}).get('name', f"Chat {chat['id']}")
        
        # Apply filters
        if search_term and search_term.lower() not in chat_name.lower():
            continue
            
        if model_filter and not any(m in stats.get('models', []) for m in model_filter):
            continue
            
        filtered_chats.append(chat)
    
    # Display stats about the history
    if filtered_chats:
        all_stats = [chat_history.get_chat_stats(chat['id']) for chat in filtered_chats]
        avg_response_time = sum(s.get('avg_response_time', 0) for s in all_stats) / len(all_stats) if all_stats else 0
        
        st.sidebar.markdown(f"**Found {len(filtered_chats)} chats**")
        st.sidebar.markdown(f"**Avg Response Time:** {avg_response_time:.2f}s")
    
    # Display each chat as an expander
    for i, chat in enumerate(filtered_chats, 1):
        metadata = chat.get("metadata", {})
        stats = chat_history.get_chat_stats(chat["id"])
        
        chat_name = metadata.get("name", f"Chat {chat['id']}")
        created = datetime.datetime.fromisoformat(chat["created_at"]).strftime("%m/%d %H:%M")
        avg_time = stats.get('avg_response_time', 0)
        models = ', '.join(stats.get('models', [])[:2])
        if len(stats.get('models', [])) > 2:
            models += " +more"
        
        # Create an expander for each chat
        with st.sidebar.expander(f"{chat_name} (`{chat['id']}`)", expanded=False):
            st.markdown(f"**🕒 Created:** {created}")
            st.markdown(f"**⏱️ Avg Response:** {avg_time:.2f}s")
            st.markdown(f"**🤖 Models:** {models}")
            st.markdown(f"**💬 Messages:** {stats.get('total_messages', 0)}")
            
            # Add action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("👁️ View", key=f"view_{chat['id']}"):
                    st.session_state.view_chat_id = chat['id']
            with col2:
                if st.button("🔄 Resume", key=f"resume_{chat['id']}"):
                    st.session_state.current_chat_id = chat['id']
                    restored_memory = restore_memory_from_chat(chat)
                    st.session_state.memory = restored_memory
                    st.session_state.settings = {
                        "models": metadata.get("models", [])[0] if metadata.get("models") else "",
                        "system_prompt": metadata.get("system_prompt", SYSTEM_PROMPT_DEFAULT),
                        "persona": metadata.get("persona", ""),
                        "tools": ",".join(metadata.get("tools", []))
                    }
                    st.rerun()
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{chat['id']}", type="secondary"):
                    data = chat_history.load_chats()
                    if chat['id'] in data["chats"]:
                        del data["chats"][chat['id']]
                        chat_history.save_chats(data)
                        st.rerun()

def display_chat_stats(chat_id: str):
    """Display detailed statistics for a chat"""
    if not chat_id:
        return
        
    stats = chat_history.get_chat_stats(chat_id)
    chat_data = chat_history.get_chat(chat_id)
    
    if not chat_data:
        return
        
    metadata = chat_data.get("metadata", {})
    st.subheader("📊 Chat Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Messages", stats.get('total_messages', 0))
    with col2:
        st.metric("Avg Response Time", f"{stats.get('avg_response_time', 0):.2f}s")
    with col3:
        st.metric("Models Used", len(stats.get('models', [])))
    
    # Create a timeline of responses
    messages = [m for m in chat_data.get("messages", []) if m["type"] == "assistant"]
    response_times = []
    timestamps = []
    models = []
    
    for msg in messages:
        metadata = msg.get("metadata", {})
        if "response_time" in metadata:
            response_times.append(metadata["response_time"])
            timestamps.append(msg["timestamp"])
            models.append(metadata.get("model_name", "Unknown"))
    
    if response_times:
        # Create a DataFrame for visualization
        import pandas as pd
        df = pd.DataFrame({
            "Timestamp": timestamps,
            "Response Time (s)": response_times,
            "Model": models
        })
        
        # Convert timestamps to datetime
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        
        # Create a line chart
        st.subheader("⏱️ Response Time Trend")
        fig = px.line(df, x="Timestamp", y="Response Time (s)", color="Model",
                     title="Response Time Over Conversation")
        st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison if multiple models were used
    if len(stats.get('models', [])) > 1:
        st.subheader("🤖 Model Comparison")
        model_data = []
        for model in stats.get('models', []):
            # Get stats for this specific model
            model_msgs = [m for m in messages if m.get("metadata", {}).get("model_name") == model]
            model_times = [m.get("metadata", {}).get("response_time", 0) for m in model_msgs]
            
            if model_times:
                model_data.append({
                    "Model": model,
                    "Avg Response Time": sum(model_times) / len(model_times),
                    "Messages": len(model_msgs)
                })
        
        if model_data:
            import pandas as pd
            model_df = pd.DataFrame(model_data)
            st.dataframe(model_df.set_index("Model"))

# Initialize session state
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationMemory(max_turns=30)
if 'settings' not in st.session_state:
    st.session_state.settings = {
        "models": get_model_names()[0] if get_model_names() else "",
        "system_prompt": SYSTEM_PROMPT_DEFAULT,
        "persona": "",
        "tools": ",".join(list(_TOOL_REGISTRY.keys()))
    }
if 'current_chat_id' not in st.session_state:
    # Create a new chat session
    chat_metadata = {
        "name": generate_chat_name(st.session_state.settings),
        "models": [st.session_state.settings.get("models")] if st.session_state.settings.get("models") else [],
        "persona": st.session_state.settings.get("persona", ""),
        "system_prompt": st.session_state.settings.get("system_prompt", SYSTEM_PROMPT_DEFAULT),
        "tools": [t.strip() for t in str(st.session_state.settings.get("tools", "")).split(",") if t.strip()],
        "providers": [],
        "session_type": "new"
    }
    st.session_state.current_chat_id = chat_history.create_chat(chat_metadata)
if 'view_chat_id' not in st.session_state:
    st.session_state.view_chat_id = None

# Set page config
st.set_page_config(
    page_title="Model Comparison Sandbox",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main app
st.title("🚀 Model Comparison Sandbox")

# Display chat history in sidebar
display_chat_history_sidebar()

# Handle view mode
if st.session_state.view_chat_id:
    chat_data = chat_history.get_chat(st.session_state.view_chat_id)
    if chat_data:
        display_chat_conversation(chat_data)
        display_chat_stats(st.session_state.view_chat_id)
        
        # Add back button
        if st.button("← Back to Chat"):
            st.session_state.view_chat_id = None
            st.rerun()
    else:
        st.error(f"Chat {st.session_state.view_chat_id} not found!")
        st.session_state.view_chat_id = None
        st.rerun()
else:
    # Show welcome message if no messages yet
    if len(st.session_state.memory.messages) == 0:
        st.info("""**Welcome to Model Comparison Sandbox!**

Compare multiple LLM models side-by-side with:
- ⚡ Real-time response time tracking
- 🔧 Tool integration support  
- 📊 Performance analytics
- 💾 Persistent chat history with full conversation viewing
- 🔄 Resume any previous session
- 📋 Compare different chat sessions

Configure your test below and start comparing models!""")
    
    # Settings panel
    with st.expander("⚙️ Configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            selected_model = st.selectbox(
                "🤖 Model to Test", 
                get_model_names(),
                index=get_model_names().index(st.session_state.settings["models"]) if st.session_state.settings["models"] in get_model_names() else 0
            )
            
            system_prompt = st.text_area(
                "📝 System Prompt", 
                value=st.session_state.settings["system_prompt"],
                height=100
            )
            
        with col2:
            persona = st.selectbox(
                "🎭 Persona (optional)", 
                [""] + persona_values,
                index=0 if not st.session_state.settings["persona"] else persona_values.index(st.session_state.settings["persona"]) + 1
            )
            
            tools = st.multiselect(
                "🔧 Available Tools", 
                list(_TOOL_REGISTRY.keys()),
                default=[t.strip() for t in str(st.session_state.settings["tools"]).split(",") if t.strip()]
            )
        
        # Update settings when changed
        if st.button("Apply Settings"):
            st.session_state.settings = {
                "models": selected_model,
                "system_prompt": system_prompt,
                "persona": persona if persona else "",
                "tools": ",".join(tools)
            }
            
            # Update chat metadata
            chat_history.update_chat_metadata(
                st.session_state.current_chat_id,
                {
                    "name": generate_chat_name(st.session_state.settings),
                    "models": [selected_model],
                    "persona": persona if persona else "",
                    "system_prompt": system_prompt,
                    "tools": tools,
                }
            )
            st.rerun()


    # ADD THE NEW CHAT BUTTON HERE
    st.markdown("---")  # Divider line for visual separation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 New Chat", help="Start a fresh conversation with current settings"):
            # Create new chat with current settings
            chat_metadata = {
                "name": generate_chat_name(st.session_state.settings),
                "models": [st.session_state.settings.get("models")] if st.session_state.settings.get("models") else [],
                "persona": st.session_state.settings.get("persona", ""),
                "system_prompt": st.session_state.settings.get("system_prompt", SYSTEM_PROMPT_DEFAULT),
                "tools": [t.strip() for t in str(st.session_state.settings.get("tools", "")).split(",") if t.strip()],
                "providers": [],
                "session_type": "new"
            }
            
            # Create new chat and update session state
            st.session_state.current_chat_id = chat_history.create_chat(chat_metadata)
            st.session_state.memory = ConversationMemory(max_turns=30)
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.memory.messages:
        role = message["role"]
        content = message["content"]
        metadata = message["metadata"]
        
        with st.chat_message("user" if role == "user" else "assistant"):
            st.write(content)
            if role == "assistant" and "response_time" in metadata:
                st.caption(f"⏱️ Response Time: {metadata['response_time']:.2f}s | Model: {metadata.get('model_name', 'Unknown')}")

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.memory.add_message("user", prompt)
        chat_history.add_message(st.session_state.current_chat_id, "user", prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get settings
        settings = st.session_state.settings
        selected_model = settings.get("models")
        system_prompt = settings.get("system_prompt", SYSTEM_PROMPT_DEFAULT)
        persona = settings.get("persona", "")
        available_tools = [t.strip() for t in str(settings.get("tools", "")).split(",") if t.strip()]
        
        if persona:
            system_prompt += f"\n\nPersona: {persona}"
        
        # Find model config
        model_config = None
        for model in CFG.get("models", []):
            if model.get("name") == selected_model:
                model_config = model
                break
        
        if not model_config:
            with st.chat_message("assistant"):
                st.error(f"Model configuration not found for: {selected_model}")
            st.stop()
        
        provider = model_config["provider"]
        model_id = model_config["id"]
        model_name = model_config["name"]
        
        # Get conversation history
        history = st.session_state.memory.get_context()
        
        # Call model with timing
        start_time = time.time()
        try:
            # This is a placeholder - replace with your actual model calling function
            response = call_model(provider, model_id, system_prompt, history)
            # In a real app, you would use: response = call_model(provider, model_id, system_prompt, history)
        except Exception as e:
            response = f"Error calling {model_name}: {str(e)}"
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Add to memory
        st.session_state.memory.add_message(
            "assistant", 
            response, 
            model=model_id, 
            provider=provider, 
            response_time=response_time,
            model_name=model_name
        )
        
        # Save to chat history
        chat_history.add_message(
            st.session_state.current_chat_id, 
            "assistant", 
            response,
            {
                "model": model_id,
                "provider": provider,
                "response_time": response_time,
                "model_name": model_name
            }
        )
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)
            st.caption(f"⏱️ Response Time: {response_time:.2f}s | Model: {model_name}")