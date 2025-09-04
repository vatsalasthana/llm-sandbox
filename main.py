import os, json, yaml, re, datetime, time, sqlite3, hashlib
import chainlit as cl
from chainlit.input_widget import Select, TextInput
from dotenv import load_dotenv
from model_router import call_model
from typing import Optional, Dict, List, Any
import uuid

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
            if "response_time" in metadata:
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

# Helper Functions
async def display_chat_conversation(chat_data: Dict):
    """Display a full chat conversation in a readable format"""
    metadata = chat_data.get("metadata", {})
    messages = chat_data.get("messages", [])
    
    # Chat header
    chat_name = metadata.get("name", f"Chat {chat_data['id']}")
    created_date = datetime.datetime.fromisoformat(chat_data["created_at"]).strftime("%Y-%m-%d %H:%M")
    
    header = f"""# 💬 {chat_name}
**ID:** `{chat_data['id']}`  
**Created:** {created_date}  
**Models:** {', '.join(metadata.get('models', []))}  
**System Prompt:** {metadata.get('system_prompt', 'Default')[:100]}...

---
"""
    
    await cl.Message(content=header, author="📊 Chat History").send()
    
    # Display messages
    current_user_msg = None
    for i, message in enumerate(messages):
        msg_type = message.get("type")
        content = message.get("content", "")
        timestamp = datetime.datetime.fromisoformat(message["timestamp"]).strftime("%H:%M:%S")
        msg_metadata = message.get("metadata", {})
        
        if msg_type == "user":
            current_user_msg = content
            await cl.Message(
                content=f"**[{timestamp}] You:** {content}",
                author="👤 User"
            ).send()
            
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
            
            msg_content = f"**[{timestamp}]** {display_content}"
            if response_time:
                msg_content += f"\n\n*⏱️ Response Time: {response_time:.2f}s*"
            
            await cl.Message(
                content=msg_content,
                author=author_name
            ).send()
            
        elif msg_type == "tool":
            tool_name = msg_metadata.get("tool", "Unknown")
            await cl.Message(
                content=f"**[{timestamp}] Tool Result ({tool_name}):**\n```\n{content[:500]}{'...' if len(content) > 500 else ''}\n```",
                author="🔧 Tool"
            ).send()

async def show_chat_history_interactive():
    """Show interactive chat history with options to view/resume chats"""
    chats = chat_history.list_chats(10)
    
    if not chats:
        await cl.Message(content="📚 **No chat history yet.** Start a conversation to create your first chat history!").send()
        return
    
    # Create summary with clickable options
    history_text = """📚 **Recent Chat History**

Choose an action:
- Type `/view <chat_id>` to see the full conversation
- Type `/resume <chat_id>` to continue that chat
- Type `/compare <chat_id1> <chat_id2>` to compare two chats
- Type `/delete <chat_id>` to delete a chat

"""
    
    for i, chat in enumerate(chats, 1):
        metadata = chat.get("metadata", {})
        stats = chat_history.get_chat_stats(chat["id"])
        
        chat_name = metadata.get("name", f"Chat {chat['id']}")
        created = datetime.datetime.fromisoformat(chat["created_at"]).strftime("%m/%d %H:%M")
        models_used = stats.get("models", [])
        providers_used = stats.get("providers", [])
        
        # Create a nice summary card for each chat
        card = f"""
**{i}. {chat_name}** 
- **ID:** `{chat['id']}`
- **Created:** {created}
- **Messages:** {stats.get('total_messages', 0)} 
- **Models:** {', '.join(models_used[:3])}
- **Providers:** {', '.join(providers_used)}
- **Avg Response:** {stats.get('avg_response_time', 0):.2f}s

"""
        history_text += card
    
    await cl.Message(content=history_text).send()

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

async def compare_chats(chat_id1: str, chat_id2: str):
    """Compare two chats side by side"""
    chat1 = chat_history.get_chat(chat_id1)
    chat2 = chat_history.get_chat(chat_id2)
    
    if not chat1 or not chat2:
        missing = []
        if not chat1:
            missing.append(chat_id1)
        if not chat2:
            missing.append(chat_id2)
        await cl.Message(content=f"❌ Chat(s) not found: {', '.join(missing)}").send()
        return
    
    stats1 = chat_history.get_chat_stats(chat_id1)
    stats2 = chat_history.get_chat_stats(chat_id2)
    
    # Create comparison header
    comparison = f"""# 🔄 Chat Comparison

## 📊 Chat 1: {chat1['metadata'].get('name', 'Unnamed')} (`{chat_id1}`)
- **Models:** {', '.join(stats1.get('models', []))}
- **Messages:** {stats1.get('total_messages', 0)}
- **Avg Response Time:** {stats1.get('avg_response_time', 0):.2f}s
- **Created:** {datetime.datetime.fromisoformat(chat1['created_at']).strftime('%Y-%m-%d %H:%M')}

## 📊 Chat 2: {chat2['metadata'].get('name', 'Unnamed')} (`{chat_id2}`)
- **Models:** {', '.join(stats2.get('models', []))}
- **Messages:** {stats2.get('total_messages', 0)}
- **Avg Response Time:** {stats2.get('avg_response_time', 0):.2f}s
- **Created:** {datetime.datetime.fromisoformat(chat2['created_at']).strftime('%Y-%m-%d %H:%M')}

---

## 🎯 Performance Comparison
- **Faster Average Response:** {"Chat 1" if stats1.get('avg_response_time', 0) < stats2.get('avg_response_time', 0) else "Chat 2"}
- **More Messages:** {"Chat 1" if stats1.get('total_messages', 0) > stats2.get('total_messages', 0) else "Chat 2"}
- **Unique Models Used:** {set(stats1.get('models', [])).symmetric_difference(set(stats2.get('models', [])))}

Type `/view {chat_id1}` or `/view {chat_id2}` to see full conversations.
"""
    
    await cl.Message(content=comparison, author="🔄 Comparison").send()

async def handle_commands(message_content: str, settings: Dict, memory: ConversationMemory) -> bool:
    """Handle special commands - returns True if command was handled"""
    if not message_content.startswith('/'):
        return False
    
    parts = message_content.split()
    command = parts[0].lower()
    
    if command == '/help':
        help_text = """🔧 **Available Commands:**

**Basic Commands:**
- `/help` - Show this help message
- `/new` - Start a new chat session
- `/clear` - Clear current conversation memory

**History Commands:**
- `/history` - Show recent chat sessions (interactive)
- `/view <chat_id>` - View full conversation of a specific chat
- `/resume <chat_id>` - Resume a previous chat session
- `/compare <chat_id1> <chat_id2>` - Compare two chat sessions
- `/stats [chat_id]` - Show detailed stats for current or specific chat
- `/delete <chat_id>` - Delete a specific chat

**Examples:**
- `/view abc123` - View chat with ID abc123
- `/compare abc123 def456` - Compare two chats
- `/resume abc123` - Continue chat abc123
"""
        await cl.Message(content=help_text).send()
        
    elif command == '/history':
        await show_chat_history_interactive()
        
    elif command == '/view':
        if len(parts) < 2:
            await cl.Message(content="❌ Please provide a chat ID. Usage: `/view <chat_id>`").send()
        else:
            chat_id = parts[1]
            chat_data = chat_history.get_chat(chat_id)
            if not chat_data:
                await cl.Message(content=f"❌ Chat `{chat_id}` not found!").send()
            else:
                await display_chat_conversation(chat_data)
    
    elif command == '/compare':
        if len(parts) < 3:
            await cl.Message(content="❌ Please provide two chat IDs. Usage: `/compare <chat_id1> <chat_id2>`").send()
        else:
            await compare_chats(parts[1], parts[2])
    
    elif command == '/resume':
        if len(parts) < 2:
            await cl.Message(content="❌ Please provide a chat ID. Usage: `/resume <chat_id>`").send()
        else:
            chat_id = parts[1]
            chat_data = chat_history.get_chat(chat_id)
            if not chat_data:
                await cl.Message(content=f"❌ Chat `{chat_id}` not found!").send()
            else:
                # Restore memory and session
                restored_memory = restore_memory_from_chat(chat_data)
                cl.user_session.set("memory", restored_memory)
                cl.user_session.set("chat_id", chat_id)
                
                # Restore settings
                metadata = chat_data.get("metadata", {})
                restored_settings = {
                    "models": metadata.get("models", [])[0] if metadata.get("models") else "",
                    "system_prompt": metadata.get("system_prompt", SYSTEM_PROMPT_DEFAULT),
                    "persona": metadata.get("persona", ""),
                    "tools": ",".join(metadata.get("tools", []))
                }
                cl.user_session.set("settings", restored_settings)
                
                stats = chat_history.get_chat_stats(chat_id)
                resume_msg = f"""🔄 **Chat Resumed: {metadata.get('name', 'Unnamed Chat')}**

📊 **Session Stats:**
- 💬 Messages: {stats.get('total_messages', 0)}
- 🤖 Models: {', '.join(stats.get('models', []))}
- 🌐 Providers: {', '.join(stats.get('providers', []))}
- ⏱️ Avg Response Time: {stats.get('avg_response_time', 0):.2f}s
- 📅 Created: {datetime.datetime.fromisoformat(chat_data['created_at']).strftime('%Y-%m-%d %H:%M')}

Continue your conversation!"""
                await cl.Message(content=resume_msg).send()
    
    elif command == '/delete':
        if len(parts) < 2:
            await cl.Message(content="❌ Please provide a chat ID. Usage: `/delete <chat_id>`").send()
        else:
            chat_id = parts[1]
            data = chat_history.load_chats()
            if chat_id in data["chats"]:
                chat_name = data["chats"][chat_id]["metadata"].get("name", f"Chat {chat_id}")
                del data["chats"][chat_id]
                chat_history.save_chats(data)
                await cl.Message(content=f"🗑️ **Deleted chat:** {chat_name} (`{chat_id}`)").send()
            else:
                await cl.Message(content=f"❌ Chat `{chat_id}` not found!").send()
    
    elif command == '/stats':
        chat_id = parts[1] if len(parts) > 1 else cl.user_session.get("chat_id")
        if not chat_id:
            await cl.Message(content="❌ No active chat session. Provide a chat ID or start a new session.").send()
        else:
            stats = chat_history.get_chat_stats(chat_id)
            chat_data = chat_history.get_chat(chat_id)
            if not chat_data:
                await cl.Message(content=f"❌ Chat `{chat_id}` not found!").send()
            else:
                metadata = chat_data.get("metadata", {})
                stats_msg = f"""📊 **Chat Statistics for `{chat_id}`**

📝 **Chat:** {metadata.get('name', 'Unnamed')}
📅 **Created:** {datetime.datetime.fromisoformat(chat_data['created_at']).strftime('%Y-%m-%d %H:%M')}
💬 **Total Messages:** {stats.get('total_messages', 0)}
👤 **User Messages:** {stats.get('user_messages', 0)}
🤖 **Assistant Messages:** {stats.get('assistant_messages', 0)}
🔧 **Tool Calls:** {stats.get('tool_calls', 0)}
⏱️ **Avg Response Time:** {stats.get('avg_response_time', 0):.2f}s
🤖 **Models Used:** {', '.join(stats.get('models', []))}
🌐 **Providers Used:** {', '.join(stats.get('providers', []))}"""
                await cl.Message(content=stats_msg).send()
    
    elif command == '/clear':
        memory.clear()
        await cl.Message(content="🧹 **Conversation memory cleared!** Previous messages removed from context.").send()
    
    elif command == '/new':
        # Create new chat session
        chat_metadata = {
            "name": generate_chat_name(settings),
            "models": [settings.get("models")] if settings.get("models") else [],
            "persona": settings.get("persona", ""),
            "system_prompt": settings.get("system_prompt", SYSTEM_PROMPT_DEFAULT),
            "tools": [t.strip() for t in str(settings.get("tools", "")).split(",") if t.strip()],
            "providers": [],
            "session_type": "new"
        }
        
        new_chat_id = chat_history.create_chat(chat_metadata)
        cl.user_session.set("chat_id", new_chat_id)
        
        # Clear memory
        memory.clear()
        
        models_str = settings.get("models", "None")
        await cl.Message(content=f"✅ **New session created!** \n\nChat ID: `{new_chat_id}`\nModel: {models_str}\n\nStart sending messages to test the model!").send()
    
    else:
        await cl.Message(content=f"❌ Unknown command: `{command}`. Type `/help` for available commands.").send()
    
    return True

# Chainlit handlers
@cl.on_chat_start
async def start():
    welcome_msg = """🚀 **Model Comparison Sandbox**

Compare multiple LLM models side-by-side with:
- ⚡ Real-time response time tracking
- 🔧 Tool integration support  
- 📊 Performance analytics
- 💾 Persistent chat history with full conversation viewing
- 🔄 Resume any previous session
- 📋 Compare different chat sessions

**Quick Commands:**
- `/help` - Show all commands
- `/history` - Interactive chat history browser
- `/view <id>` - View full conversation
- `/compare <id1> <id2>` - Compare two chats

Configure your test below and start comparing models!"""
    
    await cl.Message(content=welcome_msg).send()
    
    # Show recent chat history automatically
    await show_chat_history_interactive()
    
    available_models = get_model_names()
    init_models = available_models[0] if available_models else "GPT-4"
    
    controls = [
        Select(
            id="models", 
            label="🤖 Model to Test", 
            values=available_models, 
            initial=init_models,
            tooltip="Select model to test"
        ),
        TextInput(
            id="system_prompt", 
            label="📝 System Prompt", 
            initial=SYSTEM_PROMPT_DEFAULT,
            tooltip="Instructions that guide the model's behavior"
        ),
        Select(
            id="persona", 
            label="🎭 Persona (optional)", 
            values=persona_values,
            tooltip="Apply a specific persona to the model"
        ) if persona_values else TextInput(
            id="persona", 
            label="🎭 Persona (optional)",
            tooltip="Describe the persona for the model"
        ),
        TextInput(
            id="tools", 
            label="🔧 Available Tools", 
            initial=",".join(list(_TOOL_REGISTRY.keys())),
            tooltip="Comma-separated list of tools model can use"
        )
    ]

    settings = await cl.ChatSettings(controls).send()
    if not settings:
        settings = {
            "models": init_models,
            "system_prompt": SYSTEM_PROMPT_DEFAULT,
            "persona": "",
            "tools": ",".join(list(_TOOL_REGISTRY.keys()))
        }
    cl.user_session.set("settings", settings)

    # Initialize memory
    memory = ConversationMemory(max_turns=30)
    cl.user_session.set("memory", memory)

    # Create new chat with metadata
    chat_metadata = {
        "name": generate_chat_name(settings),
        "models": [settings.get("models")] if settings.get("models") else [],
        "persona": settings.get("persona", ""),
        "system_prompt": settings.get("system_prompt", SYSTEM_PROMPT_DEFAULT),
        "tools": [t.strip() for t in str(settings.get("tools", "")).split(",") if t.strip()],
        "providers": [],
        "session_type": "new"
    }
    
    chat_id = chat_history.create_chat(chat_metadata)
    cl.user_session.set("chat_id", chat_id)
    
    models_str = settings.get("models", "None")
    await cl.Message(
        content=f"✅ **New session created!** \n\nChat ID: `{chat_id}`\nModel: {models_str}\n\nStart sending messages to test the model!"
    ).send()

@cl.on_settings_update
async def update_settings(settings):
    """Handle settings updates"""
    cl.user_session.set("settings", settings)
    chat_id = cl.user_session.get("chat_id")
    
    if chat_id:
        # Update chat metadata with new settings
        updated_metadata = {
            "name": generate_chat_name(settings),
            "models": [settings.get("models")] if settings.get("models") else [],
            "persona": settings.get("persona", ""),
            "system_prompt": settings.get("system_prompt", SYSTEM_PROMPT_DEFAULT),
            "tools": [t.strip() for t in str(settings.get("tools", "")).split(",") if t.strip()]
        }
        chat_history.update_chat_metadata(chat_id, updated_metadata)
    
    await cl.Message(content="⚙️ **Settings updated!** New messages will use the updated configuration.").send()

@cl.on_message
async def main(message: cl.Message):
    settings = cl.user_session.get("settings") or {}
    memory: ConversationMemory = cl.user_session.get("memory")
    chat_id = cl.user_session.get("chat_id")

    # Handle commands first
    command_handled = await handle_commands(message.content, settings, memory)
    if command_handled:
        return

    selected_model = settings.get("models")
    if not selected_model:
        await cl.Message(content="⚠️ Please select a model in the settings!").send()
        return

    system_prompt = settings.get("system_prompt", SYSTEM_PROMPT_DEFAULT)
    persona = settings.get("persona", "")
    available_tools = [t.strip() for t in str(settings.get("tools", "")).split(",") if t.strip()]

    if persona:
        system_prompt += f"\n\nPersona: {persona}"

    user_input = message.content
    memory.add_message("user", user_input)
    
    # Save user message to chat history
    if chat_id:
        chat_history.add_message(chat_id, "user", user_input)

    # Find model config
    model_config = None
    for model in CFG.get("models", []):
        if model.get("name") == selected_model:
            model_config = model
            break
    
    if not model_config:
        await cl.Message(content=f"❌ Model configuration not found for: {selected_model}").send()
        return
        
    provider = model_config["provider"]
    model_id = model_config["id"]
    model_name = model_config["name"]

    # Get conversation history
    history = memory.get_context()

    # Call model with timing
    start_time = time.time()
    try:
        response = call_model(provider, model_id, system_prompt, history)
        if not response:
            response = "No response received from model."
    except Exception as e:
        response = f"❌ Error calling {model_name}: {str(e)}"
    
    end_time = time.time()
    response_time = end_time - start_time

    # Add to memory
    memory.add_message("assistant", response, model=model_id, provider=provider, response_time=response_time)
    
    # Save to chat history
    if chat_id:
        chat_history.add_message(
            chat_id, 
            "assistant", 
            response,
            {
                "model": model_id,
                "provider": provider,
                "response_time": response_time,
                "model_name": model_name
            }
        )

    # Handle tool calls
    tool_used = False
    if response and available_tools:
        tool_obj = parse_tool_json(response)
        if isinstance(tool_obj, dict) and "tool" in tool_obj:
            tool_name = tool_obj.get("tool")
            tool_input = tool_obj.get("input", "")
            
            if tool_name in available_tools:
                try:
                    tool_result = call_tool(tool_name, tool_input)
                    tool_used = True
                except Exception as e:
                    tool_result = f"Tool error: {str(e)}"
            else:
                tool_result = f"Tool '{tool_name}' not available. Available: {', '.join(available_tools)}"
            
            # Add tool result to memory
            tool_message = f"Result from {tool_name}:\n{tool_result}"
            memory.add_message("tool", tool_message)
            
            # Save tool result
            if chat_id:
                chat_history.add_message(chat_id, "tool", tool_message, {"tool": tool_name, "input": tool_input})

            # Get final response after tool
            history_with_tool = memory.get_context()
            start_time2 = time.time()
            try:
                final_response = call_model(provider, model_id, system_prompt, history_with_tool)
            except Exception as e:
                final_response = f"Error in follow-up: {str(e)}"
            end_time2 = time.time()
            followup_time = end_time2 - start_time2
            
            total_time = response_time + followup_time
            memory.add_message("assistant", final_response, model=model_id, provider=provider, response_time=followup_time)
            
            # Save final response
            if chat_id:
                chat_history.add_message(
                    chat_id, 
                    "assistant", 
                    final_response,
                    {
                        "model": model_id,
                        "provider": provider,
                        "response_time": followup_time,
                        "model_name": model_name,
                        "after_tool": True
                    }
                )

            # Display tool-enhanced response
            await cl.Message(
                author=f"🔧 {model_name} ({provider})",
                content=f"""**Initial Response:**
{response}

**🔧 Tool Used:** `{tool_name}`
**📥 Tool Input:** `{tool_input}`
**📤 Tool Result:**`{tool_result}`

**🎯 Final Response:**
{final_response}

*⏱️ Total Time: {total_time:.2f}s (Initial: {response_time:.2f}s + Follow-up: {followup_time:.2f}s)*"""
            ).send()
        else:
            # Regular response without tools
            await cl.Message(
                author=f"🤖 {model_name} ({provider})",
                content=f"{response}\n\n*⏱️ Response Time: {response_time:.2f}s*"
            ).send()
    else:
        # Regular response without tools
        await cl.Message(
            author=f"🤖 {model_name} ({provider})",
            content=f"{response}\n\n*⏱️ Response Time: {response_time:.2f}s*"
        ).send()

    # Update chat metadata with provider used
    if chat_id:
        updated_metadata = {"providers": [provider]}
        chat_history.update_chat_metadata(chat_id, updated_metadata)