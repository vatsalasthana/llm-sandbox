import os, json, yaml, re, datetime
import chainlit as cl
from chainlit.input_widget import Select, TextInput
from dotenv import load_dotenv
from model_router import call_model
from tools import _TOOL_REGISTRY, call_tool, RAG
from conversation_memory import ConversationMemory

load_dotenv()
with open("config.yaml","r") as f:
    CFG = yaml.safe_load(f)

SYSTEM_PROMPT_DEFAULT = CFG.get("system_prompt","You are a helpful assistant.")
personas = CFG.get("personas", [])
persona_values = [p["name"] for p in personas if "name" in p]


def get_model_names():
    return [m["name"] for m in CFG.get("models", [])]

def parse_tool_json(text: str):
    line = text.strip()
    try:
        obj = json.loads(line)
        return obj
    except Exception:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None

@cl.on_chat_start
async def start():
    await cl.Message(content="👋 Pick models, choose persona, and start testing!").send()
    init_models = get_model_names()[:1] if get_model_names() else []
    controls = [
        Select(id="models", label="Model to Compare", values=get_model_names(), initial=init_models[0] if init_models else None),
        TextInput(id="system_prompt", label="System Prompt", initial=SYSTEM_PROMPT_DEFAULT),
        Select(id="persona", label="Persona (optional)", values=persona_values) if persona_values else cl.TextInput(id="persona", label="Persona (optional)"),
        TextInput(id="tools", label="Enable Tools (comma separated)", initial=",".join(list(_TOOL_REGISTRY.keys())))
    ]

    settings = await cl.ChatSettings(controls).send()
    cl.user_session.set("settings", settings)

    # initialize conversation memory for this session
    memory = ConversationMemory(max_turns=30)
    cl.user_session.set("memory", memory)


@cl.on_message
async def main(message: cl.Message):
    settings = cl.user_session.get("settings") or {}
    memory: ConversationMemory = cl.user_session.get("memory")  # retrieve memory

    selected_models = settings.get("models", [])
    system_prompt = settings.get("system_prompt", SYSTEM_PROMPT_DEFAULT)
    persona = settings.get("persona", "") or ""
    enable_tools = settings.get("enable_tools", True)
    selected_tools = settings.get("tools", [])

    if persona:
        system_prompt += f"\n\nPersona: {persona}"

    user_input = message.content
    memory.add_message("user", user_input)   # add user message to memory

    for m in CFG.get("models", []):
        if m["name"] not in selected_models:
            continue
        provider = m["provider"]
        model_id = m["id"]

        # use full conversation history
        history = memory.get_context()

        try:
            out1 = call_model(provider, model_id, system_prompt, history)
            memory.add_message("assistant", out1)  # add assistant reply to memory
        except Exception as e:
            await cl.Message(author=m["name"], content=f"⚠️ Error: {e}").send()
            continue

        if enable_tools and out1:
            obj = parse_tool_json(out1)
            if isinstance(obj, dict) and "tool" in obj:
                tool = obj.get("tool")
                tool_input = obj.get("input","")
                tool_result = ""
                if tool in selected_tools:
                    tool_result = call_tool(tool, tool_input)
                else:
                    tool_result = f"Tool '{tool}' is disabled."

                # add tool call result into memory
                memory.add_message("tool", f"Result from {tool}:\n{tool_result}")

                history2 = memory.get_context()
                try:
                    final = call_model(provider, model_id, system_prompt, history2)
                    memory.add_message("assistant", final)
                except Exception as e:
                    final = f"⚠️ Error after tool call: {e}"
                await cl.Message(author=m["name"], content=f"**Tool call:** `{tool}` with input `{tool_input}`\n\n**Tool result:**\n{tool_result}\n\n**Final answer:**\n{final}").send()
                continue

            elif isinstance(obj, dict) and "final" in obj:
                memory.add_message("assistant", obj.get("final",""))
                await cl.Message(author=m["name"], content=obj.get("final","")).send()
                continue

        await cl.Message(author=m["name"], content=out1 or "").send()


    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("logs", exist_ok=True)
    with open(os.path.join("logs", f"chat-{ts}.jsonl"), "a", encoding="utf-8") as f:
        rec = {"time": ts, "input": user_input, "models": selected_models}
        f.write(json.dumps(rec) + "\n")
