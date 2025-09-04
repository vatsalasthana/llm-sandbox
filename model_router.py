import os
import logging
from typing import List, Dict, Any, Optional
from tools import RAG
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def call_groq(model_id: str, system_prompt: str, history: List[Dict[str, str]]) -> str:
    """Call Groq API using the official client."""
    try:
        from groq import Groq
        
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Prepare messages with system prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return f"Error calling Groq: {str(e)}"

def call_ollama(model_id: str, system_prompt: str, history: List[Dict[str, str]]) -> str:
    """Call Ollama API using the official client."""
    try:
        import ollama
        
        # Prepare messages with system prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        response = ollama.chat(
            model=model_id,
            messages=messages,
            options={
                'temperature': 0.7,
                'top_p': 0.9,
            }
        )
        
        return response['message']['content']
        
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        return f"Error calling Ollama: {str(e)}"

def call_openai(model_id: str, system_prompt: str, history: List[Dict[str, str]]) -> str:
    """Call OpenAI API using the modern client."""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Prepare messages with system prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            # temperature=0.7,
            # max_tokens=1024
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"Error calling OpenAI: {str(e)}"

def call_deepseek(model_id: str, system_prompt: str, history: List[Dict[str, str]]) -> str:
    """
    Call DeepSeek model via OpenRouter API.

    Args:
        model_id (str): The DeepSeek model to use, e.g., "deepseek-chat" or "deepseek-coder".
        system_prompt (str): The system-level instruction.
        history (List[Dict[str, str]]): Conversation history with roles ["system", "user", "assistant"].

    Returns:
        str: Model response text.
    """
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment")

        url = "https://openrouter.ai/api/v1/chat/completions"

        # Construct messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model_id,
            "messages": messages,
            # "temperature": 0.7,
            # "max_tokens": 1024,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        logger.error(f"DeepSeek (OpenRouter) API error: {e}")
        return f"Error calling DeepSeek: {str(e)}"

def call_anthropic(model_id: str, system_prompt: str, history: List[Dict[str, str]]) -> str:
    """Call Anthropic API using the official client."""
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Anthropic requires separate system parameter
        # Filter out system messages from history for Anthropic
        user_messages = [msg for msg in history if msg["role"] != "system"]
        
        response = client.messages.create(
            model=model_id,
            system=system_prompt if system_prompt else "You are a helpful assistant.",
            messages=user_messages,
            max_tokens=1024,
            temperature=0.7
        )
        
        return response.content[0].text
        
    except Exception as e:
        logger.error(f"Anthropic API error: {e}")
        return f"Error calling Anthropic: {str(e)}"

def call_huggingface(model_id: str, system_prompt: str, history: List[Dict[str, str]]) -> str:
    """Call Hugging Face models using modern transformers or Inference API."""
    try:
        # Option 1: Using Hugging Face Inference API (recommended for production)
        if os.getenv("HUGGINGFACE_API_KEY"):
            import requests
            
            api_url = f"https://api-inference.huggingface.co/models/{model_id}"
            headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
            
            # Format prompt
            prompt = ""
            if system_prompt:
                prompt += f"System: {system_prompt}\n\n"
            
            for msg in history:
                role = msg["role"].capitalize()
                prompt += f"{role}: {msg['content']}\n"
            
            prompt += "Assistant:"
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 512,
                    "temperature": 0.7,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "No response generated")
            else:
                return str(result)
        
        # Option 2: Local transformers (fallback)
        else:
            from transformers import pipeline, AutoTokenizer
            
            # Use a more efficient approach with proper tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            pipe = pipeline(
                "text-generation",
                model=model_id,
                tokenizer=tokenizer,
                device_map="auto",
                torch_dtype="auto"
            )
            
            # Format messages for chat template if available
            if hasattr(tokenizer, 'apply_chat_template'):
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.extend(history)
                
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                # Fallback formatting
                formatted_prompt = ""
                if system_prompt:
                    formatted_prompt += f"System: {system_prompt}\n\n"
                
                for msg in history:
                    role = msg["role"].capitalize()
                    formatted_prompt += f"{role}: {msg['content']}\n"
                formatted_prompt += "Assistant:"
            
            outputs = pipe(
                formatted_prompt,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = outputs[0]['generated_text']
            # Extract only the new generated part
            if generated_text.startswith(formatted_prompt):
                return generated_text[len(formatted_prompt):].strip()
            else:
                return generated_text
        
    except Exception as e:
        logger.error(f"Hugging Face API error: {e}")
        return f"Error calling Hugging Face: {str(e)}"

def call_model(provider: str, model_id: str, system_prompt: str, history: List[Dict[str, str]]) -> str:
    """
    Universal function to call different AI model providers.
    
    Args:
        provider: The AI provider ('groq', 'ollama', 'openai', 'anthropic', 'huggingface')
        model_id: The specific model ID/name
        system_prompt: System prompt for the model
        history: Conversation history as list of {"role": str, "content": str} dicts
    
    Returns:
        Generated response string
    """
    provider = provider.lower()
    
    # Validate inputs
    if not isinstance(history, list) or not history:
        raise ValueError("History must be a non-empty list")
    
    for msg in history:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise ValueError("Each history item must be a dict with 'role' and 'content' keys")
    
    try:
        if provider == "groq":
            return call_groq(model_id, system_prompt, history)
        elif provider == "ollama":
            return call_ollama(model_id, system_prompt, history)
        elif provider == "openai":
            return call_openai(model_id, system_prompt, history)
        elif provider == "anthropic":
            return call_anthropic(model_id, system_prompt, history)
        elif provider == "huggingface":
            return call_huggingface(model_id, system_prompt, history)
        elif provider == "openrouter":
            return call_deepseek(model_id, system_prompt, history)
        else:
            raise ValueError(f"Unknown provider: {provider}. Supported providers: groq, ollama, openai, anthropic, huggingface")
    
    except Exception as e:
        logger.error(f"Error calling {provider} with model {model_id}: {e}")
        raise