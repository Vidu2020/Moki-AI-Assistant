import streamlit as st
import speech_recognition as sr
import pyttsx3
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama
import os
import sys
import openai
from typing import Optional, Literal

# ✅ Fix imports (for Streamlit)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.memory import setup_qdrant, store_memory, extract_pdf_text, store_pdf_content
from app.rag import build_prompt


# ============================================
# ✅ API CONFIGURATION MANAGEMENT
# ============================================

class APIManager:
    """Manages API configuration and LLM interactions"""

    # Supported API providers
    OPEN_PROVIDERS = ["OpenAI", "Groq", "Anthropic", "Moonshot AI", "Custom"]

    def __init__(self):
        self.init_session_state()

    @staticmethod
    def init_session_state():
        """Initialize session state variables for API management"""
        defaults = {
            "api_type": None,  # "open" or "closed"
            "api_provider": None,
            "api_key": None,
            "api_configured": False,
            "pdf_loaded": False,
            "messages": [],
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def set_api_config(api_type: str, provider: Optional[str] = None, api_key: Optional[str] = None):
        """Store API configuration in session state securely"""
        st.session_state.api_type = api_type
        st.session_state.api_provider = provider
        st.session_state.api_key = api_key
        st.session_state.api_configured = True

    @staticmethod
    def clear_api_config():
        """Clear API configuration from session state"""
        st.session_state.api_type = None
        st.session_state.api_provider = None
        st.session_state.api_key = None
        st.session_state.api_configured = False

    @staticmethod
    def get_api_key() -> Optional[str]:
        """Get API key from session state or environment variable"""
        # Priority: session_state > environment variable
        if st.session_state.get("api_key"):
            return st.session_state.api_key

        provider = st.session_state.get("api_provider", "").upper().replace(" ", "_")
        env_var_name = f"{provider}_API_KEY"
        return os.getenv(env_var_name) or os.getenv("OPENAI_API_KEY")

    @staticmethod
    def is_configured() -> bool:
        """Check if API is configured"""
        return st.session_state.get("api_configured", False)


# ============================================
# ✅ SIDEBAR: API SELECTION
# ============================================

def render_api_sidebar():
    """Render API selection sidebar"""
    with st.sidebar:
        st.title("⚙️ API Configuration")
        st.markdown("---")

        # API Type Selection
        api_type = st.selectbox(
            "Select API Type",
            options=["Choose...", "Open API", "Closed API"],
            index=0,
            help="Choose between external APIs or internal model"
        )

        if api_type == "Choose...":
            st.info("👆 Please select an API type to continue")
            return False

        elif api_type == "Open API":
            return render_open_api_section()

        elif api_type == "Closed API":
            return render_closed_api_section()

    return False


def render_open_api_section() -> bool:
    """Render Open API configuration section"""
    st.subheader("🔓 Open API Configuration")

    # API Provider Selection
    provider = st.selectbox(
        "API Provider",
        options=["Select Provider..."] + APIManager.OPEN_PROVIDERS,
        index=0,
        help="Select your LLM provider"
    )

    if provider == "Select Provider...":
        st.warning("Please select a provider")
        return False

    # Show provider-specific info
    provider_info = {
        "OpenAI": "https://platform.openai.com/api-keys",
        "Groq": "https://console.groq.com/keys",
        "Anthropic": "https://console.anthropic.com/settings/keys",
        "Moonshot AI": "https://platform.moonshot.cn",
        "Custom": "Enter your custom API endpoint"
    }

    st.caption(f"💡 Get your API key: {provider_info.get(provider, 'N/A')}")

    # API Key Input (password field)
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder=f"Enter your {provider} API key",
        help="Your API key is stored securely in session state and never logged"
    )

    # Environment variable fallback info
    env_var_name = f"{provider.upper().replace(' ', '_')}_API_KEY"
    st.caption(f"🔄 Or set env var: `{env_var_name}`")

    # Validation and Save
    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save API Key", type="primary", use_container_width=True):
            if not api_key and not APIManager.get_api_key():
                st.error("❌ Please enter an API key")
                return False

            # Store configuration
            APIManager.set_api_config("open", provider, api_key if api_key else None)
            st.success("✅ API Key saved successfully!")
            st.rerun()

    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            APIManager.clear_api_config()
            st.info("API configuration cleared")
            st.rerun()

    # Show current status
    if APIManager.is_configured() and st.session_state.get("api_type") == "open":
        st.markdown("---")
        st.success(f"🟢 Connected to {st.session_state.api_provider}")

        # Show masked key
        if st.session_state.get("api_key"):
            masked_key = st.session_state.api_key[:8] + "..." + st.session_state.api_key[-4:]
            st.caption(f"Key: `{masked_key}`")

        return True

    return False


def render_closed_api_section() -> bool:
    """Render Closed API (Internal Model) section"""
    st.subheader("🔒 Internal Model")

    st.info("Using default internal model (Ollama)")

    # Show internal model details
    st.markdown("""
    **Model Details:**
    - Provider: Ollama (Local)
    - Model: llama3:8b
    - Endpoint: http://localhost:11434
    - Status: Ready ✅
    """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Use Internal Model", type="primary", use_container_width=True):
            APIManager.set_api_config("closed", "Ollama", None)
            st.success("✅ Internal model configured!")
            st.rerun()

    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            APIManager.clear_api_config()
            st.info("Configuration cleared")
            st.rerun()

    # Show current status
    if APIManager.is_configured() and st.session_state.get("api_type") == "closed":
        st.markdown("---")
        st.success("🟢 Using Internal Ollama Model")
        return True

    return False


# ============================================
# ✅ LLM CLIENT INITIALIZATION
# ============================================

def get_llm_client():
    """Get the appropriate LLM client based on configuration"""
    api_type = st.session_state.get("api_type")
    provider = st.session_state.get("api_provider")

    if api_type == "closed" or not api_type:
        # Use Ollama (default/internal)
        return {
            "type": "ollama",
            "client": ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434")),
            "model": "llama3:8b"
        }

    elif api_type == "open":
        api_key = APIManager.get_api_key()

        if not api_key:
            st.error("❌ API key not configured. Please set it in the sidebar.")
            return None

        # Configure provider-specific settings
        if provider == "OpenAI":
            return {
                "type": "openai",
                "client": openai.OpenAI(api_key=api_key),
                "model": "gpt-3.5-turbo"
            }

        elif provider == "Groq":
            return {
                "type": "openai",  # Groq is OpenAI-compatible
                "client": openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1"
                ),
                "model": "llama3-8b-8192"
            }

        elif provider == "Anthropic":
            # Note: Anthropic uses different client, wrap for compatibility
            try:
                import anthropic
                return {
                    "type": "anthropic",
                    "client": anthropic.Anthropic(api_key=api_key),
                    "model": "claude-3-haiku-20240307"
                }
            except ImportError:
                st.error("❌ Anthropic package not installed. Run: `pip install anthropic`")
                return None

        elif provider == "Moonshot AI":
            return {
                "type": "openai",
                "client": openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.moonshot.cn/v1"
                ),
                "model": "moonshot-v1-8k"
            }

        elif provider == "Custom":
            custom_url = st.text_input("Custom API Base URL", placeholder="https://api.example.com/v1")
            custom_model = st.text_input("Custom Model Name", placeholder="model-name")

            if custom_url and custom_model:
                return {
                    "type": "openai",
                    "client": openai.OpenAI(api_key=api_key, base_url=custom_url),
                    "model": custom_model
                }
            return None

    return None


def get_llm_response(llm_config: dict, prompt: str) -> str:
    """Get response from configured LLM"""
    if not llm_config:
        return "Error: LLM not configured. Please check the sidebar."

    try:
        if llm_config["type"] == "ollama":
            response = llm_config["client"].chat(
                model=llm_config["model"],
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]

        elif llm_config["type"] == "openai":
            response = llm_config["client"].chat.completions.create(
                model=llm_config["model"],
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        elif llm_config["type"] == "anthropic":
            response = llm_config["client"].messages.create(
                model=llm_config["model"],
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

    except Exception as e:
        st.error(f"❌ LLM Error: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"


# ============================================
# ✅ BACKEND INITIALIZATION
# ============================================

@st.cache_resource
def init_backend():
    """Initialize backend resources (cached)"""
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=6333,
        check_compatibility=False
    )
    model = SentenceTransformer("all-MiniLM-L6-v2")
    setup_qdrant(client)
    return client, model


# Initialize backend
client, model = init_backend()

# TTS engine
engine = pyttsx3.init()


# ============================================
# ✅ CORE FUNCTIONS
# ============================================

def get_response(user_input: str) -> str:
    """Get response from LLM with memory storage"""
    prompt = build_prompt(client, model, user_input)

    # Get LLM configuration
    llm_config = get_llm_client()

    if not llm_config:
        return "Please configure your API settings in the sidebar first."

    answer = get_llm_response(llm_config, prompt)

    # Store memory
    store_memory(client, model, user_input, "user")
    store_memory(client, model, answer, "assistant")

    return answer


def listen() -> str:
    """Capture voice input"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Listening...")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        return text
    except Exception:
        st.warning("❌ Could not understand audio")
        return ""


def text_to_audio(text: str) -> str:
    """Convert text to audio file"""
    file_path = "temp_audio.mp3"
    engine.save_to_file(text, file_path)
    engine.runAndWait()
    return file_path


# ============================================
# ✅ MAIN UI
# ============================================

def main():
    """Main application"""
    # Page config
    st.set_page_config(
        layout="wide",
        page_title="AI Voice Assistant",
        page_icon="🤖"
    )

    # Initialize API manager
    api_manager = APIManager()

    # Render sidebar and check if API is configured
    api_ready = render_api_sidebar()

    # Show main content
    st.title("🤖 AI Voice Assistant")

    # Check if API is configured before allowing interaction
    if not APIManager.is_configured():
        st.warning("⚠️ Please configure your API settings in the sidebar to start chatting")
        st.info("👈 Select 'Open API' to use external providers like OpenAI/Groq, or 'Closed API' for local Ollama")

        # Still show the layout but disable input
        show_interface(disabled=True)
    else:
        # Show active configuration
        api_type = st.session_state.get("api_type")
        provider = st.session_state.get("api_provider", "Ollama")

        if api_type == "open":
            st.success(f"🟢 Connected to {provider} API")
        else:
            st.info("🔒 Using Internal Ollama Model")

        show_interface(disabled=False)


def show_interface(disabled: bool = False):
    """Show main chat interface"""
    # Layout: 2 columns
    col1, col2 = st.columns([3, 1])

    # LEFT PANEL (Chat)
    with col1:
        st.subheader("💬 Chat")

        # Initialize messages
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_input = st.chat_input(
            "Type your message..." if not disabled else "Configure API first...",
            disabled=disabled
        )

        # Voice button
        if st.button("🎤 Speak", disabled=disabled):
            voice_text = listen()
            if voice_text:
                user_input = voice_text
                st.write(f"You (voice): {voice_text}")

        # Handle input
        if user_input and not disabled:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.markdown(user_input)

            # Get response
            with st.spinner("Thinking..."):
                response = get_response(user_input)

            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})

            with st.chat_message("assistant"):
                st.markdown(response)

            # Audio playback
            try:
                audio_path = text_to_audio(response)
                with open(audio_path, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")
            except Exception as e:
                st.caption("🔇 Audio playback unavailable")

    # RIGHT PANEL (Avatar + PDF Upload)
    with col2:
        st.subheader("🎭 Avatar")

        st.image(
            "https://cdn-icons-png.flaticon.com/512/4712/4712035.png",
            caption="AI Avatar"
        )

        st.markdown("### Status")
        if APIManager.is_configured():
            st.success("Online ✅")
        else:
            st.error("Offline ❌")

        st.markdown("### Mode")
        if st.session_state.get("api_type") == "open":
            st.info(f"🌐 {st.session_state.get('api_provider', 'Open API')}")
        else:
            st.info("🔒 Internal Ollama")

        # PDF Upload Section
        st.markdown("---")
        st.markdown("### 📄 PDF Import")

        if disabled:
            st.warning("Configure API to enable PDF import")
        else:
            uploaded_pdf = st.file_uploader(
                "Upload a PDF document",
                type=["pdf"],
                help="Upload a PDF to enable question answering based on its content",
                disabled=disabled
            )

            if uploaded_pdf is not None:
                if st.button("🔄 Process PDF", disabled=disabled):
                    with st.spinner("Extracting text from PDF..."):
                        pdf_text = extract_pdf_text(uploaded_pdf)

                        if pdf_text:
                            chunks_stored = store_pdf_content(
                                client, model, pdf_text, uploaded_pdf.name
                            )
                            st.success(f"✅ PDF processed! Stored {chunks_stored} chunks.")
                            st.session_state.pdf_loaded = True
                        else:
                            st.error("❌ Could not extract text from PDF.")

        # Show PDF status
        if st.session_state.get("pdf_loaded", False):
            st.markdown("**PDF Status:** 🟢 Loaded")
        else:
            st.markdown("**PDF Status:** ⚪ No PDF loaded")


# Run main app
if __name__ == "__main__":
    main()
