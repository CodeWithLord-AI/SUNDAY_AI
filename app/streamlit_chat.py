"""
SUNDAY AI - Streamlit Chat Interface
=======================================

Phase 11 Implementation.

A beautiful interactive chat UI for SUNDAY LLM built with Streamlit.

Features:
  - Real-time streaming generation
  - Conversation history display
  - Model configuration controls in sidebar
  - Load model from checkpoint
  - Display model info (parameters, config)
  - Reset conversation button

Usage:
    streamlit run app/streamlit_chat.py

Requirements:
    - Trained model checkpoint in checkpoints/best_model.pt
    - Trained tokenizer in tokenizer_files/sunday_tokenizer.json
    - Or use a demo mode with a tiny untrained model
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import torch

# --- Page Config ---
st.set_page_config(
    page_title="SUNDAY AI — Chat",
    page_icon="🌅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B35;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .model-info {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .chat-message-user {
        background: #e3f2fd;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .chat-message-ai {
        background: #fff3e0;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State Init
# ============================================================
def init_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "generator" not in st.session_state:
        st.session_state.generator = None
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = None
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = None


init_session_state()


# ============================================================
# Model Loading
# ============================================================
@st.cache_resource(show_spinner="Loading SUNDAY model...")
def load_model(
    model_size: str,
    checkpoint_path: str,
    tokenizer_path: str,
    device_str: str,
):
    """Load model, tokenizer, and create a Generator."""
    from src.model.sunday_model import SundayModel
    from src.tokenizer.tokenizer import SundayTokenizer
    from src.inference.generator import Generator
    from src.training.checkpoint import load_checkpoint

    device = torch.device(device_str)

    # Load tokenizer
    tokenizer = SundayTokenizer.from_file(tokenizer_path)

    # Load model
    model = SundayModel.from_config(model_size)

    if checkpoint_path and os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model, device=device_str)
    else:
        st.warning("⚠️ No checkpoint found. Using randomly initialized weights (for demo only).")

    model.to(device)
    model.eval()

    # Create Generator
    generator = Generator(model, tokenizer, device)

    return generator, tokenizer, model


# ============================================================
# Sidebar: Configuration
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ SUNDAY AI Configuration")
    st.markdown("---")

    # Model settings
    st.markdown("### 🧠 Model")
    model_size = st.selectbox(
        "Model Size",
        ["tiny", "50m", "125m"],
        index=0,
        help="Tiny (5M) for demo, 50M/125M require training."
    )

    checkpoint_path = st.text_input(
        "Checkpoint Path",
        value="checkpoints/best_model.pt",
        help="Path to trained model checkpoint (.pt file)"
    )

    tokenizer_path = st.text_input(
        "Tokenizer Path",
        value="tokenizer_files/sunday_tokenizer.json",
        help="Path to trained BPE tokenizer (.json file)"
    )

    device_options = ["cpu"]
    if torch.cuda.is_available():
        device_options.insert(0, "cuda")
    device_str = st.selectbox("Device", device_options)

    instruction_tuned = st.checkbox(
        "Instruction-tuned model",
        value=False,
        help="Enable if model was fine-tuned on Alpaca data"
    )

    if st.button("🚀 Load Model", type="primary"):
        try:
            gen, tok, mdl = load_model(
                model_size, checkpoint_path, tokenizer_path, device_str
            )
            st.session_state.generator = gen
            st.session_state.tokenizer = tok

            # Create chat session
            from src.inference.chat import ChatSession
            st.session_state.chat_session = ChatSession(
                gen, instruction_tuned=instruction_tuned
            )

            st.session_state.model_loaded = True

            # Show model info
            counts = mdl.count_parameters()
            st.success(f"✅ Model loaded! ({counts['total']/1e6:.1f}M params)")
            st.info(
                f"**{model_size.upper()} Config:**\n"
                f"- Parameters: {counts['total']:,}\n"
                f"- Device: {device_str}"
            )
        except Exception as e:
            st.error(f"❌ Failed to load model:\n{e}")

    st.markdown("---")
    st.markdown("### 🎛️ Generation Settings")

    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.05,
        help="Higher = more creative, Lower = more focused"
    )

    top_k = st.slider(
        "Top-K",
        min_value=0,
        max_value=200,
        value=50,
        step=5,
        help="Sample from top K tokens (0 = disabled)"
    )

    top_p = st.slider(
        "Top-P (Nucleus)",
        min_value=0.5,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Sample from top-p probability mass"
    )

    max_new_tokens = st.slider(
        "Max New Tokens",
        min_value=20,
        max_value=500,
        value=200,
        step=10,
    )

    st.markdown("---")

    if st.button("🗑️ Reset Conversation"):
        st.session_state.messages = []
        if st.session_state.chat_session:
            st.session_state.chat_session.reset()
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "**SUNDAY AI** is a GPT-style language model "
        "built from scratch with PyTorch.\n\n"
        "- 🔤 Custom BPE Tokenizer (32K vocab)\n"
        "- 🧠 Decoder-only Transformer\n"
        "- 💡 Optionally instruction-tuned\n"
        "- 🎓 Educational project"
    )


# ============================================================
# Main Chat Area
# ============================================================
st.markdown('<div class="main-header">🌅 SUNDAY AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">GPT-style Language Model — Built from Scratch</div>',
    unsafe_allow_html=True
)

if not st.session_state.model_loaded:
    st.info(
        "👈 **Configure and load a model using the sidebar to start chatting.**\n\n"
        "If you don't have a trained checkpoint yet, you can still load the model "
        "with random weights (use **tiny** size) to test the UI pipeline."
    )

    # Demo mode without loading
    with st.expander("🚀 Quick Demo (Random Weights)"):
        st.markdown(
            "Click **Load Model** in the sidebar with **tiny** size "
            "and any tokenizer path. The model will use random weights."
        )
else:
    # --- Display Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat Input ---
    if prompt := st.chat_input("Chat with SUNDAY AI..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                if st.session_state.chat_session:
                    # Streaming generation
                    for token in st.session_state.chat_session.stream_chat(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    ):
                        full_response += token
                        message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)

                else:
                    full_response = "❌ Model not loaded. Please load the model first."
                    message_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"❌ Generation error: {str(e)}"
                message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.markdown("---")
st.markdown(
    "<small>SUNDAY AI — A GPT-style LLM built from scratch | "
    "Educational project | PyTorch + HuggingFace</small>",
    unsafe_allow_html=True
)
