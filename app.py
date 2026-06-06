import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nrclex import NRCLex
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Config ───────────────────────────────────────────────────
HF_MODEL_NAME = "ShiaHC/empathetic-chatbot-v4"

# ── Emotion detection ────────────────────────────────────────
def extract_top_emotion(text):
    text = str(text).strip()
    if not text:
        return "neutral"
    try:
        emotion = NRCLex()
        emotion.load_raw_text(text)
        scores = emotion.affect_frequencies
        scores = {
            k: v for k, v in scores.items()
            if k not in ["positive", "negative"]
        }
        if not scores:
            return "neutral"
        return max(scores, key=scores.get)
    except:
        return "neutral"

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME).to(device)
    model.eval()
    return model, tokenizer, device

# ── Generate response ────────────────────────────────────────
def generate_response(model, tokenizer, device,
                      situation, user_msg, emotion, history):
    history_text = ""
    if history:
        for turn in history[-3:]:
            history_text += f"User: {turn['user']}\nBot: {turn['bot']}\n"

    prompt  = f"Situation: {situation or 'General conversation'}\n"
    if emotion and emotion != 'neutral':
        prompt += f"Detected Emotion: {emotion}\n"
    if history_text:
        prompt += f"Previous conversation:\n{history_text}\n"
    prompt += f"User: {user_msg}\n\n"
    prompt += "Provide an empathetic response to the user."

    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Empathetic Mental Health Chatbot",
    page_icon="💚",
    layout="centered"
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #f0f7f4; }
.chat-user {
    background-color: #dcf8c6;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 8px 0;
    text-align: right;
    margin-left: 20%;
}
.chat-bot {
    background-color: #ffffff;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 8px 0;
    text-align: left;
    border-left: 4px solid #4CAF50;
    margin-right: 20%;
}
.emotion-badge {
    background-color: #e8f5e9;
    color: #2e7d32;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 0.75em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/mental-health.png", width=80)
    st.header("⚙️ Settings")
    situation = st.text_area(
        "Describe your situation:",
        placeholder="e.g. I have been feeling overwhelmed at work lately...",
        height=120
    )
    st.divider()
    st.markdown("### 💡 About")
    st.markdown("""
    This chatbot uses an **emotion-aware AI**
    fine-tuned on empathetic dialogues to
    provide supportive responses.

    - **Model:** Flan-T5 Base
    - **Emotion:** NRCLex detection
    - **Dataset:** EmpatheticDialogues
    """)
    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()

# ── Session state ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Welcome message
    st.session_state.messages.append({
        'role'   : 'assistant',
        'content': "Hello! I am here to listen and support you. How are you feeling today?",
        'emotion': ''
    })
if "history" not in st.session_state:
    st.session_state.history = []

# ── Header ───────────────────────────────────────────────────
st.title("💚 Empathetic Mental Health Chatbot")
st.markdown("*A safe space to share your feelings. I am here to listen.*")
st.divider()

# ── Chat display ─────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        st.markdown(
            f'<div class="chat-user">🧑 {msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        badge = ""
        if msg.get('emotion') and msg['emotion'] != 'neutral':
            badge = f'<br><span class="emotion-badge">🎭 Detected: {msg["emotion"]}</span>'
        st.markdown(
            f'<div class="chat-bot">💚 {msg["content"]}{badge}</div>',
            unsafe_allow_html=True
        )

# ── Chat input ───────────────────────────────────────────────
user_input = st.chat_input("Share what is on your mind...")

if user_input:
    # Show user message
    st.session_state.messages.append({
        'role'   : 'user',
        'content': user_input
    })

    # Detect emotion
    with st.spinner("Understanding your feelings..."):
        detected_emotion = extract_top_emotion(user_input)

    # Generate response
    with st.spinner("Thinking of a caring response..."):
        try:
            model, tokenizer, device = load_model()
            response = generate_response(
                model, tokenizer, device,
                situation = situation or "General conversation",
                user_msg  = user_input,
                emotion   = detected_emotion,
                history   = st.session_state.history
            )
            if not response.strip():
                response = "I am here to listen. Could you tell me more about how you are feeling?"
        except Exception as e:
            response = "I am here to listen. Could you tell me more?"
            st.error(f"Error: {e}")

    # Update history
    st.session_state.history.append({
        'user': user_input,
        'bot' : response
    })

    # Show bot response
    st.session_state.messages.append({
        'role'   : 'assistant',
        'content': response,
        'emotion': detected_emotion
    })

    st.rerun()

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>⚠️ For research purposes only. "
    "If you are in crisis, please contact a mental health professional.</small></center>",
    unsafe_allow_html=True
)
