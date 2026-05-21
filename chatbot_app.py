import os
import streamlit as st
import time

from transformers import AutoTokenizer, AutoModelForCausalLM

GEMMA_MODEL = os.getenv("GEMMA_MODEL", "google/gemma-2-2b-it")

@st.cache_resource
def load_llm(model_name: str = GEMMA_MODEL):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # FIX 1: Changed to AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model '{model_name}'. "
            "Make sure the model is available locally or set GEMMA_MODEL to a valid Hugging Face repo ID, "
            "and authenticate with `huggingface-cli login` if necessary."
        ) from exc
    return model, tokenizer

def generate_answer(model, tokenizer, context: str, question: str, max_length: int = 256):
    prompt = (
        "Use the following context to answer the question. "
        "If the answer is not contained in the context, say that you do not know.\n\n"
        "Context:\n" + context + "\n\nQuestion: " + question + "\nAnswer:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_length,
        do_sample=False,
    )
    
    # FIX 3: Slice the output to only grab the newly generated tokens, ignoring the prompt
    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

st.set_page_config(page_title="SnarkyBot", page_icon="🤖")

st.title("🤖 SnarkyBot Interface")
st.markdown("A simple conversational interface built with Streamlit.")

with st.sidebar:
    st.header("LLM Options")
    llm_model_name = GEMMA_MODEL
    st.text_input("LLM model (fixed)", value=llm_model_name, disabled=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am SnarkyBot. Say something, and I'll reply."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message here..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner(f"Generating reply with {llm_model_name}..."):
            try:
                llm_model, llm_tokenizer = load_llm(llm_model_name)
                history = "\n\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])
                assistant_response = generate_answer(llm_model, llm_tokenizer, history, prompt)
            except Exception as exc:
                assistant_response = f"LLM generation failed: {exc}"

        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.08)
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})