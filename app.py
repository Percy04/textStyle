import streamlit as st
import torch
import torch.nn.functional as F
import os

# ==== Load model code and functions ====
from updated_model import StyleClassifier, StyleConditionalGenerator, classify_text

# Modified version of generate_text_in_style to accept prompt

def generate_text_in_style(generator, author_choice, num_tokens, stoi, itos, prompt=None):
    """Generate text in chosen author's style with better prompting"""
    # Use provided prompt or default
    if prompt is None:
        prompt = "I was " if author_choice == "Mark Twain" else "It was "

    context = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=generator.lm_head.weight.device)
    context = context.to(generator.lm_head.weight.device)

    style_id = torch.tensor([0 if author_choice == "Mark Twain" else 1], device=generator.lm_head.weight.device)
    style_id = style_id.to(generator.lm_head.weight.device)

    generator = generator.to('cuda' if torch.cuda.is_available() else 'cpu')
    generator.eval()
    with torch.no_grad():
        generated = generator.generate(context, style_id, num_tokens)
        text = ''.join([itos.get(i, '') for i in generated[0].tolist()])

    return text

# ==== Load checkpoint ====
@st.cache_resource
def load_models():
    checkpoint = torch.load('best_twain_dickens_models.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu')

    vocab_size = checkpoint['vocab_size']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']

    classifier = StyleClassifier(
        vocab_size, checkpoint['n_embd'], checkpoint['n_styles']
    )
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    classifier = classifier.to('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.eval()

    generator = StyleConditionalGenerator(
        vocab_size, checkpoint['n_embd'], checkpoint['n_styles']
    )
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator = generator.to('cuda' if torch.cuda.is_available() else 'cpu')
    generator.eval()

    return classifier, generator, stoi, itos

classifier, generator, stoi, itos = load_models()

# ==== Streamlit UI ====
st.set_page_config(page_title="TextStyle")
st.title("Style analyzer and generator")

st.markdown("""
Paste any text below. The app will:
1. Predict whether the style is more like **Mark Twain** or **Charles Dickens** (Can scale)
2. Let you choose a style to generate a continuation
""")

# User input
input_text = st.text_area("Enter your text:", height=200)

if input_text.strip():
    with st.spinner("Classifying style..."):
        result = classify_text(input_text, classifier, stoi, itos)
        predicted_author = result['predicted_author']
        confidence = result['confidence']

    st.success(f"Predicted Style: **{predicted_author}** ({confidence*100:.1f}% confidence)")
    st.markdown("---")

    # Style selection for generation
    style_choice = st.selectbox("Choose a style to continue in:", ["Mark Twain", "Charles Dickens"])
    generate_button = st.button("Generate Continuation")

    if generate_button:
        with st.spinner(f"Generating text in {style_choice}'s style..."):
            continuation = generate_text_in_style(generator, style_choice, 200, stoi, itos, prompt=input_text)
        st.markdown("**Generated Text:**")
        st.code(continuation)

else:
    st.info("Please enter some text to begin.")

