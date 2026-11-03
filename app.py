"""
Neural Text Summarization Application
Powered by facebook/bart-large-cnn & Hugging Face Transformers
"""

from __future__ import annotations
import streamlit as st
from src.summarizer import (
    MODEL_NAME,
    SUMMARY_PRESETS,
    load_model_and_tokenizer,
    generate_summary,
    count_bpe_tokens,
    get_torch_device,
    ModelLoadingError,
)
from src.utils import (
    count_words,
    count_characters,
    calculate_compression_ratio,
    validate_parameters,
    validate_input_text,
)
from src.evaluator import compute_all_rouge_scores
from src.sample_data import SAMPLE_TEXTS

# -----------------------------------------------------------------------------
# 1. Streamlit Application Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Neural Text Summarizer | BART Large CNN",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 500;
        color: #94a3b8;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700;
    }
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #1e293b;
        color: #38bdf8;
        border: 1px solid #0284c7;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-accent {
        background-color: #1e1b4b;
        color: #a78bfa;
        border-color: #6366f1;
    }
    .card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .summary-box {
        font-size: 1.05rem;
        line-height: 1.65;
        padding: 18px;
        background: rgba(56, 189, 248, 0.04);
        border-left: 4px solid #38bdf8;
        border-radius: 6px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .rouge-box {
        background: rgba(167, 139, 250, 0.05);
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
if "input_text_widget" not in st.session_state:
    st.session_state["input_text_widget"] = ""
if "reference_text_widget" not in st.session_state:
    st.session_state["reference_text_widget"] = ""
if "summary_output" not in st.session_state:
    st.session_state.summary_output = ""
if "summary_metadata" not in st.session_state:
    st.session_state.summary_metadata = {}
if "sample_selector_idx" not in st.session_state:
    st.session_state.sample_selector_idx = 0


# -----------------------------------------------------------------------------
# 3. Cached Model Pipeline Loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_summarizer():
    """Cache tokenizer and seq2seq pipeline across user sessions."""
    return load_model_and_tokenizer(MODEL_NAME)


# -----------------------------------------------------------------------------
# 4. Sidebar Controls and Architectural Overview
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/imgs/transformers_logo_name.png",
        width=200,
    )
    st.title("Model & Settings")
    
    st.markdown(
        """
        <div class="card">
            <strong>Model:</strong> facebook/bart-large-cnn<br>
            <strong>Architecture:</strong> Seq2Seq Bidirectional Encoder + Autoregressive Decoder<br>
            <strong>Parameters:</strong> ~406 Million<br>
            <strong>Context Window:</strong> 1,024 Tokens<br>
            <strong>Pretraining:</strong> CNN / DailyMail Corpus
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    device_label, _ = get_torch_device()
    st.caption(f"⚡ **Compute Backend:** `{device_label}`")

    st.subheader("Summary Length Preset")
    preset_choice = st.radio(
        "Choose a target length:",
        options=["Short", "Medium", "Long", "Custom"],
        index=1,
        help="Presets automatically calibrate minimum and maximum token generation constraints.",
    )

    if preset_choice != "Custom":
        preset_cfg = SUMMARY_PRESETS[preset_choice]
        min_length = preset_cfg["min_length"]
        max_length = preset_cfg["max_length"]
        num_beams = preset_cfg["num_beams"]
        length_penalty = preset_cfg["length_penalty"]
        st.info(f"ℹ️ **{preset_choice}**: {preset_cfg['description']} (`{min_length}`–`{max_length}` tokens)")
    else:
        st.markdown("#### Custom Parameters")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            min_length = st.number_input("Min Tokens", min_value=5, max_value=300, value=50, step=5)
        with col_p2:
            max_length = st.number_input("Max Tokens", min_value=15, max_value=500, value=150, step=5)
            
        is_valid_param, param_err = validate_parameters(min_length, max_length)
        if not is_valid_param:
            st.error(f"⚠️ {param_err}")
            
        num_beams = st.slider("Beam Width (num_beams)", min_value=1, max_value=8, value=4,
                              help="Higher values explore more sequence candidates but increase latency.")
        length_penalty = st.slider("Length Penalty", min_value=0.5, max_value=3.0, value=2.0, step=0.1,
                                   help="> 1.0 encourages longer summaries; < 1.0 encourages conciseness.")

    st.markdown("---")
    st.subheader("Context Handling Strategy")
    chunk_strategy = st.selectbox(
        "For texts exceeding 1024 tokens:",
        options=["Smart Chunking", "Safe Truncation"],
        index=0,
        help="Smart Chunking splits text across sentence boundaries and synthesizes chunk summaries.",
    )

    st.markdown("---")
    with st.expander("📖 Step-by-Step Instructions"):
        st.markdown(
            """
            1. **Select or Paste Text**: Choose one of the sample articles or paste your own document.
            2. **Configure Length**: Choose a preset (*Short*, *Medium*, *Long*) or customize beam constraints.
            3. **Generate**: Click **Generate Summary** to run BART abstractive inference.
            4. **Evaluate (Optional)**: Open the **ROUGE Evaluation** drawer and paste a reference summary to measure n-gram recall and precision.
            """
        )

# -----------------------------------------------------------------------------
# 5. Main Application Header
# -----------------------------------------------------------------------------
st.title("Neural Text Summarizer")
st.markdown(
    """
    Transform long-form articles, research passages, and documents into clear, coherent abstractive summaries.
    Powered by Meta AI's **BART Large CNN** model using deterministic beam search.
    """
)

st.markdown(
    """
    <div>
        <span class="badge">facebook/bart-large-cnn</span>
        <span class="badge badge-accent">Abstractive Seq2Seq</span>
        <span class="badge">PyTorch</span>
        <span class="badge">Deterministic Beam Search</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# -----------------------------------------------------------------------------
# 6. Model Loading Verification
# -----------------------------------------------------------------------------
try:
    with st.spinner("Initializing BART summarization pipeline..."):
        tokenizer, model, summarizer_pipeline, _ = get_summarizer()
except ModelLoadingError as e:
    st.error(f"❌ **Model Initialization Failed**: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"❌ **Unexpected Error Loading Model**: {str(e)}")
    st.stop()

# -----------------------------------------------------------------------------
# 7. Sample Text Selection and User Inputs
# -----------------------------------------------------------------------------
col_sample, col_spacer = st.columns([2, 1])
with col_sample:
    sample_options = ["-- Or load an exemplary sample article --"] + [
        f"{s['category']}: {s['title']}" for s in SAMPLE_TEXTS
    ]
    
    def on_sample_change():
        chosen_idx = st.session_state.sample_selector_idx
        if chosen_idx > 0:
            sample_data = SAMPLE_TEXTS[chosen_idx - 1]
            st.session_state["input_text_widget"] = sample_data["text"]
            st.session_state["reference_text_widget"] = sample_data["reference_summary"]
            st.session_state.summary_output = ""
            st.session_state.summary_metadata = {}
            
    st.selectbox(
        "Load Sample Article:",
        options=range(len(sample_options)),
        format_func=lambda i: sample_options[i],
        key="sample_selector_idx",
        on_change=on_sample_change,
    )

# Text Area for Input (bound to st.session_state["input_text_widget"])
input_text = st.text_area(
    "Source Document / Passage:",
    height=240,
    placeholder="Paste or type article, research paper abstract, or news passage here (at least 10 words)...",
    key="input_text_widget",
)

# Live Input Word / Token Counters
live_words = count_words(input_text)
live_chars = count_characters(input_text, include_spaces=True)
live_tokens = count_bpe_tokens(input_text, tokenizer)

col_live1, col_live2, col_live3 = st.columns([1, 1, 3])
with col_live1:
    st.caption(f"📄 Words: **{live_words}**")
with col_live2:
    st.caption(f"🔤 Characters: **{live_chars}**")
with col_live3:
    if live_tokens > 1024:
        st.caption(f"⚠️ Tokens: **{live_tokens}** / 1,024 (*Exceeds single context window; using {chunk_strategy}*)")
    else:
        st.caption(f"🔢 Subword Tokens: **{live_tokens}** / 1,024")

# Action Buttons
btn_col1, btn_col2, _ = st.columns([1.5, 1.2, 5])

with btn_col1:
    generate_clicked = st.button("✨ Generate Summary", type="primary", use_container_width=True)

with btn_col2:
    def clear_inputs():
        st.session_state["input_text_widget"] = ""
        st.session_state["reference_text_widget"] = ""
        st.session_state.summary_output = ""
        st.session_state.summary_metadata = {}
        st.session_state.sample_selector_idx = 0

    st.button("🧹 Clear Text", on_click=clear_inputs, use_container_width=True)

# -----------------------------------------------------------------------------
# 8. Generation Execution
# -----------------------------------------------------------------------------
if generate_clicked:
    is_valid, validation_msg = validate_input_text(input_text, min_words=10)
    if not is_valid:
        st.warning(f"⚠️ {validation_msg}")
    else:
        is_param_valid, param_msg = validate_parameters(min_length, max_length)
        if not is_param_valid:
            st.error(f"⚠️ {param_msg}")
        else:
            with st.spinner("🧠 Generating abstractive summary with BART..."):
                gen_result = generate_summary(
                    pipeline_obj=summarizer_pipeline,
                    tokenizer=tokenizer,
                    text=input_text,
                    min_length=min_length,
                    max_length=max_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    chunk_strategy=chunk_strategy,
                )

            if gen_result["success"]:
                st.session_state.summary_output = gen_result["summary_text"]
                st.session_state.summary_metadata = gen_result
            else:
                st.error(f"❌ {gen_result['error']}")

# -----------------------------------------------------------------------------
# 9. Summary Output and Text Statistics
# -----------------------------------------------------------------------------
if st.session_state.summary_output:
    summary_text = st.session_state.summary_output
    metadata = st.session_state.summary_metadata
    
    st.markdown("---")
    st.subheader("Generated Summary")
    
    if metadata.get("was_chunked"):
        st.info(
            f"ℹ️ Input text was processed in **{metadata.get('chunks_count', 1)} segments** "
            "using hierarchical chunk synthesis to stay within the model's 1024-token context limit."
        )
    elif metadata.get("was_truncated"):
        st.warning(f"⚠️ {metadata.get('note', 'Input was safely truncated to 1024 tokens.')}")

    # Display Summary
    st.markdown(f'<div class="summary-box">{summary_text}</div>', unsafe_allow_html=True)

    # Copy and Download Utilities
    util_col1, util_col2, _ = st.columns([1.5, 2, 4])
    with util_col1:
        st.download_button(
            label="📥 Download (.txt)",
            data=summary_text,
            file_name="bart_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with util_col2:
        with st.popover("📋 View / Copy Raw Text"):
            st.caption("Click the copy icon in the top right corner of the box below:")
            st.code(summary_text, language=None)

    # -------------------------------------------------------------------------
    # Text Statistics & Compression Ratio
    # -------------------------------------------------------------------------
    st.write("")
    st.subheader("Text Statistics")
    
    input_wc = count_words(input_text)
    summary_wc = count_words(summary_text)
    input_cc = count_characters(input_text, include_spaces=True)
    compression = calculate_compression_ratio(input_wc, summary_wc)

    stat1, stat2, stat3, stat4 = st.columns(4)
    with stat1:
        st.metric(label="Original Word Count", value=f"{input_wc:,}")
    with stat2:
        st.metric(label="Summary Word Count", value=f"{summary_wc:,}")
    with stat3:
        st.metric(label="Original Character Count", value=f"{input_cc:,}")
    with stat4:
        st.metric(
            label="Compression Ratio",
            value=f"{compression}%",
            help="Summary word count divided by input word count, expressed as a percentage.",
        )

    st.caption(
        "💡 *Note: Compression ratio reflects the volumetric reduction of text relative to the original source. "
        "It is a quantitative length metric, not an assessment of factual fidelity, semantic quality, or readability.*"
    )

# -----------------------------------------------------------------------------
# 10. Optional Evaluation Section (ROUGE Metrics)
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("🔬 Summary Evaluation (ROUGE Metrics)", expanded=False):
    st.markdown(
        """
        ### Quantitative Evaluation with ROUGE
        **ROUGE** (*Recall-Oriented Understudy for Gisting Evaluation*) evaluates summary quality by measuring 
        lexical overlap against a human-authored **reference summary (ground truth)**.
        
        * **ROUGE-1**: Measures unigram (single word) overlap.
        * **ROUGE-2**: Measures bigram (two consecutive words) overlap, indicating phrase-level fluency.
        * **ROUGE-L**: Measures the Longest Common Subsequence (LCS), assessing structural and sentence-level fidelity.
        
        > ⚠️ **Evaluation requires a human reference summary.** ROUGE is a comparative metric; it cannot evaluate a summary in isolation.
        """
    )
    
    ref_input = st.text_area(
        "Ground-Truth Reference Summary:",
        height=120,
        placeholder="Enter a human-written ground-truth reference summary to compare against...",
        key="reference_text_widget",
    )

    col_eval_btn, _ = st.columns([1.5, 5])
    with col_eval_btn:
        compute_eval = st.button("📊 Calculate ROUGE Scores", use_container_width=True)

    if compute_eval:
        if not st.session_state.summary_output:
            st.warning("⚠️ Please generate a summary first before computing evaluation metrics.")
        elif not ref_input or not ref_input.strip():
            st.warning("⚠️ Please enter a reference summary to calculate ROUGE metrics.")
        else:
            with st.spinner("Computing ROUGE-1, ROUGE-2, and ROUGE-L metrics..."):
                rouge_results = compute_all_rouge_scores(
                    candidate=st.session_state.summary_output,
                    reference=ref_input,
                )

            st.write("")
            r1, r2, rl = st.columns(3)
            
            with r1:
                st.markdown(
                    f"""
                    <div class="rouge-box">
                        <h4 style="margin:0 0 8px 0; color:#38bdf8;">ROUGE-1 (Unigrams)</h4>
                        <strong>F1-Score:</strong> {rouge_results['rouge1']['f1'] * 100:.1f}%<br>
                        <strong>Recall:</strong> {rouge_results['rouge1']['recall'] * 100:.1f}%<br>
                        <strong>Precision:</strong> {rouge_results['rouge1']['precision'] * 100:.1f}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
            with r2:
                st.markdown(
                    f"""
                    <div class="rouge-box">
                        <h4 style="margin:0 0 8px 0; color:#818cf8;">ROUGE-2 (Bigrams)</h4>
                        <strong>F1-Score:</strong> {rouge_results['rouge2']['f1'] * 100:.1f}%<br>
                        <strong>Recall:</strong> {rouge_results['rouge2']['recall'] * 100:.1f}%<br>
                        <strong>Precision:</strong> {rouge_results['rouge2']['precision'] * 100:.1f}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
            with rl:
                st.markdown(
                    f"""
                    <div class="rouge-box">
                        <h4 style="margin:0 0 8px 0; color:#a78bfa;">ROUGE-L (LCS Sequence)</h4>
                        <strong>F1-Score:</strong> {rouge_results['rougeL']['f1'] * 100:.1f}%<br>
                        <strong>Recall:</strong> {rouge_results['rougeL']['recall'] * 100:.1f}%<br>
                        <strong>Precision:</strong> {rouge_results['rougeL']['precision'] * 100:.1f}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                "📌 *Higher recall indicates the summary successfully captures key concepts from the reference. "
                "Higher precision indicates the summary avoids extraneous words not present in the reference.*"
            )

# -----------------------------------------------------------------------------
# 11. Footer and Academic Attribution
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 15px 0;">
        <strong>Neural Text Summarization Platform</strong> &bull; 
        Powered by <a href="https://huggingface.co/facebook/bart-large-cnn" target="_blank" style="color:#38bdf8;">facebook/bart-large-cnn</a> 
        and <a href="https://streamlit.io" target="_blank" style="color:#38bdf8;">Streamlit</a><br>
        BART (Lewis et al., 2019): Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.
    </div>
    """,
    unsafe_allow_html=True,
)
