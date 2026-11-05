import streamlit as st
import pandas as pd
import plotly.express as px
from html import escape
from predict import classify_support_ticket


st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071426 0%, #102747 55%, #071426 100%);
    color: #f1f5f9;
}

header[data-testid="stHeader"] {
    background: #071426 !important;
}

[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    background: transparent !important;
}

.block-container {
    max-width: 1450px;
    padding: 2rem 1rem;
}

h1, h2, h3, p, label {
    color: #f1f5f9 !important;
}

.hero {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 28px;
}

.hero-icon {
    width: 54px;
    height: 54px;
    background: #173e75;
    border: 1px solid #37669f;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
}

.hero h1 {
    font-size: 38px;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: #a9bfdc !important;
    font-size: 16px;
    margin: 5px 0 0;
}

.section-heading {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 25px 0 15px;
}

.section-icon {
    width: 34px;
    height: 34px;
    border-radius: 8px;
    background: #173e75;
    border: 1px solid #37669f;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 17px;
}

.section-heading h3 {
    font-size: 19px;
    margin: 0;
    font-weight: 700;
}

.metric-card {
    background: linear-gradient(135deg, #173e75, #102b50);
    border: 1px solid #37669f;
    border-radius: 12px;
    padding: 20px;
    min-height: 125px;
}

.metric-top {
    display: flex;
    align-items: center;
    gap: 9px;
}

.metric-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: rgba(255,255,255,0.1);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 17px;
}

.metric-label {
    color: #b9cce5;
    font-size: 14px;
}

.metric-value {
    color: white;
    font-size: 25px;
    font-weight: 750;
    margin-top: 12px;
}

.priority-high {
    background: linear-gradient(135deg, #632b3b, #421d32);
    border-color: #b94b60;
}

.priority-medium {
    background: linear-gradient(135deg, #65491e, #403018);
    border-color: #c28b35;
}

.priority-low {
    background: linear-gradient(135deg, #174d43, #12362f);
    border-color: #2b9679;
}

.review-banner,
.accepted-banner {
    display: flex;
    align-items: center;
    gap: 13px;
    border-radius: 10px;
    padding: 15px 18px;
    margin: 15px 0;
}

.review-banner {
    background: rgba(126, 83, 16, 0.35);
    border: 1px solid #e7a928;
    color: #fff2cc;
}

.accepted-banner {
    background: rgba(20, 111, 75, 0.25);
    border: 1px solid #27ae83;
    color: #d1fae5;
}

.banner-icon {
    font-size: 23px;
}

.banner-title {
    font-weight: 700;
    margin-bottom: 3px;
}

.banner-description {
    font-size: 13px;
    opacity: 0.9;
}

div.stButton > button {
    background: linear-gradient(90deg, #1689f8, #1764db);
    color: white;
    border: none;
    border-radius: 8px;
    min-height: 46px;
    font-weight: 700;
}

div.stButton > button:hover {
    background: #2879ed;
    color: white;
    border: none;
}

div[data-baseweb="select"] > div,
.stTextArea textarea {
    background-color: #172d4c !important;
    color: white !important;
    border-color: #39577d !important;
}

.stTextArea textarea::placeholder {
    color: #8fa7c5 !important;
}

.stProgress > div > div > div > div {
    background-color: #299af2;
}

[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"],
[data-testid="stChart"],
[data-testid="stPlotlyChart"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

[data-testid="stJson"],
[data-testid="stCode"] {
    background: transparent !important;
    border: none !important;
}

hr {
    border-color: #29466b;
}

.category-row {
    margin: 15px 0 20px;
}

.category-heading {
    display: flex;
    justify-content: space-between;
    color: #dceaff;
    font-size: 14px;
    margin-bottom: 8px;
}

.category-track {
    height: 9px;
    width: 100%;
    background: #243d5d;
    border-radius: 20px;
    overflow: hidden;
}

.category-fill {
    height: 100%;
    background: linear-gradient(90deg, #1689f8, #38bdf8);
    border-radius: 20px;
}

.analysis-label {
    color: #8faed1;
    font-size: 13px;
    margin-top: 17px;
}

.analysis-value {
    color: #f1f5f9;
    font-size: 15px;
    margin-top: 5px;
    line-height: 1.5;
}

.analysis-item {
    display: flex;
    align-items: center;
    gap: 9px;
}

.analysis-icon {
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)


def section_heading(icon, title):
    st.markdown(f"""
    <div class="section-heading">
        <div class="section-icon">{icon}</div>
        <h3>{title}</h3>
    </div>
    """, unsafe_allow_html=True)


def metric_card(icon, label, value, extra_class=""):
    st.markdown(f"""
    <div class="metric-card {extra_class}">
        <div class="metric-top">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
        </div>
        <div class="metric-value">{escape(str(value))}</div>
    </div>
    """, unsafe_allow_html=True)


def analysis_item(icon, label, value=None):
    st.markdown(f"""
    <div class="analysis-item">
        <span class="analysis-icon">{icon}</span>
        <div class="analysis-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

    if value is not None:
        st.markdown(
            f'<div class="analysis-value">{escape(str(value))}</div>',
            unsafe_allow_html=True
        )


st.markdown("""
<div class="hero">
    <div class="hero-icon">🎫</div>
    <div>
        <h1>Support Ticket Classification</h1>
        <p>An NLP-powered system for classifying IT support tickets and assigning priority levels.</p>
    </div>
</div>
""", unsafe_allow_html=True)


section_heading("📝", "Submit a Support Ticket")

examples = {
    "Select an example": "",
    "Critical system failure": "Critical system down. Employees cannot access their company accounts.",
    "Password problem": "I cannot access my account because my password is not working.",
    "Hardware issue": "My laptop screen is broken and needs replacement.",
    "Storage request": "Please increase the storage capacity on my company drive.",
    "HR assistance": "I need assistance with my employee records."
}

selected_example = st.selectbox(
    "Try an example ticket",
    list(examples.keys())
)

ticket_text = st.text_area(
    "Describe your issue",
    value=examples[selected_example],
    height=130,
    placeholder="Enter your support ticket here..."
)

if st.button("✦  Classify Ticket", type="primary", use_container_width=True):
    if not ticket_text.strip():
        st.warning("Please enter a support ticket.")
    else:
        with st.spinner("Analysing support ticket..."):
            st.session_state["result"] = classify_support_ticket(ticket_text)
            st.session_state["ticket_text"] = ticket_text


if "result" in st.session_state:
    result = st.session_state["result"]
    submitted_ticket = st.session_state["ticket_text"]
    alternatives = result["Alternative Categories"]

    st.markdown("---")
    section_heading("📊", "Classification Results")

    category_col, confidence_col, priority_col = st.columns(3)

    with category_col:
        metric_card("📁", "Predicted Category", result["Category"])

    with confidence_col:
        metric_card("🎯", "Confidence", f'{result["Confidence"]:.2f}%')
        st.progress(result["Confidence"] / 100)

    with priority_col:
        priority_icons = {
            "High": "🔴",
            "Medium": "🟠",
            "Low": "🟢"
        }

        priority_classes = {
            "High": "priority-high",
            "Medium": "priority-medium",
            "Low": "priority-low"
        }

        priority = result["Priority"]
        metric_card(
            priority_icons.get(priority, "⚪"),
            "Priority Level",
            priority,
            priority_classes.get(priority, "")
        )

    if result["Status"] == "Needs human review":
        st.markdown("""
        <div class="review-banner">
            <div class="banner-icon">⚠️</div>
            <div>
                <div class="banner-title">Needs human review</div>
                <div class="banner-description">
                    This ticket has been flagged for manual review.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="accepted-banner">
            <div class="banner-icon">✓</div>
            <div>
                <div class="banner-title">Prediction accepted</div>
                <div class="banner-description">
                    This ticket meets the current prediction acceptance criteria.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    chart_col, analysis_col = st.columns([1.2, 1])

    with chart_col:
        section_heading("📈", "Alternative Categories")

        chart_data = pd.DataFrame({
            "Category": list(alternatives.keys()),
            "Confidence": list(alternatives.values())
        })

        fig = px.bar(
            chart_data,
            x="Category",
            y="Confidence",
            color_discrete_sequence=["#1689f8"]
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#f1f5f9",
                family="Arial"
            ),
            xaxis=dict(
                title=None,
                showgrid=False,
                tickangle=-25,
                tickfont=dict(color="#dceaff")
            ),
            yaxis=dict(
                title="Confidence (%)",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.12)",
                tickfont=dict(color="#dceaff"),
                range=[0, 100]
            ),
            margin=dict(l=10, r=10, t=20, b=10),
            height=380,
            showlegend=False
        )

        fig.update_traces(
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>Confidence: %{y:.2f}%<extra></extra>"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with analysis_col:
        section_heading("🔍", "Ticket Analysis")

        analysis_item("📄", "Original Ticket", submitted_ticket)
        analysis_item("📌", "Prediction Status", result["Status"])
        analysis_item("📊", "Top Category Predictions")

        for category, confidence in alternatives.items():
            width = max(0, min(confidence, 100))

            st.markdown(f"""
            <div class="category-row">
                <div class="category-heading">
                    <span>{escape(str(category))}</span>
                    <span>{confidence:.2f}%</span>
                </div>
                <div class="category-track">
                    <div class="category-fill" style="width: {width}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


st.markdown("---")

st.caption(
    "Classification uses a trained Random Forest model with TF-IDF features. "
    "Priority is assigned using predefined keyword rules. "
    "High-priority and low-confidence tickets are flagged for human review."
)