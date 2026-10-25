import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Farmer Assistant",
    page_icon="🌱",
    layout="wide"
)

MODEL_NAME = "mesabo/agri-plant-disease-resnet50"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME
    )

    model = model.to(device)
    model.eval()

    return processor, model, device


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(image, processor, model, device):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(
        outputs.logits,
        dim=-1
    )[0]

    top_results = torch.topk(
        probabilities,
        min(5, len(probabilities))
    )

    results = []

    for score, index in zip(
        top_results.values,
        top_results.indices
    ):

        label = model.config.id2label[
            index.item()
        ]

        confidence = score.item() * 100

        results.append({
            "label": label,
            "confidence": confidence
        })

    return results


# ============================================================
# LABEL CLEANING
# ============================================================

def clean_label(label):

    return label.replace(
        "___",
        " — "
    ).replace(
        "_",
        " "
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

def get_recommendations(label):

    label_lower = label.lower()

    if "healthy" in label_lower:

        return {
            "status": "Healthy / No major disease pattern detected",

            "checks": [
                "Leaf color",
                "New growth",
                "Pest activity",
                "Soil moisture"
            ],

            "actions": [
                "Continue regular crop monitoring.",
                "Maintain appropriate irrigation.",
                "Monitor new leaves for changes.",
                "Check plants regularly for pests or disease."
            ]
        }

    if "early_blight" in label_lower:

        return {
            "status": "Possible Tomato Early Blight",

            "checks": [
                "Circular lesions",
                "Concentric ring patterns",
                "Yellowing around lesions",
                "Spread to nearby leaves"
            ],

            "actions": [
                "Inspect older leaves carefully.",
                "Remove severely affected leaves where appropriate.",
                "Avoid prolonged leaf wetness.",
                "Improve air circulation around plants.",
                "Monitor nearby plants."
            ]
        }

    if "late_blight" in label_lower:

        return {
            "status": "Possible Tomato Late Blight",

            "checks": [
                "Dark irregular lesions",
                "Rapid spread",
                "Water-soaked appearance",
                "Nearby affected plants"
            ],

            "actions": [
                "Inspect both sides of affected leaves.",
                "Look for rapidly expanding dark lesions.",
                "Avoid overhead irrigation.",
                "Monitor nearby plants.",
                "Seek local agricultural guidance."
            ]
        }

    if "septoria" in label_lower:

        return {
            "status": "Possible Septoria Leaf Spot",

            "checks": [
                "Small circular spots",
                "Dark borders",
                "Spots on older leaves",
                "Increasing number of lesions"
            ],

            "actions": [
                "Inspect lower and older leaves.",
                "Remove severely affected foliage where appropriate.",
                "Avoid unnecessary leaf wetness.",
                "Maintain field sanitation.",
                "Monitor disease progression."
            ]
        }

    if "yellow_leaf_curl" in label_lower:

        return {
            "status": "Possible Tomato Yellow Leaf Curl Virus",

            "checks": [
                "Leaf curling",
                "Yellowing",
                "Stunted growth",
                "Whitefly presence"
            ],

            "actions": [
                "Inspect leaves for curling and yellowing.",
                "Check plants for stunted growth.",
                "Inspect for whitefly activity.",
                "Monitor nearby plants.",
                "Seek agricultural expert confirmation."
            ]
        }

    return {
        "status": "Possible crop disease or plant stress",

        "checks": [
            "Leaf appearance",
            "Stem condition",
            "Pest activity",
            "Recent weather",
            "Soil and irrigation conditions"
        ],

        "actions": [
            "Inspect the plant carefully.",
            "Compare symptoms across multiple plants.",
            "Avoid applying treatments based only on this prediction.",
            "Seek local agricultural guidance if symptoms persist."
        ]
    }


# ============================================================
# HEADER
# ============================================================

st.title("🌱 AI Farmer Assistant")

st.markdown(
    """
### AI-Powered Crop Health & Decision Support

Upload a crop image and the system will analyze
visual patterns and generate a farmer-friendly report.
"""
)

st.warning(
    "⚠️ Academic prototype: AI predictions are not a confirmed "
    "agricultural diagnosis. Human expert verification is recommended."
)


# ============================================================
# MODEL
# ============================================================

with st.spinner("Loading Agriculture AI model..."):

    processor, model, device = load_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("👨‍🌾 Farmer Information")

    farmer_name = st.text_input(
        "Farmer Name"
    )

    location = st.text_input(
        "Location",
        placeholder="Example: Nashik, Maharashtra"
    )

    crop = st.selectbox(
        "Crop",
        [
            "Tomato",
            "Potato",
            "Squash",
            "Other"
        ]
    )

    growth_stage = st.selectbox(
        "Growth Stage",
        [
            "Seedling",
            "Vegetative",
            "Flowering",
            "Fruiting",
            "Maturity"
        ]
    )

    st.divider()

    st.info(
        f"AI Device: {device}"
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🌾 AI Crop Diagnosis",
        "🧪 Prompt Engineering",
        "📊 Evaluation"
    ]
)


# ============================================================
# TAB 1
# ============================================================

with tab1:

    st.header(
        "📷 Crop Image Analysis"
    )

    uploaded_file = st.file_uploader(
        "Upload a crop / leaf image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns(
            2
        )

        with col1:

            st.subheader(
                "📷 Uploaded Image"
            )

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader(
                "📝 Farmer Observation"
            )

            symptoms = st.text_area(
                "Describe what you observe",
                placeholder=(
                    "Example: Brown spots are appearing "
                    "on older leaves and the leaves are turning yellow."
                ),
                height=150
            )

            analyze_button = st.button(
                "🔍 Analyze Crop",
                type="primary",
                use_container_width=True
            )

        if analyze_button:

            with st.spinner(
                "AI is analyzing the crop..."
            ):

                results = analyze_image(
                    image,
                    processor,
                    model,
                    device
                )

            best = results[0]

            clean_best = clean_label(
                best["label"]
            )

            confidence = best["confidence"]

            recommendation = get_recommendations(
                best["label"]
            )

            # =================================================
            # AI RESULTS
            # =================================================

            st.divider()

            st.header(
                "🧠 AI Analysis"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Top Prediction",
                    clean_best
                )

            with c2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            with c3:

                if confidence >= 70:
                    reliability = "Higher"
                elif confidence >= 40:
                    reliability = "Moderate"
                else:
                    reliability = "Low"

                st.metric(
                    "Reliability",
                    reliability
                )

            # =================================================
            # CONFIDENCE WARNING
            # =================================================

            if confidence < 40:

                st.error(
                    "⚠️ LOW CONFIDENCE — The model is uncertain. "
                    "Do not treat this prediction as a diagnosis."
                )

            elif confidence < 70:

                st.warning(
                    "⚠️ MODERATE CONFIDENCE — Additional inspection "
                    "and expert verification are recommended."
                )

            else:

                st.success(
                    "The model produced a stronger prediction, "
                    "but expert verification is still recommended."
                )

            # =================================================
            # TOP 5
            # =================================================

            st.subheader(
                "📊 Top 5 AI Predictions"
            )

            prediction_data = []

            for result in results:

                prediction_data.append({

                    "Prediction":
                        clean_label(
                            result["label"]
                        ),

                    "Confidence (%)":
                        round(
                            result["confidence"],
                            2
                        )
                })

            prediction_df = pd.DataFrame(
                prediction_data
            )

            st.dataframe(
                prediction_df,
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                prediction_df.set_index(
                    "Prediction"
                )
            )

            # =================================================
            # FARMER INTERPRETATION
            # =================================================

            st.divider()

            st.header(
                "🌾 Farmer-Friendly Interpretation"
            )

            st.write(
                recommendation["status"]
            )

            # =================================================
            # CROP HEALTH REPORT
            # =================================================

            st.divider()

            st.header(
                "📋 Crop Health Report"
            )

            st.write(
                "Summary generated from the AI image analysis "
                "and farmer-provided information."
            )

            r1, r2 = st.columns(2)

            with r1:

                st.subheader(
                    "🌱 Crop Information"
                )

                st.write(
                    f"**Crop:** {crop}"
                )

                st.write(
                    f"**Growth Stage:** {growth_stage}"
                )

                if farmer_name:

                    st.write(
                        f"**Farmer:** {farmer_name}"
                    )

                if location:

                    st.write(
                        f"**Location:** {location}"
                    )

                st.subheader(
                    "🦠 Possible Condition"
                )

                st.write(
                    clean_best
                )

                st.write(
                    f"**AI Confidence:** {confidence:.2f}%"
                )

            with r2:

                st.subheader(
                    "🔎 AI Evidence"
                )

                for result in results:

                    st.write(
                        f"• {clean_label(result['label'])} — "
                        f"{result['confidence']:.2f}%"
                    )

            st.subheader(
                "📝 Farmer Observation"
            )

            if symptoms:

                st.write(
                    symptoms
                )

            else:

                st.write(
                    "No additional observation provided."
                )

            st.subheader(
                "🔬 What Should Be Checked?"
            )

            for item in recommendation["checks"]:

                st.write(
                    "•",
                    item
                )

            st.subheader(
                "🌾 Immediate Low-Risk Actions"
            )

            for item in recommendation["actions"]:

                st.write(
                    "•",
                    item
                )

            st.subheader(
                "🛡️ Prevention"
            )

            prevention = [

                "Monitor the crop regularly.",

                "Maintain good field sanitation.",

                "Avoid unnecessary leaf wetness.",

                "Maintain appropriate plant spacing and airflow.",

                "Monitor nearby plants for similar symptoms."

            ]

            for item in prevention:

                st.write(
                    "•",
                    item
                )

            st.subheader(
                "⚠️ AI Uncertainty"
            )

            st.warning(
                "The AI identifies visual patterns. "
                "Similar symptoms can be caused by diseases, "
                "pests, nutrition problems or environmental stress."
            )

            st.error(
                "👨‍🌾 Human verification is recommended before "
                "major agricultural treatment decisions."
            )


# ============================================================
# TAB 2 — PROMPT ENGINEERING
# ============================================================

with tab2:

    st.header(
        "🧪 Prompt Engineering Challenge"
    )

    scenario = st.text_area(
        "Agricultural Scenario",

        value=(
            "A tomato farmer observes brown circular spots "
            "and yellowing on older leaves."
        ),

        height=100
    )

    prompt_type = st.selectbox(
        "Prompt Strategy",

        [
            "Zero-shot",
            "Role Prompting",
            "Few-shot Prompting",
            "Structured Prompting",
            "Refined Prompt"
        ]
    )

    if prompt_type == "Zero-shot":

        prompt = f"""
What disease might affect this tomato plant?

Scenario:
{scenario}
"""

        explanation = (
            "Minimal instructions. The model must determine "
            "the task and response style."
        )

    elif prompt_type == "Role Prompting":

        prompt = f"""
Act as an agricultural extension officer.

Analyze this tomato crop problem using
simple language suitable for a farmer.

Scenario:
{scenario}
"""

        explanation = (
            "The role establishes expertise and communication style."
        )

    elif prompt_type == "Few-shot Prompting":

        prompt = f"""
You are an agricultural assistant.

Example 1:
Observation: Yellow leaves without distinct spots.
Possible issue: Nutrient deficiency.

Example 2:
Observation: Brown circular lesions with ring patterns.
Possible issue: Early blight.

Now analyze:

{scenario}

Possible issue:
Evidence:
"""

        explanation = (
            "Examples demonstrate the desired input-output relationship."
        )

    elif prompt_type == "Structured Prompting":

        prompt = f"""
ROLE:
Agricultural extension advisor

CONTEXT:
{scenario}

TASK:
Analyze the possible crop health issue.

CONSTRAINTS:
- Do not claim certainty.
- Use farmer-friendly language.
- Mention uncertainty.
- Avoid unsafe treatment instructions.

OUTPUT:
1. Possible problem
2. Evidence
3. Alternative causes
4. What to check
5. Low-risk actions
6. Prevention
"""

        explanation = (
            "Role, context, constraints and output format are explicitly defined."
        )

    else:

        prompt = f"""
ROLE:
You are a cautious agricultural decision-support assistant.

CONTEXT:
{scenario}

TASK:
Analyze the crop symptoms.

REQUIREMENTS:
- Identify the most likely explanation.
- Give at least one alternative explanation.
- Explain the visual evidence.
- Separate observations from inferences.
- Do not claim a confirmed diagnosis.
- Give only low-risk immediate actions.
- Recommend expert verification when appropriate.

OUTPUT FORMAT:

POSSIBLE PROBLEM:
EVIDENCE:
ALTERNATIVE EXPLANATION:
WHAT TO CHECK:
LOW-RISK ACTIONS:
PREVENTION:
UNCERTAINTY:
"""

        explanation = (
            "The refined prompt adds constraints, alternatives, "
            "evidence and uncertainty handling."
        )

    st.subheader(
        "📋 Prompt"
    )

    st.code(
        prompt,
        language="text"
    )

    st.info(
        "Why this strategy changes the output: "
        + explanation
    )


# ============================================================
# TAB 3 — EVALUATION
# ============================================================

with tab3:

    st.header(
        "📊 Prompt Strategy Evaluation"
    )

    evaluation = pd.DataFrame({

        "Criteria": [
            "Context understanding",
            "Output structure",
            "Farmer friendliness",
            "Uncertainty handling",
            "Controllability",
            "Practical usefulness"
        ],

        "Zero-shot": [
            2, 1, 2, 1, 2, 2
        ],

        "Role Prompt": [
            3, 2, 4, 2, 3, 3
        ],

        "Few-shot": [
            4, 3, 4, 3, 4, 4
        ],

        "Structured": [
            5, 5, 5, 4, 5, 5
        ],

        "Refined": [
            5, 5, 5, 5, 5, 5
        ]
    })

    st.dataframe(
        evaluation,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        evaluation.set_index(
            "Criteria"
        )
    )

    st.caption(
        "These scores illustrate prompt-engineering principles "
        "for the academic demonstration and are not commercial "
        "AI benchmark scores."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌱 AI Farmer Assistant | GenAI in Agriculture | Academic Prototype"
)