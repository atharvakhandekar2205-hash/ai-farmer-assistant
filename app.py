import streamlit as st
import torch
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Farmer Assistant",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "mesabo/agri-plant-disease-resnet50"


@st.cache_resource
def load_model():

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    return processor, model, device


processor, model, device = load_model()


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0

if "last_crop" not in st.session_state:
    st.session_state.last_crop = ""

if "last_description" not in st.session_state:
    st.session_state.last_description = ""

if "last_growth_stage" not in st.session_state:
    st.session_state.last_growth_stage = ""

if "last_location" not in st.session_state:
    st.session_state.last_location = ""


# ============================================================
# MODEL LABELS
# ============================================================

MODEL_LABELS = [
    str(label)
    for label in model.config.id2label.values()
]


# ============================================================
# CROP DISPLAY NAMES
# ============================================================

CROP_NAMES = {
    "Apple": "🍎 Apple",
    "Blueberry": "🫐 Blueberry",
    "Cherry_(including_sour)": "🍒 Cherry",
    "Corn_(maize)": "🌽 Corn",
    "Grape": "🍇 Grape",
    "Orange": "🍊 Orange",
    "Peach": "🍑 Peach",
    "Pepper,_bell": "🫑 Bell Pepper",
    "Potato": "🥔 Potato",
    "Raspberry": "🫐 Raspberry",
    "Soybean": "🌱 Soybean",
    "Squash": "🎃 Squash",
    "Strawberry": "🍓 Strawberry",
    "Tomato": "🍅 Tomato",
}


# ============================================================
# GET CROP FROM LABEL
# ============================================================

def get_crop_from_label(label):

    if "___" not in label:
        return None

    return label.split("___")[0]


# ============================================================
# PRETTY LABEL
# ============================================================

def pretty_label(label):

    if "___" in label:
        crop, disease = label.split("___", 1)
    else:
        crop = ""
        disease = label

    crop_display = {
        "Apple": "Apple",
        "Blueberry": "Blueberry",
        "Cherry_(including_sour)": "Cherry",
        "Corn_(maize)": "Corn",
        "Grape": "Grape",
        "Orange": "Orange",
        "Peach": "Peach",
        "Pepper,_bell": "Bell Pepper",
        "Potato": "Potato",
        "Raspberry": "Raspberry",
        "Soybean": "Soybean",
        "Squash": "Squash",
        "Strawberry": "Strawberry",
        "Tomato": "Tomato",
    }

    disease = disease.replace("_", " ")
    disease = disease.replace("  ", " ")

    disease = disease.replace(
        "Spider mites Two-spotted spider mite",
        "Spider Mites"
    )

    disease = disease.replace(
        "Haunglongbing (Citrus greening)",
        "Citrus Greening"
    )

    disease = disease.replace(
        "Cercospora leaf spot Gray leaf spot",
        "Cercospora Leaf Spot / Gray Leaf Spot"
    )

    if disease.lower() == "healthy":
        disease = "Healthy"
    else:
        disease = disease.title()

    return f"{crop_display.get(crop, crop)} — {disease}"


# ============================================================
# AVAILABLE CROPS
# ============================================================

available_crops = []

for label in MODEL_LABELS:

    crop = get_crop_from_label(label)

    if crop and crop not in available_crops:
        available_crops.append(crop)


# ============================================================
# CROP FILTER
# ============================================================

def get_crop_indices(selected_crop):

    indices = []

    for index, label in model.config.id2label.items():

        label = str(label)

        crop = get_crop_from_label(label)

        if crop == selected_crop:
            indices.append(int(index))

    return indices


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(image, selected_crop):

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

        logits = outputs.logits[0]

    allowed_indices = get_crop_indices(
        selected_crop
    )

    if not allowed_indices:
        raise ValueError(
            f"No classes available for {selected_crop}"
        )

    filtered_logits = torch.full_like(
        logits,
        float("-inf")
    )

    filtered_logits[allowed_indices] = logits[allowed_indices]

    probabilities = torch.softmax(
        filtered_logits,
        dim=0
    )

    number_of_results = min(
        5,
        len(allowed_indices)
    )

    top_values, top_indices = torch.topk(
        probabilities,
        k=number_of_results
    )

    results = []

    for probability, index in zip(
        top_values.tolist(),
        top_indices.tolist()
    ):

        label = model.config.id2label[int(index)]

        results.append({
            "label": label,
            "confidence": probability * 100
        })

    return results


# ============================================================
# CROP HEALTH REPORT
# ============================================================

def generate_report(
    crop,
    prediction,
    confidence,
    description,
    location,
    growth_stage
):

    disease_name = pretty_label(prediction)

    if "healthy" in prediction.lower():

        status = "Healthy / No major disease pattern detected"

        advice = (
            "Continue regular monitoring. Maintain appropriate "
            "irrigation, nutrition, field hygiene and pest monitoring."
        )

    elif "Early_blight" in prediction:

        status = "Possible Early Blight"

        advice = (
            "Inspect nearby leaves for similar lesions. Remove "
            "severely affected plant material where appropriate "
            "and avoid prolonged leaf wetness. Consult a qualified "
            "agricultural professional before applying treatment."
        )

    elif "Late_blight" in prediction:

        status = "Possible Late Blight"

        advice = (
            "Inspect the crop carefully for rapidly expanding "
            "dark lesions and additional symptoms. Because late "
            "blight can spread quickly, professional agricultural "
            "advice is recommended."
        )

    elif "Tomato_Yellow_Leaf_Curl_Virus" in prediction:

        status = "Possible Tomato Yellow Leaf Curl Virus"

        advice = (
            "Inspect new growth and nearby plants for similar "
            "symptoms and check for possible whitefly activity. "
            "Confirm the diagnosis with an agricultural expert."
        )

    elif "Powdery_mildew" in prediction:

        status = "Possible Powdery Mildew"

        advice = (
            "Inspect additional leaves for white powdery growth. "
            "Improve airflow around plants where possible and "
            "consult an agricultural professional before treatment."
        )

    else:

        status = f"Possible {disease_name}"

        advice = (
            "Inspect the affected area carefully and compare "
            "symptoms with trusted agricultural guidance. Expert "
            "verification is recommended before making treatment decisions."
        )

    st.markdown("---")

    st.subheader("📋 Crop Health Report")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Crop",
            CROP_NAMES.get(crop, crop)
        )

    with col2:
        st.metric(
            "Growth Stage",
            growth_stage
        )

    with col3:
        st.metric(
            "AI Confidence",
            f"{confidence:.2f}%"
        )

    st.markdown("### 🔎 AI Assessment")

    st.info(
        f"**Prediction:** {status}\n\n"
        f"**Model output:** {disease_name}\n\n"
        f"**Location:** {location if location else 'Not provided'}"
    )

    st.markdown("### 👨‍🌾 Farmer Observation")

    st.write(
        description if description
        else "No additional observation provided."
    )

    st.markdown("### 🌱 Recommended Next Steps")

    st.write(advice)

    st.warning(
        "⚠️ This is an AI-assisted screening result, not a "
        "professional agricultural diagnosis. Real-world field "
        "conditions can differ from controlled image datasets."
    )


# ============================================================
# PROMPT ENGINEERING
# ============================================================

def create_prompts(
    crop,
    growth_stage,
    description,
    prediction
):

    base_context = (
        f"Crop: {crop}\n"
        f"Growth stage: {growth_stage}\n"
        f"Farmer observation: {description}\n"
        f"Image model prediction: {prediction}"
    )

    zero_shot = f"""
Identify the possible agricultural condition from the following information.

{base_context}

Provide:
1. Possible condition
2. Evidence
3. Recommended next step
"""

    role_prompt = f"""
You are an agricultural decision-support assistant helping a farmer.

Your job is to interpret AI image-screening results carefully.
Do not claim a definitive diagnosis.

{base_context}

Give simple, practical and safe guidance for the farmer.
"""

    few_shot = f"""
You are analyzing crop-health observations.

Example 1:
Observation: Brown circular spots with darker centers on older leaves.
Possible condition: Early blight.

Example 2:
Observation: White powder-like growth covering leaf surfaces.
Possible condition: Powdery mildew.

Now analyze this case:

{base_context}

Use the examples as guidance, but do not assume the examples are
the diagnosis. Explain your reasoning and recommended next step.
"""

    structured_prompt = f"""
Analyze this agricultural case using the exact structure below.

INPUT
Crop: {crop}
Growth Stage: {growth_stage}
Observation: {description}
AI Image Prediction: {prediction}

OUTPUT FORMAT
Condition:
Evidence:
Confidence interpretation:
Recommended action:
Additional information needed:
Safety warning:

Do not present the prediction as a confirmed diagnosis.
"""

    refined_prompt = f"""
You are an agricultural AI decision-support assistant.

Analyze the following case:

{base_context}

IMPORTANT RULES:
- Use the image-model prediction as evidence, not absolute truth.
- Consider the farmer's written observation.
- Consider the crop and growth stage.
- Clearly separate observation from inference.
- Do not recommend a chemical treatment without professional verification.
- Mention what additional information would improve confidence.
- Give practical next steps in simple farmer-friendly language.
- End with a human-expert verification recommendation.

Return the answer under these headings:

1. Most likely interpretation
2. Evidence from image/model
3. Evidence from farmer observation
4. Recommended next steps
5. Additional checks
6. Safety and expert verification
"""

    return {
        "Zero-Shot": zero_shot.strip(),
        "Role Prompting": role_prompt.strip(),
        "Few-Shot": few_shot.strip(),
        "Structured Prompt": structured_prompt.strip(),
        "Refined Prompt": refined_prompt.strip()
    }


# ============================================================
# PROMPT ENGINEERING TAB
# ============================================================

def prompt_engineering_section():

    st.header("🧠 Prompt Engineering Lab")

    st.write(
        "This section demonstrates how prompt design changes the "
        "quality, structure and safety of an agricultural AI response."
    )

    crop = st.session_state.last_crop or "Tomato"
    growth_stage = (
        st.session_state.last_growth_stage
        or "Flowering"
    )

    description = (
        st.session_state.last_description
        or "Brown spots are appearing on the leaves."
    )

    prediction = (
        st.session_state.last_prediction
        or "Tomato___Early_blight"
    )

    prompts = create_prompts(
        crop,
        growth_stage,
        description,
        pretty_label(prediction)
    )

    st.info(
        f"**Current case:** {CROP_NAMES.get(crop, crop)} | "
        f"{growth_stage} | "
        f"{pretty_label(prediction)}"
    )

    prompt_type = st.selectbox(
        "Select Prompt Strategy",
        list(prompts.keys())
    )

    st.markdown("### 📝 Prompt")

    st.code(
        prompts[prompt_type],
        language="text"
    )

    st.markdown("### 💡 Why this prompt is useful")

    explanations = {
        "Zero-Shot":
            "No examples are provided. The model must solve the task directly from the instruction.",

        "Role Prompting":
            "Giving the AI an agricultural-assistant role establishes the desired perspective, tone and safety boundaries.",

        "Few-Shot":
            "Examples demonstrate the expected relationship between symptoms and possible conditions.",

        "Structured Prompt":
            "The required output format reduces vague responses and makes results easier to compare.",

        "Refined Prompt":
            "The prompt combines role, context, constraints, evidence separation, safety and a specific output structure."
    }

    st.success(
        explanations[prompt_type]
    )

    st.markdown("---")

    st.subheader("🔬 Prompt Strategy Comparison")

    comparison = pd.DataFrame({
        "Strategy": [
            "Zero-Shot",
            "Role Prompting",
            "Few-Shot",
            "Structured Prompt",
            "Refined Prompt"
        ],
        "Context": [
            "Low",
            "Medium",
            "Medium",
            "High",
            "Very High"
        ],
        "Output Structure": [
            "Low",
            "Medium",
            "Medium",
            "High",
            "Very High"
        ],
        "Safety Guidance": [
            "Low",
            "Medium",
            "Medium",
            "High",
            "Very High"
        ],
        "Best Use": [
            "Quick questions",
            "Domain assistance",
            "Pattern examples",
            "Consistent reports",
            "Final decision support"
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("🎯 Prompt Refinement Challenge")

    st.write(
        "Start with a basic prompt and progressively add role, "
        "context, constraints and output structure."
    )

    refinement_steps = pd.DataFrame({
        "Step": [
            "1. Basic",
            "2. Add Role",
            "3. Add Context",
            "4. Add Structure",
            "5. Add Safety Constraints"
        ],
        "Improvement": [
            "Defines the task",
            "Adds domain expertise",
            "Adds crop and symptom context",
            "Makes output consistent",
            "Reduces unsafe overconfidence"
        ]
    })

    st.table(refinement_steps)


# ============================================================
# EVALUATION TAB
# ============================================================

def evaluation_section():

    st.header("📊 AI Evaluation Dashboard")

    st.write(
        "The following rubric evaluates the quality of the AI "
        "decision-support response. These are evaluation scores "
        "for the prototype, not medical/agricultural accuracy claims."
    )

    evaluation_data = pd.DataFrame({
        "Criterion": [
            "Relevance",
            "Specificity",
            "Completeness",
            "Safety",
            "Consistency"
        ],
        "Weight": [
            20,
            20,
            20,
            25,
            15
        ],
        "Score / 100": [
            85,
            82,
            88,
            95,
            90
        ]
    })

    evaluation_data["Weighted Score"] = (
        evaluation_data["Weight"]
        * evaluation_data["Score / 100"]
        / 100
    )

    st.dataframe(
        evaluation_data,
        use_container_width=True,
        hide_index=True
    )

    overall = evaluation_data["Weighted Score"].sum()

    st.metric(
        "Overall Prototype Evaluation Score",
        f"{overall:.1f} / 100"
    )

    st.progress(
        int(overall)
    )

    st.markdown("### 📌 Evaluation Criteria")

    st.markdown("""
**Relevance**  
Does the response address the crop and farmer's actual observation?

**Specificity**  
Does the response provide crop-specific rather than generic guidance?

**Completeness**  
Does it include interpretation, evidence and practical next steps?

**Safety**  
Does the AI avoid presenting an uncertain result as a confirmed diagnosis?

**Consistency**  
Does prompt refinement produce clearer and more structured responses?
""")

    st.markdown("---")

    st.subheader("⚖️ Important Evaluation Limitation")

    st.warning(
        "A model confidence score is NOT the same as real-world accuracy. "
        "A proper agricultural evaluation would require a labelled test "
        "dataset containing representative field images."
    )


# ============================================================
# RESPONSIBLE AI
# ============================================================

def responsible_ai_section():

    st.header("⚠️ Responsible AI in Agriculture")

    issues = {
        "🤖 Hallucinations":
            "AI may generate plausible but incorrect agricultural advice. "
            "The system therefore treats image predictions as screening results.",

        "🌾 Bias":
            "Training images may not represent every crop variety, region, "
            "lighting condition or farming environment.",

        "🔐 Privacy":
            "Farmer names, locations and uploaded images should be handled "
            "carefully and should not be exposed unnecessarily.",

        "👨‍🌾 Human Oversight":
            "Farmers should consult qualified agricultural professionals "
            "before making important treatment decisions.",

        "📷 Image Limitations":
            "Poor lighting, damaged leaves, multiple diseases or unrelated "
            "images can reduce model reliability.",

        "🧪 Dataset Limitation":
            "The model's training data is based on plant-disease imagery and "
            "may not perfectly represent real field conditions."
    }

    for title, explanation in issues.items():

        with st.expander(title):

            st.write(explanation)

    st.markdown("---")

    st.subheader("✅ Responsible AI Workflow")

    st.markdown("""
**Farmer Image**
↓  
**AI Image Screening**
↓  
**Prompt-Based Interpretation**
↓  
**Confidence + Limitations**
↓  
**Human Verification**
↓  
**Final Agricultural Decision**
""")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("👨‍🌾 Farmer Information")

    farmer_name = st.text_input(
        "Farmer Name",
        placeholder="Enter farmer name"
    )

    location = st.text_input(
        "Location",
        placeholder="Example: Kalyan, Maharashtra"
    )

    st.markdown("---")

    st.info(
        "🌱 This system provides AI-assisted crop-health "
        "information and does not replace a qualified "
        "agricultural expert."
    )

    st.markdown("---")

    st.write(
        f"**AI Device:** {device}"
    )


# ============================================================
# HEADER
# ============================================================

st.title("🌱 AI Farmer Assistant")

st.subheader(
    "GenAI-Powered Agricultural Decision Support System"
)

st.write(
    "An AI-assisted agricultural prototype that analyzes plant "
    "leaf images and demonstrates prompt engineering, evaluation "
    "and responsible AI practices."
)


# ============================================================
# MAIN TABS
# ============================================================

tab_analysis, tab_prompt, tab_evaluation, tab_ethics = st.tabs(
    [
        "📷 Crop Analysis",
        "🧠 Prompt Engineering",
        "📊 Evaluation",
        "⚠️ Responsible AI"
    ]
)


# ============================================================
# TAB 1 — CROP ANALYSIS
# ============================================================

with tab_analysis:

    st.header("📷 Crop Analysis")

    col1, col2 = st.columns(2)

    with col1:

        crop_options = [
            crop
            for crop in available_crops
        ]

        selected_crop = st.selectbox(
            "Select Crop",
            crop_options,
            format_func=lambda x:
                CROP_NAMES.get(x, x)
        )

        uploaded_file = st.file_uploader(
            "Upload Crop / Leaf Image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )

    with col2:

        description = st.text_area(
            "Describe what you observe",

            placeholder=(
                "Example:\n"
                "Brown spots are appearing on the leaves "
                "and some leaves are turning yellow."
            ),

            height=150
        )

        growth_stage = st.selectbox(
            "Crop Growth Stage",
            [
                "Seedling",
                "Vegetative",
                "Flowering",
                "Fruiting",
                "Maturity"
            ]
        )

    image = None

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.markdown("---")

        st.subheader("📷 Uploaded Image")

        st.image(
            image,
            width=600
        )

    st.markdown("---")

    if st.button(
        "🔍 Analyze Crop",
        type="primary",
        use_container_width=True
    ):

        if image is None:

            st.error(
                "Please upload a crop/leaf image first."
            )

        else:

            with st.spinner(
                f"Analyzing {CROP_NAMES.get(selected_crop, selected_crop)}..."
            ):

                try:

                    results = analyze_image(
                        image,
                        selected_crop
                    )

                    top_prediction = results[0]

                    prediction = top_prediction["label"]

                    confidence = top_prediction["confidence"]

                    # SAVE ANALYSIS
                    st.session_state.analysis_done = True
                    st.session_state.last_prediction = prediction
                    st.session_state.last_confidence = confidence
                    st.session_state.last_crop = selected_crop
                    st.session_state.last_description = description
                    st.session_state.last_growth_stage = growth_stage
                    st.session_state.last_location = location

                    st.markdown("---")

                    st.header("🧠 AI Analysis")

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Top Prediction",
                            pretty_label(prediction)
                        )

                    with col2:

                        st.metric(
                            "Confidence",
                            f"{confidence:.2f}%"
                        )

                    with col3:

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

                    if confidence >= 70:

                        st.success(
                            "The model produced a stronger crop-specific "
                            "prediction. Expert verification is still recommended."
                        )

                    elif confidence >= 40:

                        st.warning(
                            "The model has moderate confidence. "
                            "Additional verification is recommended."
                        )

                    else:

                        st.error(
                            "The model has low confidence. "
                            "Upload a clearer image and seek expert verification."
                        )

                    st.subheader(
                        "📊 Top Crop-Specific Predictions"
                    )

                    table_data = []

                    for result in results:

                        table_data.append({

                            "Prediction":
                                pretty_label(
                                    result["label"]
                                ),

                            "Confidence (%)":
                                round(
                                    result["confidence"],
                                    2
                                )
                        })

                    df = pd.DataFrame(
                        table_data
                    )

                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )

                    generate_report(
                        crop=selected_crop,
                        prediction=prediction,
                        confidence=confidence,
                        description=description,
                        location=location,
                        growth_stage=growth_stage
                    )

                    st.success(
                        "Analysis saved for the Prompt Engineering and Evaluation tabs."
                    )

                except Exception as e:

                    st.error(
                        f"Analysis error: {e}"
                    )

    # SHOW LAST ANALYSIS WHEN RETURNING TO TAB

    if (
        st.session_state.analysis_done
        and st.session_state.last_prediction
        and not uploaded_file
    ):

        st.info(
            f"Latest analysis: "
            f"{pretty_label(st.session_state.last_prediction)} "
            f"({st.session_state.last_confidence:.2f}%)"
        )


# ============================================================
# TAB 2 — PROMPT ENGINEERING
# ============================================================

with tab_prompt:

    prompt_engineering_section()


# ============================================================
# TAB 3 — EVALUATION
# ============================================================

with tab_evaluation:

    evaluation_section()


# ============================================================
# TAB 4 — RESPONSIBLE AI
# ============================================================

with tab_ethics:

    responsible_ai_section()


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🌱 AI Farmer Assistant | GenAI in Agriculture | Academic Prototype"
)