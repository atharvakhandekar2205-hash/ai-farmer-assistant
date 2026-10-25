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
# MODEL CONFIGURATION
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
# CROP CLASS FILTER
# ============================================================

CROP_CLASSES = {

    "Tomato": [
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites_Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",
    ],

    "Potato": [
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
    ],
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def pretty_label(label):

    label = label.replace("___", " — ")
    label = label.replace("_", " ")

    label = label.replace(
        "Tomato Yellow Leaf Curl Virus",
        "Tomato Yellow Leaf Curl Virus"
    )

    label = label.replace(
        "Spider mites Two spotted spider mite",
        "Spider Mites"
    )

    return label.title()


def get_crop_indices(crop):

    allowed = CROP_CLASSES[crop]

    indices = []

    for index, label in model.config.id2label.items():

        label_clean = str(label).strip()

        if label_clean in allowed:
            indices.append(int(index))

    return indices


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

        logits = outputs.logits[0].clone()

    # --------------------------------------------------------
    # IMPORTANT:
    # Keep ONLY classes belonging to selected crop.
    # --------------------------------------------------------

    allowed_indices = get_crop_indices(selected_crop)

    if not allowed_indices:
        raise ValueError(
            f"No model classes found for {selected_crop}."
        )

    filtered_logits = torch.full_like(
        logits,
        float("-inf")
    )

    filtered_logits[allowed_indices] = logits[
        allowed_indices
    ]

    probabilities = torch.softmax(
        filtered_logits,
        dim=0
    )

    top_values, top_indices = torch.topk(
        probabilities,
        k=min(5, len(allowed_indices))
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


def generate_report(
    crop,
    prediction,
    confidence,
    description,
    location,
    growth_stage
):

    disease = pretty_label(prediction)

    if "healthy" in prediction.lower():

        status = "Healthy / No major disease pattern detected"

        advice = (
            "Continue monitoring the crop regularly. "
            "Maintain appropriate irrigation, nutrition, "
            "field hygiene and pest monitoring."
        )

    elif "early_blight" in prediction.lower():

        status = "Possible Early Blight"

        advice = (
            "Inspect nearby leaves for similar lesions. "
            "Remove severely affected plant material where "
            "appropriate and avoid prolonged leaf wetness. "
            "For treatment decisions, consult a qualified "
            "agricultural professional."
        )

    elif "late_blight" in prediction.lower():

        status = "Possible Late Blight"

        advice = (
            "Inspect the crop carefully for rapidly expanding "
            "dark lesions and additional symptoms. "
            "Because late blight can spread quickly, obtain "
            "professional agricultural advice before treatment."
        )

    elif "yellow_leaf_curl" in prediction.lower():

        status = "Possible Tomato Yellow Leaf Curl Virus"

        advice = (
            "Inspect new growth and nearby plants for similar "
            "symptoms and check for whitefly activity. "
            "Confirm the diagnosis with an agricultural expert "
            "before taking disease-control decisions."
        )

    else:

        status = f"Possible {disease}"

        advice = (
            "Inspect the affected area carefully and compare "
            "symptoms with trusted agricultural guidance. "
            "Expert verification is recommended before treatment."
        )

    st.markdown("---")

    st.subheader("📋 Crop Health Report")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Crop", crop)

    with col2:
        st.metric("Growth Stage", growth_stage)

    with col3:
        st.metric("AI Confidence", f"{confidence:.2f}%")

    st.markdown("### 🔎 AI Assessment")

    st.info(
        f"**Prediction:** {status}\n\n"
        f"**Model output:** {disease}\n\n"
        f"**Location:** {location if location else 'Not provided'}"
    )

    st.markdown("### 👨‍🌾 Farmer Observation")

    if description:
        st.write(description)
    else:
        st.write("No additional observation provided.")

    st.markdown("### 🌱 Recommended Next Steps")

    st.write(advice)

    st.warning(
        "⚠️ This is an AI-assisted screening result, "
        "not a professional agricultural diagnosis. "
        "The model was trained on PlantVillage-style leaf "
        "images and may perform differently on real field "
        "conditions."
    )


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
        "🌱 This system provides AI-assisted "
        "crop-health information and does not "
        "replace a qualified agricultural expert."
    )

    st.markdown("---")

    st.write(
        f"**AI Device:** {device}"
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.title("🌱 AI Farmer Assistant")

st.subheader(
    "GenAI-Powered Agricultural Decision Support System"
)

st.write(
    "An academic AI prototype that analyzes crop leaf images "
    "and provides crop-specific disease screening."
)


st.markdown("---")


# ============================================================
# CROP ANALYSIS
# ============================================================

st.header("📷 Crop Analysis")

col1, col2 = st.columns(2)


with col1:

    crop = st.selectbox(
        "Select Crop",
        [
            "Tomato",
            "Potato"
        ]
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


# ============================================================
# IMAGE PREVIEW
# ============================================================

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


# ============================================================
# ANALYZE BUTTON
# ============================================================

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
            f"Analyzing {crop} leaf..."
        ):

            try:

                results = analyze_image(
                    image,
                    crop
                )

                top_prediction = results[0]

                prediction = top_prediction["label"]

                confidence = top_prediction[
                    "confidence"
                ]

                # ------------------------------------------------
                # AI ANALYSIS
                # ------------------------------------------------

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
                        "The model produced a stronger "
                        "crop-specific prediction. "
                        "Expert verification is still recommended."
                    )

                elif confidence >= 40:

                    st.warning(
                        "The model has moderate confidence. "
                        "Additional verification is recommended."
                    )

                else:

                    st.error(
                        "The model has low confidence. "
                        "Upload a clearer leaf image and "
                        "seek expert verification."
                    )


                # ------------------------------------------------
                # TOP 5
                # ------------------------------------------------

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


                # ------------------------------------------------
                # REPORT
                # ------------------------------------------------

                generate_report(
                    crop=crop,
                    prediction=prediction,
                    confidence=confidence,
                    description=description,
                    location=location,
                    growth_stage=growth_stage
                )


            except Exception as e:

                st.error(
                    f"Analysis error: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🌱 AI Farmer Assistant | "
    "GenAI in Agriculture | Academic Prototype"
)