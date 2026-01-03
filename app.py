import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
from PIL import Image
import matplotlib.cm as cm
import datetime

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Nutrivision",
    page_icon="🍽",
    layout="wide"
)

if "user" not in st.session_state:
    st.session_state.user = None

with st.sidebar:
    st.subheader("👤 User Login")

    if st.session_state.user is None:
        username = st.text_input("Enter username")
        if st.button("Login"):
            if username.strip():
                st.session_state.user = username.strip()
                st.session_state.diary = []
                st.success(f"Welcome, {username}!")
                st.rerun()
    else:
        st.success(f"Logged in as **{st.session_state.user}**")
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

if st.session_state.user is None:
    st.info("👈 Please login from the sidebar to start using Nutrivision.")
    st.stop()

# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
    <div style="text-align:center; padding: 1.5rem 0;">
        <h1>🍽 Nutrivision</h1>
        <p style="font-size:1.1rem; color:gray;">
            Snap your meal. Understand your nutrition. Track smarter.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_food_model():
    return load_model("mobilenetv2_food11.keras")

model = load_food_model()

food_classes = [
    "apple_pie","cheesecake","chicken_curry","french_fries","fried_rice",
    "hamburger","hot_dog","ice_cream","omelette","pizza","sushi"
]

# ----------------------------
# Nutrition API
# ----------------------------
API_KEY = "+0EWFO4JUAufJ3ihUVkhuA==atMZbd7lzO5uLWCr"   # Add your new API key here

def get_nutrition(food_name, quantity):
    url = f"https://api.calorieninjas.com/v1/nutrition?query={quantity}g {food_name}"
    response = requests.get(url, headers={"X-Api-Key": API_KEY})
    if response.status_code == 200:
        data = response.json()
        if data["items"]:
            f = data["items"][0]
            return {
                "calories": f.get("calories", 0),
                "protein": f.get("protein_g", 0),
                "fat": f.get("fat_total_g", 0),
                "carbs": f.get("carbohydrates_total_g", 0),
                "sugar": f.get("sugar_g", 0),
                "fiber": f.get("fiber_g", 0),
                "sodium": f.get("sodium_mg", 0),
                "cholesterol": f.get("cholesterol_mg", 0),
            }
    return None

# ----------------------------
# Image Input
# ----------------------------
st.subheader("📸 Add Food Image")

img = None
input_type = st.radio("Image source", ["Upload Image", "Use Webcam"], horizontal=True)

if input_type == "Upload Image":
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
else:
    cam = st.camera_input("Take a picture")
    if cam:
        img = Image.open(cam).convert("RGB")

# ----------------------------
# Prediction
# ----------------------------
if img:
    st.image(img, width=280)

    img_resized = img.resize((224,224))
    x = image.img_to_array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model(x, training=False).numpy()
    class_idx = np.argmax(preds[0])
    food_name = food_classes[class_idx]
    confidence = preds[0][class_idx] * 100
    st.progress(int(confidence))

    qty = st.slider("Portion size (grams)", 50, 1000, 200, step=50)
    nutrition = get_nutrition(food_name.replace("_"," "), qty)

    st.divider()
    st.subheader("✨ Prediction Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Food", food_name.replace("_"," ").title())
    c2.metric("Confidence", f"{confidence:.1f}%")
    c3.metric("Calories", f"{nutrition['calories']} kcal" if nutrition else "N/A")

    # ----------------------------
    # Nutrition Details
    # ----------------------------
    if nutrition:
        with st.expander("🥗 Nutrition Breakdown"):
            st.write(f"**Protein:** {nutrition['protein']} g")
            st.write(f"**Fat:** {nutrition['fat']} g")
            st.write(f"**Carbs:** {nutrition['carbs']} g")
            st.write(f"**Sugar:** {nutrition['sugar']} g")
            st.write(f"**Fiber:** {nutrition['fiber']} g")
            st.write(f"**Sodium:** {nutrition['sodium']} mg")
            st.write(f"**Cholesterol:** {nutrition['cholesterol']} mg")
                
    # ----------------------------
    # Diary (SAFE & CLEAN)
    # ----------------------------
    if "diary" not in st.session_state:
        st.session_state.diary = []

    def health_score(n):
        calories = n.get("calories", 0)
        protein = n.get("protein", 0)
        fat = n.get("fat", 0)
        carbs = n.get("carbs", 0)
        sugar = n.get("sugar", 0)
        fiber = n.get("fiber", 0)

        score = 10

        # Penalties
        if calories > 600: score -= 2
        if fat > 20: score -= 2
        if sugar > 10: score -= 2

        # Rewards
        if protein >= 15: score += 1
        if fiber >= 5: score += 1

        # Carb-heavy penalty
        if carbs > 50 and fiber < 5:
            score -= 1

        return max(1, min(score, 10))


    current_score = health_score(nutrition)

    # ----------------------------
    # Add to diary button
    # ----------------------------
    if st.button("➕ Add Meal to Diary"):
        st.session_state.diary.append({
            "date": datetime.date.today(),
            "food": food_name.replace("_", " ").title(),
            "portion": qty,
            "calories": nutrition["calories"],
            "score": current_score
        })
        st.success("Meal added to diary!")

    st.divider()
    st.subheader("📝 Food Diary")

    # ----------------------------
    # Display diary
    # ----------------------------
    if not st.session_state.diary:
        st.info("No meals logged yet.")
    else:
        for m in st.session_state.diary:
            badge = "🟢 Good" if m["score"] >= 7 else "🟡 Average" if m["score"] >= 5 else "🔴 Poor"
            st.info(
                f"**{m['food']}** ({m['portion']} g)\n\n"
                f"🔥 {m['calories']} kcal | 💪 Health Score: {m['score']}/10 {badge}"
            )

    # ----------------------------
    # Grad-CAM
    # ----------------------------
    st.subheader("🔍 Model Attention (Grad-CAM)")

    last_conv = "block_13_expand"

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(conv_out[0] * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap.numpy()

# ----------------------------
# Footer
# ----------------------------

st.divider()

st.markdown(
    f"""
    <div style="text-align:center; color: gray; font-size: 0.85rem; padding: 1rem 0;">
        © {datetime.datetime.now().year} Nutrivision | All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)

