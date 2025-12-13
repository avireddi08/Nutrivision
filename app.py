import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image 
import numpy as np
import base64
import requests
from PIL import Image
import streamlit.components.v1 as components
import matplotlib.cm as cm  # for applying colormap
import datetime

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Nutrivision - Food Classifier & Nutrition Tracker",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ----------------------------
# App Header
# ----------------------------
st.markdown(
    """
    <div style='text-align:center; padding:20px;'>
        <h1 style='font-size:50px; color:white; margin-bottom:5px;'>🍽 Nutrivision</h1>
        <h3 style='font-size:20px; color:white; margin-top:0;'>See your meal. Know your nutrients.</h3>
        <p style='font-size:16px; color:white; max-width:700px; margin:auto;'>
            Snap a picture of your food or upload an image, and Nutrivision will analyze it,
            provide detailed nutrition info, and track calories in real-time.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Background image with dark overlay
# ----------------------------
def set_bg_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stApp .main {{
            background-color: rgba(255,255,255,0.85);
            border-radius: 15px;
            padding: 30px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your background image
set_bg_local("bgimg.png")

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_food_model():
    return load_model("mobilenetv2_food11.keras")

model = load_food_model()

food_classes = ["apple_pie","cheesecake","chicken_curry","french_fries","fried_rice",
                "hamburger","hot_dog","ice_cream","omelette","pizza","sushi"]

st.markdown("---")

# ----------------------------
# Nutrition API (CalorieNinjas)
# ----------------------------
API_KEY = "+0EWFO4JUAufJ3ihUVkhuA==atMZbd7lzO5uLWCr"   # Add your new API key here

def get_nutrition(food_name, quantity=100):
    """Fetch nutrition data for a food item (default 100g)."""
    url = f"https://api.calorieninjas.com/v1/nutrition?query={quantity}g {food_name}"
    
    response = requests.get(url, headers={"X-Api-Key": API_KEY})
    
    if response.status_code == 200:
        data = response.json()
        if "items" in data and len(data["items"]) > 0:
            food = data["items"][0]
            return {
                "calories": food.get("calories", 0),
                "protein": food.get("protein_g", 0),
                "fat": food.get("fat_total_g", 0),
                "carbs": food.get("carbohydrates_total_g", 0),
                "sugar": food.get("sugar_g", "N/A"),
                "fiber": food.get("fiber_g", "N/A"),
                "sodium": food.get("sodium_mg", "N/A"),
                "cholesterol": food.get("cholesterol_mg", "N/A")
            }
    return None

# ----------------------------
# Upload image
# ----------------------------
uploaded_file = st.file_uploader("Choose a food image", type=["jpg","png","jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).resize((224,224))
    st.image(img, caption="Uploaded Image", width=300)

    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    class_idx = np.argmax(preds)
    food_name = food_classes[class_idx]
    confidence = preds[0][class_idx]*100

    # Quantity input
    qty = st.number_input("Enter portion size (grams)", min_value=50, max_value=1000, value=200, step=50)

    # Get nutrition from API
    nutrition = get_nutrition(food_name.replace("_", " "), qty)

    st.markdown("---")
    st.markdown("### ✨ Results")

    # ----------------------------
    # CSS for result cards
    # ----------------------------
    st.markdown("""
    <style>
    .result-card {
        border-radius: 15px;
        padding: 20px;
        color: black;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin-bottom: 10px;
    }
    .result-title { font-size: 16px; margin-bottom: 8px; }
    .result-value { font-size: 20px; font-weight: bold; }
    .pred-card { background: linear-gradient(135deg, #ff9a9e, #fad0c4); }
    .conf-card { background: linear-gradient(135deg, #a1c4fd, #c2e9fb); }
    .cal-card { background: linear-gradient(135deg, #fddb92, #d1fdff); }
    </style>
    """, unsafe_allow_html=True)

    # Main result cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<div class='result-card pred-card'>"
            f"<div class='result-title'>🍽 Food</div>"
            f"<div class='result-value'>{food_name.replace('_',' ').title()}</div>"
            f"</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            f"<div class='result-card conf-card'>"
            f"<div class='result-title'>📊 Confidence</div>"
            f"<div class='result-value'>{confidence:.2f}%</div>"
            f"</div>", unsafe_allow_html=True)
    with col3:
        cal_text = f"{nutrition['calories']} kcal" if nutrition else "N/A"
        st.markdown(
            f"<div class='result-card cal-card'>"
            f"<div class='result-title'>🔥 Calories</div>"
            f"<div class='result-value'>{cal_text}</div>"
            f"</div>", unsafe_allow_html=True)

    # ----------------------------
    # Additional Nutrition Info Card (Beautiful HTML)
    # ----------------------------
    if nutrition:
        nutrient_html = f"""
        <div style='border-radius:15px; padding:20px; background:linear-gradient(135deg,#fdfbfb,#ebedee); 
                    box-shadow:0 4px 10px rgba(0,0,0,0.2); margin-top:20px;'>

            <div style='font-weight:bold; font-size:18px; margin-bottom:15px;'>🥗 Nutrition Details</div>

            <div style='display:flex; justify-content:space-between; padding:10px; border-bottom:1px solid #ccc; 
                        background:#ffefef; border-radius:8px; margin-bottom:5px;'>
                <span>💪 Protein</span><span>{nutrition.get("protein","N/A")} g</span>
            </div>
            <div style='display:flex; justify-content:space-between; padding:10px; border-bottom:1px solid #ccc; 
                        background:#fff3e0; border-radius:8px; margin-bottom:5px;'>
                <span>🥓 Fat</span><span>{nutrition.get("fat","N/A")} g</span>
            </div>
            <div style='display:flex; justify-content:space-between; padding:10px; border-bottom:1px solid #ccc; 
                        background:#e0f7fa; border-radius:8px; margin-bottom:5px;'>
                <span>🍞 Carbohydrates</span><span>{nutrition.get("carbs","N/A")} g</span>
            </div>
            <div style='display:flex; justify-content:space-between; padding:10px; border-bottom:1px solid #ccc; 
                        background:#fffde7; border-radius:8px; margin-bottom:5px;'>
                <span>🍬 Sugar</span><span>{nutrition.get("sugar","N/A")} g</span>
            </div>
            <div style='display:flex; justify-content:space-between; padding:10px; border-bottom:1px solid #ccc; 
                        background:#f1f8e9; border-radius:8px; margin-bottom:5px;'>
                <span>🌾 Fiber</span><span>{nutrition.get("fiber","N/A")} g</span>
            </div>
            <div style='display:flex; justify-content:space-between; padding:10px; border-bottom:1px solid #ccc; 
                        background:#e0f2f1; border-radius:8px; margin-bottom:5px;'>
                <span>🧂 Sodium</span><span>{nutrition.get("sodium","N/A")} mg</span>
            </div>
            <div style='display:flex; justify-content:space-between; padding:10px; 
                        background:#f3e5f5; border-radius:8px;'>
                <span>🥚 Cholesterol</span><span>{nutrition.get("cholesterol","N/A")} mg</span>
            </div>
        </div>
        """
        components.html(nutrient_html, height=450)

    # ----------------------------
    # Initialize session state for diary
    # ----------------------------
    if "diary" not in st.session_state:
        st.session_state.diary = []

    # ----------------------------
    # Function: Calculate health score
    # ----------------------------
    def health_score(nutrition):
        """
        Simple health score from 1-10 based on calories, fat, sugar, and fiber.
        """
        score = 5
        calories = nutrition.get("calories", 0)
        fat = nutrition.get("fat", 0)
        sugar = nutrition.get("sugar", 0)
        fiber = nutrition.get("fiber", 0)

        if calories < 300: score += 1
        if fat < 10: score += 1
        if sugar < 5: score += 1
        if fiber > 5: score += 1

        return min(score, 10)

    # ----------------------------
    # Log current meal
    # ----------------------------
    current_score = health_score(nutrition)

    st.session_state.diary.append({
        "date": datetime.date.today().isoformat(),
        "food": food_name.replace("_"," ").title(),
        "portion": qty,
        "calories": nutrition.get("calories", 0),
        "protein": nutrition.get("protein", 0),
        "fat": nutrition.get("fat", 0),
        "carbs": nutrition.get("carbs", 0),
        "fiber": nutrition.get("fiber", 0),
        "sugar": nutrition.get("sugar", 0),
        "health_score": current_score
    })

    # ----------------------------
    # Display Daily Diary
    # ----------------------------
    st.markdown("---")
    st.markdown("### 📝 Food Diary")

    if len(st.session_state.diary) == 0:
        st.info("No meals logged yet.")
    else:
        diary_sorted = sorted(st.session_state.diary, key=lambda x: x["health_score"], reverse=True)
        for entry in diary_sorted:
            st.markdown(
                f"""
                <div style='border-radius:10px; padding:15px; margin-bottom:10px; 
                            background:linear-gradient(135deg,#f0f0f0,#ffffff); 
                            box-shadow:0 4px 8px; color:black;'>
                    <div style='font-weight:bold; font-size:16px;'>{entry['food']} ({entry['portion']} g)</div>
                    <div style='display:flex; justify-content:space-between; padding-top:5px;'>
                        <span>🔥 Calories: {entry['calories']} kcal</span>
                        <span>💪 Health Score: {entry['health_score']}/10</span>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

    # ----------------------------
    # Daily Summary
    # ----------------------------
    total_calories = sum(m["calories"] for m in st.session_state.diary)
    total_protein = sum(m["protein"] for m in st.session_state.diary)
    total_fat = sum(m["fat"] for m in st.session_state.diary)
    total_carbs = sum(m["carbs"] for m in st.session_state.diary)
    total_fiber = sum(m["fiber"] for m in st.session_state.diary)

    if len(st.session_state.diary) > 0:
        st.markdown("---")
        st.markdown("### 📊 Nutrition Summary")
        st.markdown(
            f"""
            <div style='border-radius:10px; padding:15px; background:#e8f5e9; box-shadow:0 4px 8px; color:black;'>
                <div>🔥 Total Calories: {total_calories} kcal</div>
                <div>💪 Protein: {total_protein} g | 🥓 Fat: {total_fat} g | 🍞 Carbs: {total_carbs} g | 🌾 Fiber: {total_fiber} g</div>
            </div>
            """, unsafe_allow_html=True
        )


    # # ----------------------------
    # # Grad-CAM Section
    # # ----------------------------
    # st.markdown("---")
    # st.markdown("### 🔍 Model Focus (Grad-CAM)")

    # last_conv_layer_name = "block_13_expand"
    # grad_model = tf.keras.models.Model(
    #     [model.inputs],
    #     [model.get_layer(last_conv_layer_name).output, model.output]
    # )

    # with tf.GradientTape() as tape:
    #     conv_outputs, predictions = grad_model(x)
    #     loss = predictions[:, class_idx]

    # grads = tape.gradient(loss, conv_outputs)
    # pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    # conv_outputs = conv_outputs[0].numpy()
    # heatmap = np.sum(conv_outputs * pooled_grads.numpy(), axis=-1)

    # # Normalize heatmap
    # heatmap = np.maximum(heatmap, 0)
    # heatmap /= (np.max(heatmap) + 1e-8)
    
    # # Apply colormap
    # colormap = cm.get_cmap("jet")
    # heatmap = np.uint8(255 * (heatmap ** 0.7))  # gamma correction
    # heatmap_colored = colormap(heatmap)
    # heatmap_colored = np.uint8(255 * heatmap_colored[:, :, :3])

    # # Resize and overlay
    # heatmap_img = Image.fromarray(heatmap_colored).resize(img.size)
    # overlay = Image.blend(img.convert("RGB"), heatmap_img, alpha=0.4)

    # col1, col2 = st.columns(2)
    # col1.image(heatmap_img, caption="Grad-CAM Heatmap (model attention)", use_container_width=True)
    # col2.image(overlay, caption="Overlay (Red = important features)", use_container_width=True)
