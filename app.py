import streamlit as st
import pandas as pd
import joblib

# Load trained ML model and encoder
model = joblib.load("career_model.pkl")
skill_encoder = joblib.load("skill_encoder.pkl")

# Page settings
st.set_page_config(
    page_title="AI Career Guidance System",
    page_icon="🎓"
)

# Custom design
st.markdown("""
<style>
.main {
    background-color: #f7f8fc;
}

h1 {
    text-align: center;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    padding: 10px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎓 AI Career Guidance System")

st.write(
    "Find a suitable career based on your skills, interests and academic preferences."
)

st.divider()

# Student Assessment
st.header("👤 Student Assessment")

name = st.text_input("Enter your Name")

skills = st.multiselect(
    "💻 Select your Skills",
    [
        "Python",
        "C/C++",
        "Java",
        "SQL",
        "Machine Learning",
        "Data Analysis",
        "Web Development",
        "Cyber Security",
        "UI/UX Design"
    ]
)

interests = st.multiselect(
    "❤️ Select your Interests",
    [
        "Artificial Intelligence",
        "Data Science",
        "Software Development",
        "Cyber Security",
        "Web Development",
        "Design"
    ]
)

subject = st.selectbox(
    "📚 Select your Favourite Subject",
    [
        "Computer Science",
        "Mathematics",
        "Programming",
        "Physics",
        "Design"
    ]
)

experience = st.selectbox(
    "📈 Experience Level",
    [
        "Beginner",
        "Intermediate",
        "Advanced"
    ]
)

# Recommendation
if st.button("🔍 Get Career Recommendation"):

    if not name:
        st.warning("Please enter your name.")

    elif not skills:
        st.warning("Please select at least one skill.")

    elif not interests:
        st.warning("Please select at least one interest.")

    else:
        st.success(
            f"Welcome {name}! Your assessment is complete. 🎉"
        )

        st.header("🎯 Recommended Career")

        # Combine skills, interests and favourite subject
        input_features_list = (
            list(skills) +
            list(interests) +
            [subject]
        )

        # Convert inputs into ML features
        input_features = skill_encoder.transform(
            [input_features_list]
        )

        # Predict career
        prediction = model.predict(input_features)

        st.success(
            f"🤖 Recommended Career: {prediction[0]}"
        )
      
