import streamlit as st
from course_recommendation import load_and_preprocess_data, train_recommendation_system, get_recommendations

st.title("Course Recommendation System")

# Load and preprocess the dataset
data = load_and_preprocess_data("udemy.csv")

# Train the recommendation system
similarity_matrix = train_recommendation_system(data)

# Sidebar for user inputs
st.sidebar.header("Select a Course")
course_options = data["title"].tolist()
selected_course = st.sidebar.selectbox("Choose a course to get recommendations:", course_options)

# Display recommendations
if st.sidebar.button("Get Recommendations"):
    course_id = data[data["title"] == selected_course].index[0]
    recommendations = get_recommendations(course_id, data, similarity_matrix)

    st.write("### Recommended Courses")
    for _, row in recommendations.iterrows():
        st.write(f"**Title**: {row['title']}")
        st.write(f"*Category*: {row['category']}")
        st.write(f"{row['description']}")
        st.markdown(f"[Course Link]({row['url']})")
        st.markdown("---")
