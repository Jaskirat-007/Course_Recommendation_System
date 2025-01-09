import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load and preprocess the dataset
def load_and_preprocess_data(file_path="udemy.csv"):
    """
    Load and preprocess the course dataset.
    """
    data = pd.read_csv(file_path)

    # Check for required columns
    required_columns = ["course title", "course description", "course level", "course url"]
    for col in required_columns:
        if col not in data.columns:
            raise KeyError(f"Column '{col}' is missing in the dataset. Available columns: {data.columns.tolist()}")

    # Rename columns to match expected structure
    data.rename(columns={
        "course title": "title",
        "course description": "description",
        "course level": "category",
        "course url": "url"
    }, inplace=True)

    # Fill missing values
    data.fillna("", inplace=True)

    # Combine relevant text fields for content-based filtering
    data["combined_features"] = (
        data["title"] + " " + data["description"] + " " + data["category"]
    )

    return data

# Train a content-based recommendation system
def train_recommendation_system(data):
    """
    Train a content-based recommendation system using TF-IDF and cosine similarity.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(data["combined_features"])

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return similarity_matrix

# Get recommendations for a course
def get_recommendations(course_id, data, similarity_matrix, top_n=5):
    """
    Get course recommendations based on the similarity matrix.
    """
    # Find similar courses
    similarity_scores = list(enumerate(similarity_matrix[course_id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the top N recommendations
    recommended_indices = [i[0] for i in similarity_scores[1 : top_n + 1]]
    recommended_courses = data.iloc[recommended_indices]

    return recommended_courses[["title", "description", "category", "url"]]
