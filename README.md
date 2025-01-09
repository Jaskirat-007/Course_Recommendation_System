# Course_Recommendation_System

A Streamlit-based course recommendation system leveraging content-based filtering to suggest similar courses using TF-IDF and cosine similarity.

## Features

- Loads and preprocesses a course dataset.
- Implements content-based filtering using **TF-IDF** (Term Frequency-Inverse Document Frequency) and **cosine similarity**.
- Recommends similar courses for a given input course.

## Files

- **`course_recommendation.py`**: Contains the main code for loading the dataset, training the recommendation system, and fetching recommendations.
- **`udemy.csv`**: Dataset containing course information. Ensure it includes the following columns:
  - `course title`
  - `course description`
  - `course level`
  - `course url`

## Requirements

- Python 3.7 or higher
- Required libraries:
  - `pandas`
  - `scikit-learn`

Install the dependencies using:
```bash
pip install -r requirements.txt

