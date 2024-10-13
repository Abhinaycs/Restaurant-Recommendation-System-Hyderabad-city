import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load the vectorizer, tfidf matrix, and data
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('tfidf_matrix.pkl', 'rb') as file:
    tfidf_matrix = pickle.load(file)

with open('cleaned_data.pkl', 'rb') as file:
    data = pickle.load(file)

# Function to recommend restaurants with rating-based weighting
def recommend_restaurants(liked_restaurant, cuisine, price_for_two, top_n=5):
    # Filter based on the keywords in cuisine
    filtered_data = data[
        (data['cuisine'].str.contains(cuisine, case=False, na=False)) &
        (data['price_for_two'] <= price_for_two)
    ]

    if filtered_data.empty:
        return "No restaurants found matching your criteria."

    liked_index = data[data['names'].str.contains(liked_restaurant, case=False, na=False)].index

    if liked_index.empty:
        return "The liked restaurant is not found in the dataset."

    liked_index = liked_index[0]

    similarity_scores = cosine_similarity(tfidf_matrix[liked_index], tfidf_matrix[filtered_data.index]).flatten()

    max_rating = data['ratings'].max()
    filtered_data['weighted_score'] = similarity_scores * (filtered_data['ratings'] / max_rating)

    recommendations = filtered_data.sort_values(by='weighted_score', ascending=False).head(top_n)

    return recommendations[['names', 'cuisine', 'signature dishes', 'price_for_two', 'ratings', 'location']]

# Streamlit UI setup
st.title("Restaurant Recommendation System")

# Sidebar for user inputs
st.sidebar.header("User Input Features")
liked_restaurant = st.sidebar.text_input("Enter a restaurant you liked")
cuisine = st.sidebar.text_input("Preferred Cuisine (e.g., Indian, Chinese)")
price_for_two = st.sidebar.slider("Budget for Two", 0, 3000, 500)  # Set max to 3000

# Button to get recommendations
if st.sidebar.button("Get Recommendations"):
    recommendations = recommend_restaurants(liked_restaurant, cuisine, price_for_two)
    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        st.subheader("Recommended Restaurants")
        for index, row in recommendations.iterrows():
            st.write(f"{row['names']}: {row['signature dishes']} - {row['price_for_two']} - Rating: {row['ratings']} - Location: {row['location']}")
