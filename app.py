# import streamlit as st
# import pandas as pd
# import pickle
# from sklearn.metrics.pairwise import cosine_similarity

# # Load the vectorizer, tfidf matrix, and data
# with open('tfidf_vectorizer.pkl', 'rb') as file:
#     vectorizer = pickle.load(file)

# with open('tfidf_matrix.pkl', 'rb') as file:
#     tfidf_matrix = pickle.load(file)

# with open('cleaned_data.pkl', 'rb') as file:
#     data = pickle.load(file)

# # Function to recommend restaurants with rating-based weighting
# def recommend_restaurants(liked_restaurant, cuisine, price_for_two, top_n=5):
#     # Filter based on the keywords in cuisine
#     filtered_data = data[
#         (data['cuisine'].str.contains(cuisine, case=False, na=False)) &
#         (data['price_for_two'] <= price_for_two)
#     ]

#     if filtered_data.empty:
#         return "No restaurants found matching your criteria."

#     liked_index = data[data['names'].str.contains(liked_restaurant, case=False, na=False)].index

#     if liked_index.empty:
#         return "The liked restaurant is not found in the dataset."

#     liked_index = liked_index[0]

#     similarity_scores = cosine_similarity(tfidf_matrix[liked_index], tfidf_matrix[filtered_data.index]).flatten()

#     max_rating = data['ratings'].max()
#     filtered_data['weighted_score'] = similarity_scores * (filtered_data['ratings'] / max_rating)

#     recommendations = filtered_data.sort_values(by='weighted_score', ascending=False).head(top_n)

#     return recommendations[['names', 'cuisine', 'signature dishes', 'price_for_two', 'ratings', 'location']]

# # Streamlit UI setup
# st.title("Restaurant Recommendation System")

# # Sidebar for user inputs
# st.sidebar.header("User Input Features")
# liked_restaurant = st.sidebar.text_input("Enter a restaurant you liked")
# cuisine = st.sidebar.text_input("Preferred Cuisine (e.g., Indian, Chinese)")
# price_for_two = st.sidebar.slider("Budget for Two", 0, 3000, 500)  # Set max to 3000

# # Button to get recommendations
# if st.sidebar.button("Get Recommendations"):
#     recommendations = recommend_restaurants(liked_restaurant, cuisine, price_for_two)
#     if isinstance(recommendations, str):
#         st.write(recommendations)
#     else:
#         st.subheader("Recommended Restaurants")
#         for index, row in recommendations.iterrows():
#             st.write(f"{row['names']}: {row['signature dishes']} - {row['price_for_two']} - Rating: {row['ratings']} - Location: {row['location']}")





import streamlit as st
import pandas as pd
import pickle
import requests
import math
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Load the vectorizer, tfidf matrix, and data
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('tfidf_matrix.pkl', 'rb') as file:
    tfidf_matrix = pickle.load(file)

with open('output_with_lat_lon.pkl', 'rb') as file:
    data = pickle.load(file)

# Geocoding function to get latitude and longitude from an address using LocationIQ API
def get_lat_lon_from_address(address, access_token):
    try:
        url = f"https://us1.locationiq.com/v1/search.php?key={access_token}&q={address}&format=json"
        response = requests.get(url)
        data = response.json()
        if len(data) > 0:
            lat = data[0]['lat']
            lon = data[0]['lon']
            return float(lat), float(lon)
        else:
            return None, None
    except Exception as e:
        print(f"Error fetching geocode for {address}: {e}")
        return None, None

# Haversine formula to calculate distance between two points on Earth
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c  # Distance in kilometers
    return distance

# Filter restaurants by location based on user latitude and longitude
def filter_by_location(user_lat, user_lon, restaurants, radius=7):
    filtered_restaurants = []
    for _, row in restaurants.iterrows():
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            restaurant_lat = float(row['latitude'])
            restaurant_lon = float(row['longitude'])
            distance = haversine_distance(user_lat, user_lon, restaurant_lat, restaurant_lon)
            if distance <= radius:
                filtered_restaurants.append(row)
    return pd.DataFrame(filtered_restaurants)

# Hybrid recommendation function
def recommend_restaurants(user_lat, user_lon, cuisine, price_for_two, planning_for, liked_restaurant=None, top_n=5, radius=7):
    # Filter by location
    location_filtered_data = filter_by_location(user_lat, user_lon, data, radius)
    
    # Filter by content-based criteria (cuisine, price, features)
    content_filtered_data = location_filtered_data[
        (location_filtered_data['cuisine'].str.contains(cuisine, case=False, na=False)) &
        (location_filtered_data['price_for_two'] <= price_for_two) &
        (location_filtered_data['more_info'].str.contains(planning_for, case=False, na=False))
    ].copy()

    # If not enough restaurants, relax the search criteria
    if len(content_filtered_data) < top_n:
        related_cuisines = location_filtered_data[location_filtered_data['cuisine'].str.contains(cuisine, case=False, na=False)]
        content_filtered_data = pd.concat([content_filtered_data, related_cuisines]).drop_duplicates().reset_index(drop=True)

    # If still less than top_n, return available options
    if len(content_filtered_data) < top_n:
        return content_filtered_data[['names', 'cuisine', 'signature dishes', 'price_for_two', 'ratings', 'location', 'more_info']]

    # If liked restaurant is provided
    if liked_restaurant:
        match = process.extractOne(liked_restaurant, data['names'])
        if match[1] >= 80:  # If match confidence is above 80%
            liked_index = data[data['names'] == match[0]].index[0]
        else:
            return content_filtered_data.sort_values(by='ratings', ascending=False).head(top_n)[['names', 'cuisine', 'signature dishes', 'price_for_two', 'ratings', 'location', 'more_info']]
    else:
        liked_index = None

    # Calculate similarity if a liked restaurant is provided
    if liked_index is not None:
        similarity_scores = cosine_similarity(tfidf_matrix[liked_index], tfidf_matrix[content_filtered_data.index]).flatten()
        content_filtered_data['similarity_score'] = similarity_scores
        content_filtered_data['weighted_score'] = (
            (0.5 * content_filtered_data['similarity_score']) + 
            (0.5 * (content_filtered_data['ratings'] / data['ratings'].max()))
        )
        recommendations = content_filtered_data.sort_values(by='weighted_score', ascending=False).head(top_n)
    else:
        content_filtered_data['similarity_score'] = 1  # Assigning a constant for simplicity
        content_filtered_data['weighted_score'] = (
            (0.5 * content_filtered_data['similarity_score']) + 
            (0.5 * (content_filtered_data['ratings'] / data['ratings'].max()))
        )
        recommendations = content_filtered_data.sort_values(by='weighted_score', ascending=False).head(top_n)

    return recommendations[['names', 'cuisine', 'signature dishes', 'price_for_two', 'ratings', 'location', 'more_info']]

# Streamlit UI setup
st.title("Restaurant Recommendation System")

# Sidebar for user inputs
st.sidebar.header("User Input Features")
address = st.sidebar.text_input("Enter your address")
access_token = "5ce10ccbec0d4375ab977f6d6859248c"

user_lat, user_lon = None, None
if address:
    user_lat, user_lon = get_lat_lon_from_address(address, access_token)
    if user_lat is None or user_lon is None:
        st.sidebar.write("Could not find coordinates for the given address.")

cuisine = st.sidebar.text_input("Preferred Cuisine (e.g., Indian, Chinese)")
price_for_two = st.sidebar.slider("Budget for Two", 0, 3000, 500)  # Set max to 3000
planning_for = st.sidebar.text_input("Enter the occasion (e.g., family, friends)")
liked_restaurant = st.sidebar.text_input("Enter a restaurant you liked (optional)")

# Button to get recommendations
if st.sidebar.button("Get Recommendations"):
    if user_lat and user_lon:
        recommendations = recommend_restaurants(user_lat, user_lon, cuisine, price_for_two, planning_for, liked_restaurant)
        if isinstance(recommendations, str):
            st.write(recommendations)
        else:
            st.subheader("Recommended Restaurants")
            for index, row in recommendations.iterrows():
                st.write(f"{row['names']}: {row['signature dishes']} - {row['price_for_two']} - Rating: {row['ratings']} - Location: {row['location']}")
    else:
        st.write("Please provide a valid address.")
