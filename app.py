import streamlit as st
import pandas as pd
import pickle
import requests
import math
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Load the vectorizer, tfidf matrix, and data
@st.cache_data
def load_files():
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    with open('tfidf_matrix.pkl', 'rb') as file:
        tfidf_matrix = pickle.load(file)
    with open('output_with_lat_lon.pkl', 'rb') as file:
        data = pickle.load(file)
    return vectorizer, tfidf_matrix, data

vectorizer, tfidf_matrix, data = load_files()

# Geocoding function to get latitude and longitude using LocationIQ API
def get_lat_lon_from_address(address, access_token):
    try:
        url = f"https://us1.locationiq.com/v1/search.php?key={access_token}&q={address}&format=json"
        response = requests.get(url)
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
        else:
            return None, None
    except Exception as e:
        st.error(f"Error fetching geocode: {e}")
        return None, None

# Haversine formula to calculate distance
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# Filter restaurants by location
def filter_by_location(user_lat, user_lon, restaurants, radius=7):
    return pd.DataFrame([
        row for _, row in restaurants.iterrows()
        if haversine_distance(user_lat, user_lon, float(row['latitude']), float(row['longitude'])) <= radius
    ])

# Age-based filtering function
def filter_by_age(data, age):
    if age <= 25:
        return data[data['more_info'].str.contains('birthday|casual|fast food|trendy', case=False, na=False)]
    elif 26 <= age <= 50:
        return data[data['more_info'].str.contains('family|formal|business', case=False, na=False)]
    else:
        return data[data['more_info'].str.contains('quiet|traditional', case=False, na=False)]

# Hybrid recommendation function
def recommend_restaurants(user_lat, user_lon, cuisine, price_for_two, planning_for, liked_restaurant, age, radius=7, top_n=5):
    try:
        location_filtered_data = filter_by_location(user_lat, user_lon, data, radius)
        
        # Apply age-based filtering
        age_filtered_data = filter_by_age(location_filtered_data, age)

        # Filter by content-based criteria
        content_filtered_data = age_filtered_data[
            (age_filtered_data['cuisine'].str.contains(cuisine, case=False, na=False)) &
            (age_filtered_data['price_for_two'] <= price_for_two) &
            (age_filtered_data['more_info'].str.contains(planning_for, case=False, na=False))
        ]

        if len(content_filtered_data) < top_n:
            related_cuisines = age_filtered_data[
                age_filtered_data['cuisine'].str.contains(cuisine, case=False, na=False)
            ]
            content_filtered_data = pd.concat([content_filtered_data, related_cuisines]).drop_duplicates()

        if len(content_filtered_data) < top_n:
            return content_filtered_data.sort_values(by='rating', ascending=False).head(top_n)

        if liked_restaurant:
            match = process.extractOne(liked_restaurant, data['name'])
            if match and match[1] >= 80:
                liked_index = data[data['name'] == match[0]].index[0]
                similarity_scores = cosine_similarity(
                    tfidf_matrix[liked_index], tfidf_matrix[content_filtered_data.index]
                ).flatten()
                content_filtered_data['similarity_score'] = similarity_scores
            else:
                content_filtered_data['similarity_score'] = 1  # Default similarity score for unmatched

        else:
            content_filtered_data['similarity_score'] = 1

        content_filtered_data['weighted_score'] = (
            0.5 * content_filtered_data['similarity_score'] +
            0.5 * (content_filtered_data['rating'] / data['rating'].max())
        )

        return content_filtered_data.sort_values(by='weighted_score', ascending=False).head(top_n)
    
    except Exception as e:
        print("No matching restaurant found for this criteria.")


def recommend_restaurants_city(liked_restaurant, cuisine, budget, occasion, top_n=5):
    matches = data[
        (data['cuisine'].str.contains(cuisine, case=False, na=False)) &
        (data['price_for_two'] <= budget) &
        (data['features'].str.contains(occasion, case=False, na=False))
    ]

    liked_index = data[data['name'].str.contains(liked_restaurant, case=False, na=False)].index
    if liked_index.empty:
        return "Liked restaurant not found in the dataset."

    similarity_scores = cosine_similarity(tfidf_matrix[liked_index[0]], tfidf_matrix[matches.index]).flatten()
    matches['weighted_score'] = similarity_scores * (matches['rating'] / data['rating'].max())
    return matches.sort_values(by=['rating', 'weighted_score'], ascending=[False, False]).head(top_n)

data['features'] = data['more_info'].fillna('') + ',' + data['special features'].fillna('')

st.title("Restaurant Recommendation System")

# Sidebar input
st.sidebar.header("User Input Features")
address = st.sidebar.text_input("Enter your address")
access_token = "pk.e28403f26ee75af55812430de27b810e"

cuisine = st.sidebar.text_input("Preferred Cuisine (e.g., Indian, Chinese)")
price_for_two = st.sidebar.slider("Budget for Two", 0, 3000, 500)
age = st.sidebar.slider("Your Age", 5, 80, 30)
planning_for = st.sidebar.selectbox(
    "Occasion / Planning for",
    ['Party', 'DJ', 'Music', 'Bar', 'Pool table', 'Romantic dining', 'Dance', 'Games', 'Live sports', 'Buffet']
)
liked_restaurant = st.sidebar.text_input("Liked Restaurant (optional)")

if st.sidebar.button("Get Recommendations"):
    recommendations = recommend_restaurants_city(
        liked_restaurant, cuisine, price_for_two, planning_for
    )
    if not recommendations.empty:
        st.subheader("Recommended Restaurants")
        st.dataframe(recommendations)

    if address:
        try:
            user_lat, user_lon = get_lat_lon_from_address(address, access_token)
            if user_lat and user_lon:
                recommendations = recommend_restaurants(
                    user_lat, user_lon, cuisine, price_for_two, planning_for, liked_restaurant, age
                )
                if not recommendations.empty:
                    st.subheader("Recommended Restaurants Nearby")
                    st.dataframe(recommendations)
                else:
                    st.warning("No recommendations found.")
            else:
                st.error("Invalid address.")
        
        except Exception as e:
            st.warning("No matching restaurant found for this criteria nearby.")
