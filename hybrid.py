import requests
import math
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Load the dataset and saved files
def load_csv(file_path):
    return pd.read_csv(file_path)

def load_pkl(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

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
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):  # Ensure lat/lon are not missing
            restaurant_lat = float(row['latitude'])
            restaurant_lon = float(row['longitude'])
            distance = haversine_distance(user_lat, user_lon, restaurant_lat, restaurant_lon)
            if distance <= radius:
                filtered_restaurants.append(row)
    
    return pd.DataFrame(filtered_restaurants)

# Preprocessing
data = load_csv("output_with_lat_lon.csv")
data['features'] = data['more_info'].fillna('') + ',' + data['special features'].fillna('')
data['profile'] = (
    data['cuisine'].astype(str) + " " +
    data['price_for_two'].astype(str) + " " +
    data['signature dish'].fillna("").astype(str) +
    data['features'].fillna("").astype(str)
)

# Load TF-IDF vectorizer and matrix or generate them if not available
try:
    vectorizer = load_pkl("tfidf_vectorizer.pkl")
    tfidf_matrix = load_pkl("tfidf_matrix.pkl")
    print("Loaded TF-IDF vectorizer and matrix from pickle files.")
except FileNotFoundError:
    print("Pickle files not found. Generating TF-IDF vectorizer and matrix...")
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['profile'])
    with open("tfidf_vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)
    with open("tfidf_matrix.pkl", "wb") as file:
        pickle.dump(tfidf_matrix, file)

# Hybrid recommendation function
def recommend_restaurants(user_lat, user_lon, cuisine, price_for_two, planning_for, liked_restaurant=None, top_n=5, radius=7):
    # Filter by location first
    location_filtered_data = filter_by_location(user_lat, user_lon, data, radius)
    
    # Filter by content-based criteria (cuisine, price, features)
    content_filtered_data = location_filtered_data[
        (location_filtered_data['cuisine'].str.contains(cuisine, case=False, na=False)) &
        (location_filtered_data['price_for_two'] <= price_for_two) &
        (location_filtered_data['features'].str.contains(planning_for, case=False, na=False))
    ].copy()

    # If not enough restaurants, relax the search criteria
    if len(content_filtered_data) < top_n:
        related_cuisines = location_filtered_data[location_filtered_data['cuisine'].str.contains(cuisine, case=False, na=False)]
        content_filtered_data = pd.concat([content_filtered_data, related_cuisines]).drop_duplicates().reset_index(drop=True)

    # If still less than top_n, return available options
    if len(content_filtered_data) < top_n:
        print("Fewer than 5 restaurants found. Showing available options:")
        return content_filtered_data[['name', 'cuisine', 'signature dish', 'price_for_two', 'rating', 'location', 'features']]

    # If liked restaurant is provided
    if liked_restaurant:
        match = process.extractOne(liked_restaurant, data['name'])
        if match[1] >= 80:  # If match confidence is above 80%
            liked_index = data[data['name'] == match[0]].index[0]
        else:
            print("Liked restaurant not found. Showing top options based on other criteria.")
            return content_filtered_data.sort_values(by='rating', ascending=False).head(top_n)[['name', 'cuisine', 'signature dish', 'price_for_two', 'rating', 'location', 'features']]
    else:
        liked_index = None

    # Calculate similarity if a liked restaurant is provided
    if liked_index is not None:
        similarity_scores = cosine_similarity(tfidf_matrix[liked_index], tfidf_matrix[content_filtered_data.index]).flatten()
        content_filtered_data['similarity_score'] = similarity_scores
        content_filtered_data['weighted_score'] = (
            (0.5 * content_filtered_data['similarity_score']) + 
            (0.5 * (content_filtered_data['rating'] / data['rating'].max()))
        )
        recommendations = content_filtered_data.sort_values(by='weighted_score', ascending=False).head(top_n)
    else:
        content_filtered_data['similarity_score'] = 1  # Assigning a constant for simplicity
        content_filtered_data['weighted_score'] = (
            (0.5 * content_filtered_data['similarity_score']) + 
            (0.5 * (content_filtered_data['rating'] / data['rating'].max()))
        )
        recommendations = content_filtered_data.sort_values(by='weighted_score', ascending=False).head(top_n)

    return recommendations[['name', 'cuisine', 'signature dish', 'price_for_two', 'rating', 'location', 'features']]

# Input and recommend
address = input("Enter your address: ")
access_token = "pk.e28403f26ee75af55812430de27b810e"  # Your LocationIQ API key
user_lat, user_lon = get_lat_lon_from_address(address, access_token)
if user_lat is None or user_lon is None:
    print("Could not find coordinates for the given address.")
else:
    cuisine = input("Enter preferred cuisine: ")
    price_for_two = int(input("Enter your budget for two people: "))
    planning_for = input("Enter the occasion: ")
    liked_restaurant = input("Enter a restaurant you liked (leave blank if none): ")

    # Get recommendations
    result = recommend_restaurants(user_lat, user_lon, cuisine, price_for_two, planning_for, liked_restaurant)
    print(result)
