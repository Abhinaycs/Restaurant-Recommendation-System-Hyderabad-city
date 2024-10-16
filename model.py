import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("final_cleaned.csv")

data['features'] = data['more_info'].fillna('') + ',' + data['special features'].fillna('')
data['profile'] = (
    data['cuisine'].astype(str) + " " +
    data['price_for_two'].astype(str) + " " +
    data['signature dish'].fillna("").astype(str) +
    data['features'].fillna("").astype(str)
)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['profile'])

def get_age_group_features(age):
    if 18 <= age <= 25:
        return ["Nightlife", "Live Band", "Fast Delivery", "Pocket Friendly Prices"]
    elif 26 <= age <= 35:
        return ["Rooftop View", "Couple Friendly", "Craft Beer", "Trendy Decor"]
    elif 36 <= age <= 50:
        return ["Family Restaurant", "Buffet Variety", "Kid Friendly", "Extensive Menu"]
    elif age > 50:
        return ["Calm Ambience", "Courteous Staff", "Healthy Options", "Soft Music"]
    else:
        return []

def recommend_restaurants(liked_restaurant, cuisine, price_for_two, planning_for, age, top_n=5):
    age_features = get_age_group_features(age)
    
    filtered_data = data[
        (data['cuisine'].str.contains(cuisine, case=False, na=False)) &
        (data['price_for_two'] <= price_for_two) &
        (data['features'].str.contains(planning_for, case=False, na=False)) &
        (data['features'].apply(lambda x: any(feature in x for feature in age_features)))
    ]

    if filtered_data.empty:
        return "No restaurants found matching your criteria."

    liked_index = data[data['name'].str.contains(liked_restaurant, case=False, na=False)].index

    if liked_index.empty:
        return "The liked restaurant is not found in the dataset."

    liked_index = liked_index[0]
    
    similarity_scores = cosine_similarity(tfidf_matrix[liked_index], tfidf_matrix[filtered_data.index]).flatten()

    max_rating = data['rating'].max()
    filtered_data['weighted_score'] = similarity_scores * (filtered_data['rating'] / max_rating)

    recommendations = filtered_data.sort_values(by=['rating', 'weighted_score'], ascending=[False, False]).head(top_n)
    return recommendations[['name', 'cuisine', 'signature dish', 'price_for_two', 'rating', 'location', 'features']]

liked_restaurant = input("Enter a restaurant you liked: ")
cuisine = input("Enter preferred cuisine: ")
price_for_two = int(input("Enter your budget for two people: "))
planning_for = input("Enter the occasion: ")
age = int(input("Enter your age: "))

result = recommend_restaurants(liked_restaurant, cuisine, price_for_two, planning_for, age)
print(result)
