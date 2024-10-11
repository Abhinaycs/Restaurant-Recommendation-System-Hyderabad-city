import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'
}

def get_url():
    localities = [
        "https://www.zomato.com/hyderabad/jubilee-hills-restaurants",
        "https://www.zomato.com/hyderabad/gachibowli-restaurants",
        "https://www.zomato.com/hyderabad/banjara-hills-restaurants",
        "https://www.zomato.com/hyderabad/hitech-city-restaurants",
        "https://www.zomato.com/hyderabad/madhapur-restaurants",
        "https://www.zomato.com/hyderabad/kondapur-restaurants",
        "https://www.zomato.com/hyderabad/kukatpally-restaurants",
        "https://www.zomato.com/hyderabad/begumpet-restaurants",
        "https://www.zomato.com/hyderabad/himayath-nagar-restaurants",
        "https://www.zomato.com/hyderabad/tolichowki-restaurants"
        "https://www.zomato.com/hyderabad/ameerpet-restaurants",
        "https://www.zomato.com/hyderabad/somajiguda-restaurants",
        "https://www.zomato.com/hyderabad/film-nagar-restaurants",
        "https://www.zomato.com/hyderabad/paradise-circle-restaurants",
        "https://www.zomato.com/hyderabad/sainikpuri-restaurants",
        "https://www.zomato.com/hyderabad/neclace-road-restaurants",
        "https://www.zomato.com/hyderabad/kothapet-restaurants",
        "https://www.zomato.com/hyderabad/s-d-road-restaurants",
        "https://www.zomato.com/hyderabad/abids-restaurants",
        "https://www.zomato.com/hyderabad/kompally-restaurants",
        "https://www.zomato.com/hyderabad/masab-tank-restaurants",
        "https://www.zomato.com/hyderabad/gandipet-restaurants",
        "https://www.zomato.com/hyderabad/l-b-nagar-restaurants",
        "https://www.zomato.com/hyderabad/miyapur-restaurants",
        "https://www.zomato.com/hyderabad/karkhana-restaurants",
        "https://www.zomato.com/hyderabad/basheer-bagh-restaurants",
        "https://www.zomato.com/hyderabad/panjagutta-restaurants",
        "https://www.zomato.com/hyderabad/a-s-rao-nagar/restuarants",
        "https://www.zomato.com/hyderabad/uppal/restuarants"
    ]

    all_urls = []

    driver = webdriver.Chrome()

    for link in localities:
        driver.get(link)
        time.sleep(2)

        while True:
            last_height = driver.execute_script("return document.body.scrollHeight")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        divs = soup.findAll('div', class_='jumbo-tracker')

        for parent in divs:
            name_tag = parent.find("h4")
            if name_tag is not None:
                rest_name = name_tag.text.strip()
                link_tag = parent.find("a")
                restaurant_link = urljoin("https://www.zomato.com", link_tag.get('href'))
                all_urls.append(restaurant_link)

    df = pd.DataFrame({'restaurant_links': all_urls})
    df.to_csv('restaurant_links_v2.csv', index=False)
    print("Restaurant links saved to restaurant_links_v2.csv")

    driver.quit()

get_url()

def get_data():
    links = pd.read_csv('restaurant_links_v2.csv')
    restaurant_links = links['restaurant_links'].tolist()

    all_urls = []
    all_rest_name = []
    all_ratings = []
    all_price = []
    all_cuisine = []
    all_images = []  
    all_opening_hours = []
    all_locations = []
    all_signature_dishes = []
    all_special_features = []
    all_safety_measures = [] 
    all_address = [] 

    for link in restaurant_links: ###############################
        response = requests.get(link, headers=headers)
        inner_soup = BeautifulSoup(response.content, 'html.parser')

        rest_name = inner_soup.find('h1', class_='sc-7kepeu-0 sc-iSDuPN fwzNdh')
        rest_name = rest_name.text.strip() 

        rating_tag = inner_soup.find('div', class_ ='sc-1q7bklc-6 liCXOR')
        rating_value = rating_tag.text.strip() if rating_tag else 'Not available'

        price_tag = inner_soup.find('h3', string='Average Cost') 
        price_value = price_tag.find_next_sibling().text if price_tag else 'Not available'

        cuisine_value = []
        h3_tag = inner_soup.find('h3', string='Cuisines')
        if h3_tag:
            section_tag = h3_tag.find_next('section')
            
            if section_tag:
                cuisines = section_tag.find_all('a')
                cuisine_value = ', '.join(cuisine.text.strip() for cuisine in cuisines)

        open_timing_tag = inner_soup.find('span', class_='sc-kasBVs dfwCXs')
        open_timing_value = open_timing_tag.text.strip() if open_timing_tag else 'Not available'

        # Extract location using the class found in the second image
        location_tag = inner_soup.find('a', class_='sc-clNaTc vNCcy')
        location_value = location_tag.text.strip() if location_tag else 'Not available'

        # Extract popular dishes (now renamed as signature dishes)
        popular_dishes_tag = inner_soup.find('h3', string='Popular Dishes')
        signature_dishes_text_value = popular_dishes_tag.find_next('p').text.strip() if popular_dishes_tag else 'Not available'

        # Extract what people say this place is known for (now renamed as special features)
        people_say_tag = inner_soup.find('h3', string='People Say This Place Is Known For')
        special_features_text_value = people_say_tag.find_next('p').text.strip() if people_say_tag else 'Not available'

        image_tag = inner_soup.find('img', class_='sc-s1isp7-5 eQUAyn')
        image = image_tag.get("src") if image_tag else None

        # Extract safety measures from specified section (new addition)
        safety_measures_section_1 = inner_soup.find('section', class_='sc-bgxRrC fHqOaY')
        safety_measures_value_list_items_1 = safety_measures_section_1.find_all('p') if safety_measures_section_1 else []

        # Additional HTML structure for hygiene measures 
        safety_measures_section_2_items= inner_soup.find_all('p', class_='sc-1hez2tp-0 fvARMW')  
        
        # Combine all safety measures into one list
        all_safety_measures_items = [item.text.strip() for item in safety_measures_value_list_items_1]
        all_safety_measures_items += [item.text.strip() for item in safety_measures_section_2_items]

        safety_measures_value_final = ", ".join(all_safety_measures_items) if all_safety_measures_items else 'Not available'

        # Extract address information from specified section (new addition)
        address_section = inner_soup.find('p', class_='sc-1hez2tp-0 clKRrC')
        address_value = address_section.text.strip() if address_section else 'Not available'

        # Append extracted data to lists
        all_urls.append(link)
        all_rest_name.append(rest_name)
        all_ratings.append(rating_value)
        all_price.append(price_value)
        all_cuisine.append(cuisine_value)
        all_images.append(image)  # Collecting image URLs here
        all_opening_hours.append(open_timing_value)
        all_locations.append(location_value)
        all_signature_dishes.append(signature_dishes_text_value)  # Now holds popular dishes 
        all_special_features.append(special_features_text_value)   # Now holds what people say about the restaurant 
        all_safety_measures.append(safety_measures_value_final)     # New column for safety measures 
        all_address.append(address_value)

    df = pd.DataFrame({
        'links': all_urls,
        'names': all_rest_name,
        'ratings': all_ratings,
        'price for two': all_price,
        'cuisine': all_cuisine,
        'images': all_images,
        'opening & closing time': all_opening_hours,
        'location': all_locations,
        'signature dishes': all_signature_dishes,  # Now holds popular dishes 
        'special features': all_special_features,   # Now holds what people say about the restaurant 
        'safety measures': all_safety_measures,     # New column for safety measures 
        'address': all_address  # New column for address 
    })

    df.to_csv('zomato_v2.csv', index=False)
    print('saved to zomato_v2.csv done')

get_data()
