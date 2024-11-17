# 🍽️ Restaurant Recommendation System for Hyderabad

Welcome to the **Restaurant Recommendation System** built using **content-based filtering**! This project offers personalized restaurant recommendations based on user preferences, location, and cuisine, using the power of **Python**, **Streamlit**, and **geolocation APIs**.

---

## 📋 Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Tech Stack](#tech-stack)  
4. [Setup and Installation](#setup-and-installation)  
5. [Usage](#usage)  
6. [API Details](#api-details)  
7. [How It Works](#how-it-works)  

---

## 📝 Overview

This **Restaurant Recommendation System** recommends restaurants based on:  
- **Location** (with geolocation support via LocationIQ API)  
- **User Preferences** (e.g., cuisine type, budget)  
- **User Age** (suggests age-appropriate options)  
- **Similar Restaurants** (if a liked restaurant is provided)  

Built with **content-based filtering** and advanced techniques such as cosine similarity and fuzzy matching, it delivers highly accurate recommendations.

---

## ✨ Features

- **Real-time location filtering** to find restaurants near the user.  
- **Budget and cuisine filtering** based on user input.  
- **Recommendation of similar restaurants** based on user history or input.  
- **Streamlit UI** for an interactive and seamless user experience.  
- **Fast fuzzy matching** to improve recommendation accuracy even with partial input.  

---

## ⚙️ Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python  
- **Libraries**: 
  - Cosine Similarity (for restaurant comparison)
  - Fuzzy Matching (via `fuzzywuzzy`)
- **API**: LocationIQ (for geolocation services)  

---

## 🚀 Setup and Installation

1. **Clone the repository**:  
   ```bash
   git clone https://github.com/Abhinaycs/Restaurant-Recommendation-System-Hyderabad-city.git
   cd restaurant-recommendation-system
   ```

2. **Create a virtual environment and activate it**:  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your LocationIQ API Key**:  
   - Create a `.env` file in the project directory with the following content:  
     ```text
     LOCATIONIQ_API_KEY=your_api_key_here
     ```

5. **Run the Streamlit app**:  
   ```bash
   streamlit run app.py
   ```

6. **Access the app**:  
   Visit `http://localhost:8501` in your browser.

**Try here** : https://restaurant-recommendation-system-hyderabad-city-armgmzq6zbyh28.streamlit.app/ 
---

## 📊 Usage

1. Open the web interface.  
2. Input your **location**, **age**, **budget**, and **preferred cuisine**.  
3. Optionally, enter a **restaurant you like** for similar recommendations.  
4. Click **Get Recommendations** to see personalized suggestions!

---

## 🌐 API Details

- **LocationIQ API**:  
  Used for converting user inputs into geolocations to filter restaurants based on proximity.  
  [Learn more about LocationIQ](https://locationiq.com/).

---

## 🛠️ How It Works

1. **Geolocation filtering**: Filters restaurants within a defined radius using LocationIQ API.  
2. **Content-based filtering**: Compares restaurant data using cosine similarity for personalized recommendations.  
3. **Fuzzy matching**: Enhances the recommendation system by matching user inputs with restaurant names, even if the input isn’t exact.  
4. **Preference and budget filtering**: Ensures that only restaurants matching the user’s requirements are shown.

