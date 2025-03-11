import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


st.title("Weather Prediction App")

@st.cache_data
def load_data():
    df = pd.read_csv("seattle-weather.csv")
    df['weather'] = df['weather'].astype('category')  
    return df

df = load_data()

# Select features and target
features = ["precipitation", "temp_min", "temp_max", "wind"]
target = "weather"

# Convert categorical target to numeric labels
df[target + "_code"] = df[target].cat.codes  # Create a separate numeric column

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target + "_code"], test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
st.write(f"Model Accuracy: {accuracy:.2f}")

# Predict on user input
st.sidebar.header("Enter Weather Conditions")
precipitation = st.sidebar.slider("Precipitation", min_value=float(df["precipitation"].min()), max_value=float(df["precipitation"].max()), step=0.1)
temp_min = st.sidebar.slider("Minimum Temperature", min_value=float(df["temp_min"].min()), max_value=float(df["temp_min"].max()), step=0.1)
temp_max = st.sidebar.slider("Maximum Temperature", min_value=float(df["temp_max"].min()), max_value=float(df["temp_max"].max()), step=0.1)
wind = st.sidebar.slider("Wind Speed", min_value=float(df["wind"].min()), max_value=float(df["wind"].max()), step=0.1)

input_data = pd.DataFrame([[precipitation, temp_min, temp_max, wind]], columns=features)
prediction = model.predict(input_data)
predicted_weather = df[target].cat.categories[prediction[0]]  # Convert numeric prediction to original label
st.write(f"Predicted Weather: {predicted_weather}")

# weather image
weather_images = {
    "drizzle": "D:\Projects\Streamlit-App\drizzle.jpg",
    "rain": "D:\Projects\Streamlit-App\what-is-rain.jpg",
    "sun": "D:\Projects\Streamlit-App\sun-blog.webp",
    "snow": "D:\Projects\Streamlit-App\snow.webp",
    "fog": "D:\Projects\Streamlit-App\fog-imunotes.webp"
}

if predicted_weather in weather_images:
    st.image(weather_images[predicted_weather], caption=f"{predicted_weather.capitalize()} Weather", use_column_width=True)
