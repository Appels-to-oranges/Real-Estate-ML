# %%
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import os
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# CONFIGURATION (edit these)
GOOGLE_API_KEY = ""

# Load the ZIP centroid data
zip_df = pd.read_csv("zip_cord.csv")

# Trim leading 0s from ZIP codes in zip_df
zip_df['ZIP'] = zip_df['ZIP'].astype(str)
zip_df['ZIP'] = zip_df['ZIP'].str.lstrip('0')
zip_df.sort_values(by='ZIP', inplace=True)
# print(zip_df.head())

# Cost by ZIP
zip_cost = pd.read_csv("zip_cost.csv")
zip_cost.rename(columns={"RegionName": "ZIP"}, inplace=True)

# Trim leading 0s from ZIP codes in zip_cost
zip_cost['ZIP'] = zip_cost['ZIP'].astype(str)
zip_cost['ZIP'] = zip_cost['ZIP'].str.lstrip('0')
zip_cost.sort_values(by='ZIP', inplace=True)
# print(zip_cost.head())
# order by region id

# Merge ZIP data with cost data
df = zip_df.merge(zip_cost, on="ZIP", how="left")
# print(df.head())

# to just filter to west coast states
df = df[df['State'].isin(['OR', 'WA'])]
# df length
print("Filtered dataset length:", len(df))

# distinct zip codes
print("Distinct ZIP codes:", df['ZIP'].nunique())


def get_satellite_image(lat, lng, zoom=13, size="600x600"):
    """Fetches satellite image from Google Static Maps API."""
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lng}&zoom={zoom}&size={size}&maptype=satellite&key={GOOGLE_API_KEY}"
    )
    resp = requests.get(url)
    if resp.status_code == 200:
        return Image.open(BytesIO(resp.content))
    else:
        raise Exception(f"Failed to fetch image: {resp.status_code} - {resp.text}")

# create something to store the images in
# os.makedirs("images", exist_ok=True)

def save_image(image, zip_code):
    """Saves the image to a file."""
    filename = f"images/{zip_code}.png"
    image.save(filename)
    return filename

def get_color_percentages(image, green_threshold=1.2, blue_threshold=1.2, grey_threshold=0.2):
    """Returns the percentage of green, blue and grey in the image."""
    image = image.convert("RGB")
    pixels = list(image.getdata())
    total_pixels = len(pixels)
    
    green_pixels = blue_pixels = grey_pixels = 0
    for r, g, b in pixels:
        if g > r * green_threshold and g > b * green_threshold and g > 50:
            green_pixels += 1
        if b > r * blue_threshold and b > g * blue_threshold and b > 50:
            blue_pixels += 1
        if (abs(r - g) < grey_threshold * 255 and 
            abs(r - b) < grey_threshold * 255 and 
            abs(g - b) < grey_threshold * 255):
            grey_pixels += 1
    
    return (
        round((green_pixels / total_pixels) * 100, 2),
        round((blue_pixels / total_pixels) * 100, 2),
        round((grey_pixels / total_pixels) * 100, 2)
    )

def reshape_date_columns(df):
    """
    Converts wide-format date columns to long format.
    
    Args:
        df (pd.DataFrame): DataFrame with date columns in wide format
        
    Returns:
        pd.DataFrame: Reshaped DataFrame with dates as rows
    """
    # Reset index to keep it as a column
    df = df.reset_index()
    
    # Identify date columns (columns that can be parsed as dates)
    date_cols = [col for col in df.columns if isinstance(col, str) and 
                 any(str(year) in col for year in range(2000, 2030))]
    
    # Non-date columns to keep as identifiers
    id_vars = [col for col in df.columns if col not in date_cols]
    
    # Melt the DataFrame
    melted_df = pd.melt(df, 
                        id_vars=id_vars,
                        value_vars=date_cols,
                        var_name='Date',
                        value_name='Value')
    
    # Convert date column to datetime
    melted_df['Date'] = pd.to_datetime(melted_df['Date'])
    
    # Sort by index and date
    melted_df = melted_df.sort_values(['index', 'Date'])
    
    return melted_df

# Example usage
df = reshape_date_columns(df)
# print(df.head())

# # print earliest date
print("Earliest date in dataset:", df['Date'].min())
print("Latest date in dataset:", df['Date'].max())

# df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Create future date column
df['Future_Date'] = df['Date'] + pd.DateOffset(years=5)

# Create a copy of the dataframe with future values
future_df = df.copy()
future_df = future_df.rename(columns={'Value': 'Future_Value'})
# join df with future_df on ZIP and Date (5 years later)
df = df.merge(
    future_df[['ZIP','Future_Value','Date']],
    left_on=['ZIP', 'Future_Date'],
    right_on=['ZIP', 'Date'],
    how='left',
    suffixes=('', '_y')
)

# Calculate 5-year appreciation percentage where we have both values
df['5yr_Appreciation%'] = ((df['Future_Value'] - df['Value']) / df['Value'] * 100).round(2)

# Add some debug printing
print("\nDebug Information:")
print("Total rows:", len(df))
print("Rows with valid appreciation:", df['5yr_Appreciation%'].notna().sum())


additional_columns = ['Green%', 'Blue%', 'Grey%']
# add columns
for col in additional_columns:
    if col not in df.columns:
        df[col] = np.nan

# # for each unique zip code, get the color percentages and save the image
# for idx, zip in enumerate(df['ZIP'].unique()):
#     lat = df[df['ZIP'] == zip]['LAT'].values[0]
#     lng = df[df['ZIP'] == zip]['LNG'].values[0]
#     img = get_satellite_image(lat, lng)
#     green, blue, grey = get_color_percentages(img)
#     save_image(img, zip)
#     print(f"Processed ZIP: {zip}")
#     # zip x out of 820
#     print(f"Progress: {(idx + 1) / len(df['ZIP'].unique()) * 100:.2f}%")
#     # add the color percentages to the dataframe
#     df.loc[df['ZIP'] == zip, 'Green%'] = green
#     df.loc[df['ZIP'] == zip, 'Blue%'] = blue
#     df.loc[df['ZIP'] == zip, 'Grey%'] = grey


# get color pecentages for each zip from the image folder
def local_color_data(df):
    i = 0
    for zip in df['ZIP'].unique():
        img = Image.open(f"images/{zip}.png")
        green, blue, grey = get_color_percentages(img)
        df.loc[df['ZIP'] == zip, 'Green%'] = green
        df.loc[df['ZIP'] == zip, 'Blue%'] = blue
        df.loc[df['ZIP'] == zip, 'Grey%'] = grey
        print(f"Processed ZIP: {zip}")
        print(i / len(df['ZIP'].unique()) * 100)
        i += 1
    return df

# df.head()
# model = train_model(df)
# print((df['5yr_Appreciation%'].nunique()))
# Encode categorical variables

categorical_cols = ['State']
label_encoders = {}  # Store encoders to reuse later
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col]) 
    # test[col] = le.transform(test[col])
    label_encoders[col] = le  # Store encoder in case it's needed later

df = local_color_data(df)


df = df.dropna(subset=['Green%', 'Blue%', 'Grey%', '5yr_Appreciation%','State'])
X = df[['Green%', 'Blue%', 'Grey%','State']]
y = df['5yr_Appreciation%']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostRegressor()

param_grid = {
    'depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [100, 200, 300]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='r2', 
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best R² score:", grid_search.best_score_)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

import matplotlib.pyplot as plt

# Get feature importances
importances = best_model.get_feature_importance()
features = X.columns

# Sort features by importance
sorted_indices = importances.argsort()[::-1] 
sorted_features = features[sorted_indices]
sorted_importances = importances[sorted_indices]

# Plot
plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importances)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis() 
plt.tight_layout()

plt.show()

# Save the model
# best_model.save_model("best_model.cbm")

