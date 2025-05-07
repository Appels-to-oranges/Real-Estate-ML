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

import cv2
import numpy as np
from PIL import Image
from skimage.measure import shannon_entropy

################################
# IMAGE COLLECTION

GOOGLE_API_KEY = ""

# create something to store the images in
# os.makedirs("images", exist_ok=True)

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

def save_image(image, zip_code):
    """Saves the image to a file."""
    filename = f"images/{zip_code}.png"
    image.save(filename)
    return filename

################################
# IMAGE PROCESSING

def get_edge_density(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    return np.sum(edges > 0) / edges.size

def get_entropy(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return shannon_entropy(gray)

def get_hsv_means(image):
    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return np.mean(h), np.mean(s), np.mean(v)

def local_color_data(df):
    i = 0
    zips = df['ZIP'].unique()

    for zip_code in zips:
        try:
            img = Image.open(f"images/{zip_code}.png").convert("RGB")
            green, blue, grey = get_color_percentages(img)
            edge_density = get_edge_density(img)
            entropy = get_entropy(img)
            h_mean, s_mean, v_mean = get_hsv_means(img)

            df.loc[df['ZIP'] == zip_code, 'Green%'] = green
            df.loc[df['ZIP'] == zip_code, 'Blue%'] = blue
            df.loc[df['ZIP'] == zip_code, 'Grey%'] = grey
            df.loc[df['ZIP'] == zip_code, 'EdgeDensity'] = edge_density
            df.loc[df['ZIP'] == zip_code, 'Entropy'] = entropy
            df.loc[df['ZIP'] == zip_code, 'HueMean'] = h_mean
            df.loc[df['ZIP'] == zip_code, 'SatMean'] = s_mean
            df.loc[df['ZIP'] == zip_code, 'ValMean'] = v_mean

            print(f"Processed ZIP: {zip_code} ({i / len(zips) * 100:.1f}%)")

        except Exception as e:
            print(f"Error processing {zip_code}: {e}")

        i += 1

    return df

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

################################

def melt_date_columns(df):
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

################################

# Load the ZIP centroid data
zip_df = pd.read_csv("zip_cord.csv")
# Trim leading 0s from ZIP codes in zip_df
zip_df['ZIP'] = zip_df['ZIP'].astype(str)
zip_df['ZIP'] = zip_df['ZIP'].str.lstrip('0')
zip_df.sort_values(by='ZIP', inplace=True)
print(zip_df.head())

# Cost by ZIP
zip_cost = pd.read_csv("zip_cost.csv")
zip_cost.rename(columns={"RegionName": "ZIP"}, inplace=True)
# Trim leading 0s from ZIP codes in zip_cost
zip_cost['ZIP'] = zip_cost['ZIP'].astype(str)
zip_cost['ZIP'] = zip_cost['ZIP'].str.lstrip('0')
zip_cost.sort_values(by='ZIP', inplace=True)
print(zip_cost.head())
# order by region id

# Merge ZIP data with cost data
df = zip_df.merge(zip_cost, on="ZIP", how="left")
# print(df.head())

################################

# to just filter to west coast states
df = df[df['State'].isin(['OR', 'WA'])]
# df length
print("Filtered dataset length:", len(df))
# distinct zip codes
print("Distinct ZIP codes:", df['ZIP'].nunique())

df = melt_date_columns(df)
# print(df.head())
# # print earliest date
print("Earliest date in dataset:", df['Date'].min())
print("Latest date in dataset:", df['Date'].max())

# df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
# df['Month'] = df['Date'].dt.month

df.sample(10)
df.head()
df.info()
df.columns

# Drop date, average value by every other column
df.drop(columns=['index', 'Date','RegionID','RegionType'], inplace=True)    
df = df.groupby(['ZIP', 'LAT', 'LNG', 'SizeRank',
       'StateName', 'State', 'City', 'Metro', 'CountyName',
       'Year']).agg({'Value': 'median'}).reset_index()

df.sample(10)
df.head()
df.info(0)

# Poverty/Income Data by County 2023
SAIPE = pd.read_excel("est23all.xls")
# promote first row to header
SAIPE.columns = SAIPE.iloc[0]
SAIPE.columns

df = df.merge(SAIPE, on="CountyName", how="left")

df.sample(10)
# filter to 97203 only

####################################

df['Year'] = pd.to_datetime(df['Year'], format='%Y')

# Sort so shift operates predictably
df = df.sort_values(by=['ZIP', 'Year'])

# Shift by 1, 5, and 10 years forward within each ZIP
for n in [1, 5, 10]:
    df[f'{n}y_Future_Value'] = df.groupby('ZIP')['Value'].shift(-n)
    df[f'{n}yr_Appreciation%'] = (
        (df[f'{n}y_Future_Value'] - df['Value']) / df['Value'] * 100
    ).round(2)


df.sample(10)

####################################

df.columns

additional_columns = [
    'Green%', 'Blue%', 'Grey%',
    'EdgeDensity', 'Entropy',
    'HueMean', 'SatMean', 'ValMean']

# Add missing columns with NaN
for col in additional_columns:
    if col not in df.columns:
        df[col] = np.nan

df = local_color_data(df)

backup = df.copy()
# to csv
backup.to_csv('backup.csv', index=False, header=True)

categorical_cols = ['State','Metro','City','CountyName']
label_encoders = {}  # Store encoders to reuse later
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col]) 
    # test[col] = le.transform(test[col])
    label_encoders[col] = le  # Store encoder in case it's needed later


# df to csv
# df.to_csv('zip_data_full.csv', index=False, header=True)
################################
# %%
print(df.columns)

df = df.dropna()
# df.drop(columns=['1yr_Future_Value','StateName'], inplace=True)  

X = df[['LAT', 'LNG', 'SizeRank' , 'State','City', 'Metro', 'CountyName',
        'Poverty Estimate, All Ages', 'Poverty Percent, All Ages','Poverty Estimate, Age 0-17', 'Poverty Percent, Age 0-17','Poverty Estimate, Age 5-17 in Families',
        'Poverty Percent, Age 5-17 in Families', 'Median Household Income','% Difference Avg Income',
          '1yr_Appreciation%','Green%', 'Blue%', 'Grey%','EdgeDensity', 'Entropy','HueMean', 'SatMean', 
          'ValMean']]


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

################################

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


# %%
