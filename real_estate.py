# %%
################################
# Import necessary libraries

import requests
import pandas as pd
import os
import time
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import KNNImputer

from io import BytesIO
import cv2
from PIL import Image
from skimage.measure import shannon_entropy

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import Choropleth, CircleMarker
from folium.plugins import MarkerCluster
import branca.colormap as cm

################################
# IMAGE STUFF

# GOOGLE_API_KEY = ""

# # create something to store the images in
# # os.makedirs("images", exist_ok=True)

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
# Melt
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
zip_df['ZIP'] = zip_df['ZIP'].astype(str)
zip_df['ZIP'] = zip_df['ZIP'].str.lstrip('0')
zip_df.sort_values(by='ZIP', inplace=True)
# print(zip_df.head())

# Cost by ZIP
zip_cost = pd.read_csv("zip_cost.csv")
zip_cost.rename(columns={"RegionName": "ZIP"}, inplace=True)
zip_cost['ZIP'] = zip_cost['ZIP'].astype(str)
zip_cost['ZIP'] = zip_cost['ZIP'].str.lstrip('0')
zip_cost.sort_values(by='ZIP', inplace=True)
# print(zip_cost.head())

df = zip_df.merge(zip_cost, on="ZIP", how="left")
# print(df.head())
################################
# Clean

# Filter to west coast states (not you california)
df = df[df['State'].isin(['OR', 'WA'])]

print("Filtered dataset length:", len(df))
print("Distinct ZIP codes:", df['ZIP'].nunique())

df = melt_date_columns(df)

print("Earliest date in dataset:", df['Date'].min())
print("Latest date in dataset:", df['Date'].max())

df['Year'] = df['Date'].dt.year
# df['Month'] = df['Date'].dt.month

# df.sample(10)
# df.head()
# df.info()
# df.columns

df.drop(columns=['index', 'Date','RegionID','RegionType','StateName'], inplace=True)   

################################
# IMAGE STUFF

# image_data = pd.DataFrame(df, columns=['ZIP'])

# additional_columns = [
#     'Green%', 'Blue%', 'Grey%',
#     'EdgeDensity', 'Entropy',
#     'HueMean', 'SatMean', 'ValMean']

# # # Add missing columns with NaN
# for col in additional_columns:
#     if col not in image_data.columns:
#         image_data[col] = np.nan

# image_data = local_color_data(image_data)

# image_data.to_csv('image_data.csv', index=False, header=True)

image_data = pd.read_csv('image_data.csv')
image_data.sample(10)
image_data.columns
image_data['ZIP'] = image_data['ZIP'].astype(str)
image_data['ZIP'] = image_data['ZIP'].str.lstrip('0')

df = df.merge(
    image_data,
    how='left',
    left_on='ZIP',
    right_on='ZIP'
)

backup = df.copy()

df['ZIP'] = df['ZIP'].astype(int)

print("Image data merged successfully.")

################################
# Yearly and County Medians

df.drop(['SizeRank','ZIP','City'], inplace=True, axis=1)

df = df.groupby(['State', 'Metro', 'CountyName','Year']).agg({
    'Value': 'median',
    'Green%': 'mean',
    'Blue%': 'mean',
    'Grey%': 'mean',
    'EdgeDensity': 'mean',
    'Entropy': 'mean',
    'HueMean': 'mean',
    'SatMean': 'mean',
    'ValMean': 'mean',
    'LAT': 'mean',
    'LNG': 'mean'
}).reset_index()

# df.drop_duplicates(subset=['CountyName', 'Year'], inplace=True)

# df.describe()
# df.sample(10)

print("Grouped")

####################################
def utility_stuff():
    # # Utility Data by ZIP 2020

    # iou = pd.read_csv("iou 2020.csv")
    # noniou = pd.read_csv("noniou 2020.csv")

    # iou = iou.groupby('zip').agg('mean').reset_index()
    # noniou = noniou.groupby('zip').agg('mean').reset_index()

    # utility = pd.merge(iou, noniou, on='zip', how='outer')
    # utility.rename(columns={"zip": "ZIP"}, inplace=True)
    # utility['ZIP'] = utility['ZIP'].astype(str)
    # utility['ZIP'] = utility['ZIP'].str.lstrip('0')

    # utility['ZIP'].unique()
    # utility.columns

    # df = df.merge(utility, on="ZIP", how="left")

    # df.sample(10)

    # print(df['ZIP'].nunique())
    # print(df['Year'].nunique())

    # df.drop_duplicates(subset=['ZIP', 'Year'], inplace=True)
    return True
####################################
# Appreciation Calculation
print("Calculating appreciation...")

df['Year'] = pd.to_datetime(df['Year'], format='%Y')

# df['CountyName'].nunique()
# df['Year'].nunique()

# Keep only relevant columns to merge
base = df[['CountyName', 'Year', 'Value']].copy()

def calculate_future_appreciation(base,n):
    """
    Calculate future appreciation for a given number of years.

    We won't use this function in the final code, but it's here for reference.
    """
    # for n in [1, 5, 10]:
#     # Create a future year column
#     base_future = base.copy()
#     base_future['Year'] = base_future['Year'] - pd.DateOffset(years=n)
#     base_future = base_future.rename(columns={'Value': f'{n}y_Future_Value'})

#     # Merge on ZIP and exact year offset
#     df = df.merge(base_future, on=['CountyName', 'Year'], how='left')

#     # Calculate appreciation only where future value is valid
#     df[f'{n}yr_Appreciation%'] = (
#         ((df[f'{n}y_Future_Value'] - df['Value']) 
#         / df['Value']) 
#     * 100
#     ).round(2)
    
    return df

base_future = base.copy()
base_future['Year'] = base_future['Year'] - pd.DateOffset(years=5)

# We bring the year back 5 years for merging the future

base_future = base_future.rename(columns={'Value': '5y_Future_Value'})
df = df.merge(base_future, on=['CountyName', 'Year'], how='left')
df[f'5yr_Appreciation%'] = (((df['5y_Future_Value'] - df['Value']) / df['Value']) * 100).round(2)

# df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df['Year'] = df['Year'].dt.strftime('%Y')
df['Year'] = df['Year'].astype(int)
# df['Year'].sample(10)

save = df.copy()
df = save.copy()

# save['CountyName'].nunique()
# save['Year'].nunique()

# df.sample(10)
mean_5yr_appreciation = df['5yr_Appreciation%'].mean().round(2)
print('mean 5yr appreciation: ' + str(mean_5yr_appreciation) + '%') 

# bar chart nulls by year
null_df = df[df.isnull().any(axis=1)]

plt.figure(figsize=(12, 6))
sns.countplot(data=null_df, x='Year', palette='viridis')
plt.title('Count of Null Values by Year')
plt.xticks(rotation=45)
plt.xlabel('Year')
plt.ylabel('Count of Null Values')
plt.tight_layout()
plt.show()

df = df[df['Year'] <= 2020]

####################################
# === SAIPE API ===

print('SAIPE DATA FETCHING...')

GOV_API = ""

# Map state abbreviations to FIPS for merging
# We need FIPS codes to merge SAIPE data with our zilow data.
state_abbr_to_fips = {
    'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08',
    'CT': '09', 'DE': '10', 'FL': '12', 'GA': '13', 'HI': '15', 'ID': '16',
    'IL': '17', 'IN': '18', 'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22',
    'ME': '23', 'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
    'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34',
    'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39', 'OK': '40',
    'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45', 'SD': '46', 'TN': '47',
    'TX': '48', 'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54',
    'WI': '55', 'WY': '56', 'DC': '11'
}

df['State'] = df['State'].str.upper()
df['state_fips'] = df['State'].map(state_abbr_to_fips)

# Load county FIPS reference
fips_ref = pd.read_csv("https://raw.githubusercontent.com/kjhealy/fips-codes/master/state_and_county_fips_master.csv")
fips_ref.to_csv('fips.csv', index=False, header=True)
fips_ref['state_name'] = fips_ref['state'].str.lower().str.strip()
fips_ref['county_name'] = fips_ref['name'].str.lower().str.strip()
df['State'] = df['State'].str.lower().str.strip()
df['county'] = df['CountyName'].str.lower().str.strip()

# Merge to get full 5-digit FIPS
df = df.merge(
    fips_ref[['state_name', 'county_name', 'fips']],
    how='left',
    left_on=['State', 'county'],
    right_on=['state_name', 'county_name']
)

# Extract 3-digit county FIPS
df['county_fips'] = df['fips'].astype(str).str[-3:].str.zfill(3)
len(df)

###########

def SAIPE_CALL():
    """
    This function is a placeholder for the SAIPE API call. 
    Since we're saving the data to a CSV file, we won't actually call the API here.
    The code is commented out to prevent execution.
    Just to save space.
    """

    # # === Call SAIPE API ===
    # base_url = "https://api.census.gov/data/timeseries/poverty/saipe"
    # variables = "NAME,SAEMHI_PT,SAEPOVALL_PT,SAEPOVRTALL_PT"

    # saipe_data = []

    # api_requests = df[['county_fips','county', 'state_fips', 'Year']].drop_duplicates().copy()
    # length = len(api_requests)
    # api_requests.sample(10)
    # i = 0
    # for _, row in api_requests.iterrows():
    #     i += 1
    #     params = {
    #         "get": variables,
    #         "for": f"county:{row['county_fips']}",
    #         "in": f"state:{row['state_fips']}",
    #         "YEAR": str(row['Year'])
    #     }
    #     response = requests.get(base_url, params=params)
    #     print(str((i/length)*100) + "%")
    #     if response.status_code == 200:
    #         json_data = response.json()
    #         headers = json_data[0]
    #         values = json_data[1]
    #         saipe_data.append(dict(zip(headers, values)))
    #     else:
    #         print(f"Failed for {row['county']}, ({row['Year']})")
    #     time.sleep(0.5)

# saipe_df = pd.DataFrame(saipe_data)

# saipe_df.to_csv('saipe.csv', index=False, header=True)
saipe_df = pd.read_csv('saipe.csv')

# Ensure consistent format
saipe_df['YEAR'] = saipe_df['YEAR'].astype(int)
saipe_df['state'] = saipe_df['state'].astype(str).str.zfill(2)
saipe_df['county'] = saipe_df['county'].astype(str).str.zfill(3)

saipe_df.rename(columns={
    'SAEMHI_PT': 'County median Income',
    'SAEPOVALL_PT': 'County Poverty Count',
    'SAEPOVRTALL_PT': 'County Poverty Rate'
}, inplace=True)

saipe_df.sort_values(by=['state', 'county', 'YEAR'], inplace=True)
saipe_df['Income Rank'] = saipe_df.groupby(['state', 'YEAR'])['County median Income'] \
                                  .rank(ascending=False, method='dense')
saipe_df['Poverty Rank'] = saipe_df.groupby(['state', 'YEAR'])['County Poverty Rate'] \
                                  .rank(ascending=True, method='dense')


saipe_df['Income YoY Change %'] = saipe_df.groupby(['state', 'county'])['County median Income'] \
                                          .pct_change() * 100

# group by year and avg
saipe_group_df = saipe_df.groupby(['state', 'YEAR']).agg({
    'County median Income': 'median',
    'County Poverty Count': 'median',
    'County Poverty Rate': 'median'
}).reset_index()

# rename to county poverty to state poverty
saipe_group_df.rename(columns={
    'County median Income': 'State median Income',
    'County Poverty Count': 'State Poverty Count',
    'County Poverty Rate': 'State Poverty Rate'
}, inplace=True)

saipe_group_df.sample(1)

saipe_df = saipe_df.merge(
    saipe_group_df,
    how='left',
    left_on=['state', 'YEAR'],
    right_on=['state', 'YEAR'],
    suffixes=('', '_state')
)

saipe_df['County v state income %'] = (
    (saipe_df['County median Income'] - saipe_df['State median Income'])
    / saipe_df['State median Income'] * 100
).round(2)

saipe_df['County v state poverty %'] = (
    (saipe_df['County Poverty Rate'] - saipe_df['State Poverty Rate'])
    / saipe_df['State Poverty Rate'] * 100
).round(2)

saipe_df.sort_values(by=['state', 'county', 'YEAR'], inplace=True)
saipe_df.sample(10)

# Merge SAIPE data into main DataFrame
df = df.merge(
    saipe_df,
    how='left',
    left_on=['state_fips', 'county_fips', 'Year'],
    right_on=['state', 'county', 'YEAR']
)

print('Census data merged successfully.')

####################################

####################################
# Encoding and Cleaning

print("Cleaning and encoding...")

# df.columns

df.drop(columns=['5y_Future_Value','fips','county_fips','NAME','state_fips','state_name','county_name','county_y','county_x','state','YEAR'], inplace=True)

df = df[df['Year'] >= 2000]
df = df[df['Year'] <= 2020]

df.reset_index(drop=True, inplace=True)

raw_data = df.copy()
raw_data.columns

print(raw_data['CountyName'].nunique())

# Identify categorical columns for encoding
# We will one-hot encode categorical variables with fewer than 10 unique values (State)

# categorical_cols = ['State','Metro','City','CountyName']
categorical_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        categorical_cols.append(col)

encode = []
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values")
    if df[col].nunique() < 10:
        print('For one hot encoding: ' + col)
        categorical_cols.remove(col)
        encode.append(col)

print('One hot encoding: ', encode)

print('Dropping columns: ', categorical_cols)
df.drop(columns=categorical_cols, inplace=True)

df.columns

# One Hot Encoding:
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_cols = encoder.fit_transform(df[encode])
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(encode))
encoded_df.sample(10)
df = df.drop(columns=encode)
df = pd.concat([df, encoded_df], axis=1)

# Label Encoding:
# N/A will not be used as we have one hot encoded the categorical variables
# Since they are not ordinal, label encoding is not appropriate here.

# label_encoders = {}  # Store encoders to reuse later
# for col in categorical_cols:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col]) 
#     # test[col] = le.transform(test[col])
#     label_encoders[col] = le  # Store encoder in case it's needed later

################################
# %%
df.columns
 
# # Stuff with little impact on the model :(
# df.drop(columns=['Blue%', 'ValMean', 'SizeRank', 'City', 'ind_rate_y', 'Entropy', 'SatMean', 'State AVG Poverty %', 'State', 'Green%', 'State median Income'], inplace=True)

# KNN-based imputation
imputer = KNNImputer(n_neighbors=5, weights='distance')

# Fit and transform
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# drop nulls for 5yr appreciation (target) (hopefully anything past 2020..)
df.dropna(subset=['5yr_Appreciation%'], inplace=True)

df_clean = df.copy()
# df_clean.corr()
df_clean.sample(10)
# df_clean['CountyName'].nunique()

CUTOFF_YEAR = 2020

# Train/test split based only on Year
train = df_clean[df_clean['Year'] < CUTOFF_YEAR]
test = df_clean[df_clean['Year'] >= CUTOFF_YEAR]

train.sample(10)
test.sample(10)

# Target variable
y_train = train['5yr_Appreciation%']
y_test = test['5yr_Appreciation%']

X_train = train.drop(columns=['5yr_Appreciation%', 'Year'])
X_test = test.drop(columns=['5yr_Appreciation%', 'Year'])

X_test.sample(10)

# test['Year'].nunique()

# print(X_test['county'].unique())
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Hyperparameter tuning...')

model = CatBoostRegressor()

param_grid = {
    'depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1],
    'iterations': [100, 150, 200, 300]
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

print("Predictions complete")

print("R² Score:", r2_score(y_test, y_pred))
# print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
# print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# ################################
# Get the rows in raw_data corresponding to test set
test_meta = raw_data[raw_data['Year'] >= CUTOFF_YEAR].reset_index(drop=True)

# Sanity check – make sure lengths match
assert len(test_meta) == len(y_pred), "Mismatch between test metadata and predictions."

test_meta['Predicted_5yr_Appreciation%'] = y_pred

print(test_meta[['CountyName', 'Predicted_5yr_Appreciation%']].head())

# ################################

import matplotlib.pyplot as plt

# Get feature importances
importances = best_model.get_feature_importance()
features = X_train.columns

# Sort features by importance
sorted_indices = importances.argsort()[::-1] 
sorted_features = features[sorted_indices]
sorted_importances = importances[sorted_indices]

# Print feature importances
for feature, importance in zip(sorted_features, sorted_importances):
    if importance < .1:  # Only print features with importance > 0.01
        # print(f"{feature}: {importance:.4f}")
        print(f"'{feature}'", end=", ")
        # print(feature, end=", ")

# Plot
plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importances)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis() 
plt.tight_layout()

plt.show()

# residuals = y_test - y_pred
################################

print("R² Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

################################
# Plotly Map Visualization

# fig = px.scatter_geo(
#     test_meta[['CountyName', 'Predicted_5yr_Appreciation%','LNG','LAT']],
#     lat='LAT', lon='LNG',
#     color='Predicted_5yr_Appreciation%',
#     color_continuous_scale='RdBu',
#     range_color=[30, 100],  # adjust to your residual spread
#     hover_name='CountyName',
#     scope='usa',
#     title='Prediction by County Coordinates'
# )

# fig.update_traces(marker=dict(size=6, opacity=0.7))
# fig.update_layout(geo=dict(showland=True, landcolor='lightgray'))
# fig.show()

################################
# Train/Test Distribution

# sns.histplot(y_train, kde=True, label='Train', color='blue')
# sns.histplot(y_test, kde=True, label='Test', color='red')
# plt.legend()
# plt.title("Target (5yr Appreciation%) Distribution")
# plt.show()

# # ################################
# Folium Map Visualization

# # Filter and prep your DataFrame
# df_map = test_meta[['CountyName', 'Predicted_5yr_Appreciation%', 'LAT', 'LNG']].dropna()
# df_map = df_map[df_map['Predicted_5yr_Appreciation%'].between(30, 100)]

# # Create color scale
# color_scale = cm.linear.RdBu_11.scale(30, 100).to_step(index=[30, 40, 50, 60, 70, 80, 90, 100])

# # Create base map
# m = folium.Map(location=[46, -120], zoom_start=6, tiles='CartoDB positron')

# # Add circle markers
# for i, row in df_map.iterrows():
#     CircleMarker(
#         location=(row['LAT'], row['LNG']),
#         radius=5,
#         color=color_scale(row['Predicted_5yr_Appreciation%']),
#         fill=True,
#         fill_color=color_scale(row['Predicted_5yr_Appreciation%']),
#         fill_opacity=0.7,
#         popup=folium.Popup(f"{row['CountyName']}: {row['Predicted_5yr_Appreciation%']:.2f}%", parse_html=True)
#     ).add_to(m)

# color_scale.caption = 'Predicted 5-Year Appreciation (%)'
# color_scale.add_to(m)

# # Save map
# m.save('prediction_map_folium.html')

# ################################

# This model needs more time series data to be accurate

# # The image data is also not very useful. Cool idea. Could be optimized with zoom level maybe.

# #  Some type of deep learning model could be used to extract features from the images.

# # County is important, as is city and metro.

# # I think bundled utility rate was a good feature, also poverty estimate.

# # Another idea would be to make this a classification problem and predict if the appreciation is above or below a certain threshold.

# # %%
