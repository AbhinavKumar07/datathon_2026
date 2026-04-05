
import pandas as pd
import numpy as np

#Problem: How can we help new (small buisness) restaurant owners open up their business in areas that are optimized for traffic

#Get datasets --> df
df_consumer_data = pd.read_csv('./ConsumerData.csv')
df_migration_data = pd.read_csv('./OC2025_ZIPMigration.csv')
df_us_address_Data = pd.read_csv('./USAddressData.csv')
df_zip_data = pd.read_csv('./ZipData.csv')
df_property_val_data = pd.read_csv('./property_value.csv')


#df_consumer_data.head()
#df_migration_data.head()
#df_us_address_Data.head()
#df_zip_data.head()
#df_property_val_data.sample(10)

# Fill NaN values for 'Y'/'N' type columns with 'N'
consumer_data_columns = [
    'Charitable', 'Health', 'Political', 'Religious', 'Veteran', 'SingleParent',
    'GrandChildren', 'CatOwner', 'DogOwner', 'CreditCardUser', 'SelfImprovement',
    'MusicCollector', 'MovieCollector', 'Photography', 'AutoWork', 'Fishing',
    'CampingHiking', 'HuntingShooting', 'Gardening', 'EnvironmentalIssues',
    'HomeImprovement', 'HomeImprovementDIY', 'OutdoorsGrouping',
    'InvestmentsForeign', 'BeautyCosmetics', 'TVCable', 'WirelessCellularPhoneOwner',
    'EducationOnline'
]
for col in consumer_data_columns:
    df_consumer_data[col] = df_consumer_data[col].fillna('N')

# Fill NaN values for MaritalStatus with 'Unknown'
df_consumer_data['MaritalStatus'] = df_consumer_data['MaritalStatus'].fillna('Unknown')

# Fill NaN values for OwnerRenter with 'Unknown' or the mode
# Let's check value counts first to decide for OwnerRenter
# print(df_consumer_data['OwnerRenter'].value_counts())
# Assuming 'O' (Owner) is the mode or a reasonable default for missing values, or 'Unknown'
df_consumer_data['OwnerRenter'] = df_consumer_data['OwnerRenter'].fillna('Unknown')


# Fill NaN values for numerical columns with their median
# HomePurchaseDate, NumberOfChildren, HouseholdSize, NetWorth, VehicleKnownOwnedNumber
numeric_columns = [
    'HomePurchaseDate', 'NumberOfChildren', 'HouseholdSize', 'NetWorth',
    'VehicleKnownOwnedNumber'
]
for col in numeric_columns:
    df_consumer_data[col] = df_consumer_data[col].fillna(df_consumer_data[col].median())


print(df_consumer_data.info())


#Column index numbers

column_indices = {}
for col_name in consumer_data_columns:
    if col_name in df_consumer_data.columns:
        column_indices[col_name] = df_consumer_data.columns.get_loc(col_name)
    else:
        column_indices[col_name] = 'Not Found'

for col, index in column_indices.items():
    print(f"Column '{col}': Index {index}")


print(df_consumer_data.info())
print(df_consumer_data.head())

print(df_migration_data.info())
print(df_migration_data.head())

# Check for duplicate rows
duplicates_migration = df_migration_data.duplicated().sum()
print(f"Number of duplicate rows in df_migration_data: {duplicates_migration}")

# If duplicates exist, you might choose to drop them:
# if duplicates_migration > 0:
#     df_migration_data.drop_duplicates(inplace=True)
#     print(f"Duplicates removed. New number of rows: {len(df_migration_data)}")

# Create a 'NetMigration' column (MovesIntoZip - MovesOutOfZip)
# This indicates population growth or decline in a zip code, relevant for restaurant traffic.
df_migration_data['NetMigration'] = df_migration_data['MovesIntoZip'] - df_migration_data['MovesOutOfZip']

print('\nDataFrame info after adding NetMigration column:')
print(df_migration_data.info())
print('\nFirst 5 rows of df_migration_data with NetMigration:')
print(df_migration_data.head())

# Check consistency of TotalMoves
df_migration_data['CalculatedTotalMoves'] = df_migration_data['MovesOutOfZip'] + df_migration_data['MovesIntoZip']
mismatched_total_moves = df_migration_data[df_migration_data['TotalMoves'] != df_migration_data['CalculatedTotalMoves']]

if not mismatched_total_moves.empty:
    print("Inconsistencies found in 'TotalMoves' column:")
    display(mismatched_total_moves[['MovesOutOfZip', 'MovesIntoZip', 'TotalMoves', 'CalculatedTotalMoves']])
else:
    print("No inconsistencies found in 'TotalMoves' column.")

# Check consistency of PctLeave and PctMoveIn (should sum up to ~100)
df_migration_data['CalculatedPctTotal'] = df_migration_data['PctLeave'] + df_migration_data['PctMoveIn']
mismatched_percentages = df_migration_data[abs(df_migration_data['CalculatedPctTotal'] - 100) > 0.01] # Allowing for small floating point discrepancies

if not mismatched_percentages.empty:
    print("\nInconsistencies found in 'PctLeave' and 'PctMoveIn' percentages:")
    display(mismatched_percentages[['PctLeave', 'PctMoveIn', 'CalculatedPctTotal']])
else:
    print("\nNo significant inconsistencies found in 'PctLeave' and 'PctMoveIn' percentages.")

# Drop the temporary calculated columns
df_migration_data.drop(columns=['CalculatedTotalMoves', 'CalculatedPctTotal'], inplace=True, errors='ignore')

print(df_zip_data.info())
print(df_zip_data.head())

# Drop columns with all NaN values
df_us_address_Data.drop(columns=['StreetPreDirection', 'StreetPostDirection'], inplace=True)

# Fill NaN values for object/categorical columns with 'Unknown'
object_columns_to_fill = ['BaseMAK', 'Suite', 'StreetSuffix', 'SuiteType', 'SuiteNumber', 'PlaceCode']
for col in object_columns_to_fill:
    # Only fill if the column exists and is of object type
    if col in df_us_address_Data.columns and df_us_address_Data[col].dtype == 'object':
        df_us_address_Data[col] = df_us_address_Data[col].fillna('Unknown')
    # Handle BaseMAK which is float but often acts as an ID, converting to object then filling
    elif col == 'BaseMAK' and df_us_address_Data[col].dtype == 'float64':
        df_us_address_Data['BaseMAK'] = df_us_address_Data['BaseMAK'].fillna(0).astype(int).astype(str) # Fill with 0, convert to int, then to string for 'Unknown' consistency
    elif col == 'PlaceCode' and df_us_address_Data[col].dtype == 'float64':
        df_us_address_Data['PlaceCode'] = df_us_address_Data['PlaceCode'].fillna(0).astype(int).astype(str)

# Fill NaN values for numerical columns with their median
# ElevationInMeters is the primary numerical column with NaNs
if 'ElevationInMeters' in df_us_address_Data.columns:
    df_us_address_Data['ElevationInMeters'] = df_us_address_Data['ElevationInMeters'].fillna(df_us_address_Data['ElevationInMeters'].median())

print(df_us_address_Data.info())
print(df_us_address_Data.head())


# Fill NaN values for 'LastLineIndicator' with 'Unknown'
df_zip_data['LastLineIndicator'] = df_zip_data['LastLineIndicator'].fillna('Unknown')

# Fill NaN values for numerical columns with their median
numeric_zip_columns = [
    'MedianHouseholdIncome', 'PerCapitaIncome', 'MedianHomeValue', 'MedianAge',
    'MedianAgeMale', 'MedianAgeFemale'
]
for col in numeric_zip_columns:
    df_zip_data[col] = df_zip_data[col].fillna(df_zip_data[col].median())

print(df_zip_data.info())
print(df_zip_data.head())

# Step 1: Standardize 'Zipcode' column name in df_consumer_data to 'ZipCode'
df_consumer_data.rename(columns={'Zipcode': 'ZipCode'}, inplace=True)

# Step 2: Aggregate df_consumer_data by ZipCode
# We'll calculate various summary statistics for consumer demographics at the zip code level.
# This will prevent an overly large dataset when merging with other zip-level data.

agg_consumer_data = df_consumer_data.groupby('ZipCode').agg(
    ConsumerCount=('RecordID', 'count'),
    AvgHouseholdSize=('HouseholdSize', 'mean'),
    AvgNumberOfChildren=('NumberOfChildren', 'mean'),
    AvgNetWorth=('NetWorth', 'mean'),
    AvgVehicleKnownOwned=('VehicleKnownOwnedNumber', 'mean'),
    # Count 'Y' for binary categorical features
    PctCharitable=('Charitable', lambda x: (x == 'Y').mean() * 100),
    PctHealth=('Health', lambda x: (x == 'Y').mean() * 100),
    PctPolitical=('Political', lambda x: (x == 'Y').mean() * 100),
    PctReligious=('Religious', lambda x: (x == 'Y').mean() * 100),
    PctVeteran=('Veteran', lambda x: (x == 'Y').mean() * 100),
    PctSingleParent=('SingleParent', lambda x: (x == 'Y').mean() * 100),
    PctGrandChildren=('GrandChildren', lambda x: (x == 'Y').mean() * 100),
    PctCatOwner=('CatOwner', lambda x: (x == 'Y').mean() * 100),
    PctDogOwner=('DogOwner', lambda x: (x == 'Y').mean() * 100),
    PctCreditCardUser=('CreditCardUser', lambda x: (x == 'Y').mean() * 100)
).reset_index()

print("Aggregated df_consumer_data info:")
print(agg_consumer_data.info())
print("\nAggregated df_consumer_data head:")
print(agg_consumer_data.head())

# Start with df_zip_data as the base, as it contains a broad range of demographic and geographic information.
merged_df = df_zip_data.copy()

# Merge agg_consumer_data
merged_df = pd.merge(merged_df, agg_consumer_data, on='ZipCode', how='left')

# Merge df_migration_data
merged_df = pd.merge(merged_df, df_migration_data, on='ZipCode', how='left', suffixes=('_zip', '_migration'))

# Merge agg_property_data
merged_df = pd.merge(merged_df, agg_property_data, on='ZipCode', how='left')

# Display the info and head of the merged DataFrame
print("Merged DataFrame info:")
print(merged_df.info())
print("\nMerged DataFrame head:")
print(merged_df.head())

# Aggregating df_property_val_data by ZipCode
# Calculate average property values for each zip code.

agg_property_data = df_property_val_data.groupby('ZipCode').agg(
    PropertyCount=('RecordId', 'count'),
    AvgFinalValue=('FinalValue', 'mean'),
    AvgHighValue=('HighValue', 'mean'),
    AvgLowValue=('LowValue', 'mean')
).reset_index()

print("Aggregated df_property_val_data info:")
print(agg_property_data.info())
print("\nAggregated df_property_val_data head:")
print(agg_property_data.head())

print(df_property_val_data.info())
print(df_property_val_data.head())

# Identify numerical columns with NaNs that should be filled with median
numerical_cols_with_nan = [
    'ConsumerCount', 'AvgHouseholdSize', 'AvgNumberOfChildren', 'AvgNetWorth',
    'AvgVehicleKnownOwned', 'PctCharitable', 'PctHealth', 'PctPolitical',
    'PctReligious', 'PctVeteran', 'PctSingleParent', 'PctGrandChildren',
    'PctCatOwner', 'PctDogOwner', 'PctCreditCardUser',
    'FIPS', 'MovesOutOfZip', 'MovesIntoZip', 'TotalMoves', 'PctLeave', 'PctMoveIn',
    'ZipCode_Latitude', 'ZipCode_Longitude', 'NetMigration',
    'PropertyCount', 'AvgFinalValue', 'AvgHighValue', 'AvgLowValue' # Added property columns
]

for col in numerical_cols_with_nan:
    if col in merged_df.columns:
        # Fill with 0 for counts and moves if absence implies zero activity
        # For averages and percentages, median is more appropriate.
        # Let's use median as a general strategy consistent with previous steps.
        merged_df[col] = merged_df[col].fillna(merged_df[col].median())

# Identify categorical columns with NaNs that should be filled with 'Unknown'
categorical_cols_with_nan = [
    'State_migration' # This was derived from df_migration_data
]

for col in categorical_cols_with_nan:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].fillna('Unknown')

# Drop specified columns as requested by the user
columns_to_drop = ['PctCatOwner', 'PctDogOwner', 'PctHealth']
merged_df.drop(columns=columns_to_drop, errors='ignore', inplace=True)


print("Merged DataFrame info after handling NaNs and dropping columns:")
print(merged_df.info())
print("\nMerged DataFrame head after handling NaNs and dropping columns:")
print(merged_df.head())

merged_df.head()

print("Number of remaining missing values per column in merged_df:")
print(merged_df.isnull().sum()[merged_df.isnull().sum() > 0])

# Calculate population density
merged_df['PopulationDensity'] = merged_df['TotalPopulation'] / (merged_df['ResidentialDeliveries'] + merged_df['BusinessDeliveries'])

# Calculate average income per person
merged_df['AvgIncomePerPerson'] = merged_df['PerCapitaIncome']

# Calculate the ratio of business to residential deliveries, which might indicate commercial activity
merged_df['BusinessResidentialRatio'] = merged_df['BusinessDeliveries'] / (merged_df['ResidentialDeliveries'] + 1) # Add 1 to avoid division by zero

# Calculate the most common race
population_columns = [
    'PopulationWhite', 'PopulationAfricanAmerican', 'PopulationAmericanIndianAlaskaNative',
    'PopulationAsian', 'PopulationHispanic', 'PopulationPacificIslander', 'PopulationOther', 'PopulationMultipleRace'
]
# Find the column name with the maximum value for each row among the population columns
merged_df['MostCommonRace'] = merged_df[population_columns].idxmax(axis=1).str.replace('Population', '')

# Add a feature for relative wealth based on median household income and median home value
merged_df['WealthIndex'] = (merged_df['MedianHouseholdIncome'] + merged_df['MedianHomeValue']) / 2

# Education level as a potential indicator of customer base sophistication
merged_df['HigherEducationRatio'] = (merged_df['EducationBachelorsDegree'] + merged_df['EducationAssociatesDegree']) / merged_df['TotalPopulation']

print("Merged DataFrame info after Feature Engineering:")
print(merged_df.info())
print("\nFirst 5 rows of merged_df with new features:")
print(merged_df[['ZipCode', 'PopulationDensity', 'AvgIncomePerPerson', 'BusinessResidentialRatio', 'MostCommonRace', 'WealthIndex', 'HigherEducationRatio']].head())

merged_df.info()


# Drops the first column (index 0)

#commented cuz it will cause an keyError now
#merged_df = merged_df.drop(columns = ['PctCatOwner' , 'PctDogOwner' , 'PctCharitable' , 'PctHealth' , 'PctReligious' , 'PctVeteran'], axis=1)
#merged_df = merged_df.drop(columns = ['MedianHouseholdIncome' , 'MedianHomeValue'], axis=1)
#merged_df = merged_df.drop(columns = ['EducationBachelorsDegree','EducationAssociatesDegree' , 'BusinessDeliveries' , 'ResidentialDeliveries'], axis=1)
#merged_df = merged_df.drop(columns = ['PopulationWhite' , 'PopulationAfricanAmerican' , 'PopulationAmericanIndianAlaskaNative' , 'PopulationAsian' , 'PopulationHispanic' , 'PopulationPacificIslander' , 'PopulationOther' , 'PopulationMultipleRace'], axis=1)
#merged_df = merged_df.drop(columns = ['EducationNinthGradeOrLess' , 'EducationSomeHighSchool'  , 'EducationSomeCollegeWithoutDiploma'], axis=1)
#merged_df = merged_df.drop(columns = ['ResidentialPOBoxes' , 'BusinessPOBoxes' , 'DominantAreaCode' , 'MedianAgeMale' , 'MedianAgeFemale' , 'EducationHighSchoolGraduate', 'AvgVehicleKnownOwned'], axis=1)
#merged_df = merged_df.drop(columns = ['ZipCode_Latitude' , 'ZipCode_Longitude'], axis=1)
#merged_df = merged_df.drop(columns = ['AvgHighValue' , 'AvgLowValue' , 'PropertyCount' , 'PctVeteran' , 'PctReligious'], axis=1)
#merged_df = merged_df.drop(columns = ['Latitude' , 'Longitude' , 'PerCapitaIncome' , 'AvgNumberOfChildren' , 'AvgNetWorth' , 'PctCharitable'], axis=1)
#merged_df = merged_df.drop(columns = ['PctLeave' , 'PctMoveIn' ,'TotalMoves' , 'PctCreditCardUser' , 'MovesOutOfZip' , 'MovesIntoZip'], axis=1)
#merged_df = merged_df.drop(columns = ['ConsumerCount' , 'PctGrandChildren' , 'State_migration' , 'LastLineIndicator'], axis=1)
#merged_df = merged_df.drop(columns = ['FIPS' , 'PctSingleParent'], axis=1)
#merged_df = merged_df.drop(columns = ['PctPolitical'], axis=1)
#merged_df = merged_df.drop(columns = ['AvgIncomePerPerson'], axis=1)

merged_df.info()


merged_df['SuccessScore'] = 0

# 1. Total Population (higher is more successful) - top 33%
pop_threshold = merged_df['TotalPopulation'].quantile(0.66)
merged_df.loc[merged_df['TotalPopulation'] > pop_threshold, 'SuccessScore'] += 1

# 2. Median Age (age = 13 to 50 is the most successful)
merged_df.loc[(merged_df['MedianAge'] >= 13) & (merged_df['MedianAge'] <= 50), 'SuccessScore'] += 1

# 3. Avg Household Size (higher is more successful) - top 33%
household_size_threshold = merged_df['AvgHouseholdSize'].quantile(0.66)
merged_df.loc[merged_df['AvgHouseholdSize'] > household_size_threshold, 'SuccessScore'] += 1

# 4. Net Migration (higher is more successful) - top 33%
net_migration_threshold = merged_df['NetMigration'].quantile(0.66)
merged_df.loc[merged_df['NetMigration'] > net_migration_threshold, 'SuccessScore'] += 1

# 5. Avg Final Value ($1M+ house prices is most favorable)
merged_df.loc[merged_df['AvgFinalValue'] >= 1000000, 'SuccessScore'] += 1

# 6. Population Density (higher is more favorable) - top 33%
density_threshold = merged_df['PopulationDensity'].quantile(0.66)
merged_df.loc[merged_df['PopulationDensity'] > density_threshold, 'SuccessScore'] += 1

# 7. Business Residential Ratio (higher is favorable) - top 33%
ratio_threshold = merged_df['BusinessResidentialRatio'].quantile(0.66)
merged_df.loc[merged_df['BusinessResidentialRatio'] > ratio_threshold, 'SuccessScore'] += 1

# 8. Most Common Race (white ppl unless the majority is a minority) - simplified for now to just 'White'
merged_df.loc[merged_df['MostCommonRace'] == 'White', 'SuccessScore'] += 1

# 9. Wealth Index (higher is favorable) - top 33%
wealth_threshold = merged_df['WealthIndex'].quantile(0.66)
merged_df.loc[merged_df['WealthIndex'] > wealth_threshold, 'SuccessScore'] += 1

# 10. Higher Education Ratio (higher is more favorable) - top 66%
edu_ratio_threshold = merged_df['HigherEducationRatio'].quantile(0.33)
merged_df.loc[merged_df['HigherEducationRatio'] > edu_ratio_threshold, 'SuccessScore'] += 1

# Create the binary 'Success' target variable based on a threshold of the SuccessScore
# Let's say a location is 'successful' if it meets at least 5 out of 10 criteria
success_criteria_count = 10 # Total number of criteria applied
success_threshold_score = success_criteria_count * 0.5 # e.g., 50% of criteria met
merged_df['Success'] = (merged_df['SuccessScore'] >= success_threshold_score).astype(int)

print("Distribution of SuccessScore:")
print(merged_df['SuccessScore'].value_counts())
print("\nDistribution of binary Success variable:")
print(merged_df['Success'].value_counts())
print("\nDataFrame head with new SuccessScore and Success columns:")
print(merged_df[['ZipCode', 'SuccessScore', 'Success']].head())

merged_df.head(500)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd # Import pandas if not already imported

# Store ZipCode before dropping it from features
zipcodes = merged_df['ZipCode']

# Define features (X) and target (y)
# Drop 'Success' as it's derived from SuccessScore and other non-predictive columns like RecordID
X = merged_df.drop(columns=['SuccessScore', 'Success', 'RecordID', 'ZipCode'])
y = merged_df['SuccessScore']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=np.number).columns

# Create a column transformer for one-hot encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep numerical columns as they are
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Start of added code for error diagnosis and handling ---

# Define min/max values for float32
float32_max = np.finfo(np.float32).max
float32_min = np.finfo(np.float32).min

# Process numerical columns to handle inf, NaNs, and values outside float32 range
for col in numerical_features:
    # 1. Handle infinite values: Replace inf/-inf with NaN
    if np.isinf(X_train[col]).any():
        print(f"Column '{col}' contains infinite values. Replacing with NaN.")
        X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan)
        X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)

    # 2. Fill NaN values (original NaNs or those converted from inf) with median
    if X_train[col].isnull().any():
        median_val = X_train[col].median()
        print(f"Column '{col}' contains NaN values. Filling with median: {median_val}")
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    # 3. Cap excessively large/small values to float32 limits
    if (X_train[col].max() > float32_max) or (X_train[col].min() < float32_min):
        print(f"Column '{col}' contains values outside float32 range (max: {X_train[col].max()}, min: {X_train[col].min()}). Capping to float32 limits.")
        X_train[col] = np.clip(X_train[col], float32_min, float32_max)
        X_test[col] = np.clip(X_test[col], float32_min, float32_max)

# end--------

# Create a pipeline with preprocessing and RandomForestRegressor
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', RandomForestRegressor(random_state=42))])

# Train the model
model_pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Create a DataFrame to compare actual and predicted SuccessScores
predictions_df = pd.DataFrame({
    'ZipCode': zipcodes.loc[y_test.index],
    'Actual_SuccessScore': y_test,
    'Predicted_SuccessScore': y_pred,
    'Actual_Score_0_100': y_test * 10,
    'Predicted_Score_0_100': y_pred * 10
})
display(predictions_df.head(100))

from sklearn.metrics import mean_absolute_error

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared: {r2:.2f}")

# Calculate and display mean and median for success scores
print("\nDescriptive Statistics for Success Scores:")
print("\nOriginal 0-10 Scale:")
print(predictions_df[['Actual_SuccessScore', 'Predicted_SuccessScore']].agg(['mean', 'median', 'min', 'max']))

print("\nScaled 0-100 Scale:")
print(predictions_df[['Actual_Score_0_100', 'Predicted_Score_0_100']].agg(['mean', 'median', 'min', 'max']))
     
