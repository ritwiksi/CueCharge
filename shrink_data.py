import pandas as pd

# 1. Load the giant 100MB file
print("Reading giant file...")
df = pd.read_csv("sf_building_profiles.csv")

# 2. Keep only the first 1500 hours (roughly 2 months)
# This is plenty for a 30-day simulation + 24hr history
print("Shrinking data...")
df_lite = df.head(1500)

# 3. Save as a new, smaller file
df_lite.to_csv("sf_building_profiles_lite.csv", index=False)
print("Done! 'sf_building_profiles_lite.csv' created.")