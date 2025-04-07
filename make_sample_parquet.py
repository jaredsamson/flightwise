import pyarrow.dataset as ds
import pandas as pd
import numpy as np

# === Step 1: Load + filter large sample from full dataset ===
dataset = ds.dataset("itineraries_snappy.parquet", format="parquet")
scanner = dataset.scanner(columns=[
    "searchDate", "flightDate", "startingAirport", "destinationAirport",
    "segmentsAirlineName", "segmentsCabinCode", "isNonStop", "totalFare"
])

df_list = []
rows_target = 500_000
rows_loaded = 0

for batch in scanner.to_batches():
    df = batch.to_pandas()

    # Clean and filter
    df['searchDate'] = pd.to_datetime(df['searchDate'], errors='coerce')
    df['flightDate'] = pd.to_datetime(df['flightDate'], errors='coerce')
    df['daysUntilFlight'] = (df['flightDate'] - df['searchDate']).dt.days

    df = df[
        (df['isNonStop'] == True) &
        (~df['segmentsAirlineName'].str.contains(r'\|\|', na=False)) &
        (df['daysUntilFlight'].between(0, 180)) &
        (df['totalFare'].between(50, 1000))
    ].dropna(subset=['searchDate', 'flightDate', 'totalFare'])

    df_list.append(df)
    rows_loaded += len(df)

    if rows_loaded >= rows_target:
        break

df_all = pd.concat(df_list, ignore_index=True)
print(f"✅ Loaded {len(df_all):,} filtered rows.")

# === Step 2: Route-Based Sampling ===
df_all['route'] = df_all['startingAirport'] + "_" + df_all['destinationAirport']
top_routes = df_all['route'].value_counts().head(50).index
df_filtered = df_all[df_all['route'].isin(top_routes)]

# Limit per route (preserves airline diversity)
df_sampled = df_filtered.groupby('route', group_keys=False).apply(
    lambda x: x.sample(n=min(2000, len(x)), random_state=42)
).reset_index(drop=True)

# === Step 3: Final sample to ~100k rows
final_sample = df_sampled.sample(n=min(100_000, len(df_sampled)), random_state=42)
final_sample.drop(columns=['route'], inplace=True)
final_sample.to_parquet("sample_itineraries.parquet")

print(f"✅ Saved {len(final_sample):,} rows from {final_sample['segmentsAirlineName'].nunique()} airlines across {final_sample[['startingAirport', 'destinationAirport']].drop_duplicates().shape[0]} routes.")
