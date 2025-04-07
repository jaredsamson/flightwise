import pyarrow.dataset as ds
import pyarrow.compute as pc
import pandas as pd

def load_filtered_sample(parquet_path="sample_itineraries.parquet", sample_fraction=0.05, target_sample_size=100_000):
    dataset = ds.dataset(parquet_path, format="parquet")
    scanner = dataset.scanner(columns=[
        "searchDate", "flightDate", "startingAirport", "destinationAirport",
        "segmentsAirlineName", "segmentsCabinCode", "isNonStop", "totalFare"
    ])

    sampled_batches = []
    total_rows = 0

    for batch in scanner.to_batches():
        df_batch = batch.to_pandas()

        # Filter + clean
        df_batch['searchDate'] = pd.to_datetime(df_batch['searchDate'], errors='coerce')
        df_batch['flightDate'] = pd.to_datetime(df_batch['flightDate'], errors='coerce')
        df_batch['daysUntilFlight'] = (df_batch['flightDate'] - df_batch['searchDate']).dt.days

        df_batch = df_batch[
            (df_batch['isNonStop'] == True) &
            (~df_batch['segmentsAirlineName'].str.contains(r'\|\|', na=False)) &
            (df_batch['daysUntilFlight'].between(0, 180)) &
            (df_batch['totalFare'].between(50, 1000))
        ].dropna(subset=['searchDate', 'flightDate', 'totalFare'])

        if len(df_batch) > 0:
            df_sample = df_batch.sample(frac=sample_fraction, random_state=42)
            sampled_batches.append(df_sample)
            total_rows += len(df_sample)

            if total_rows >= target_sample_size:
                break

    return pd.concat(sampled_batches, ignore_index=True)
