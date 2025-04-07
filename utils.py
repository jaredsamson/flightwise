import pandas as pd

def load_airport_coords():
    coords_url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    coords_cols = ['Airport ID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Latitude', 'Longitude',
                   'Altitude', 'Timezone', 'DST', 'Tz', 'Type', 'Source']
    df = pd.read_csv(coords_url, header=None, names=coords_cols)
    df = df[(df['Country'] == 'United States') & (df['IATA'].str.len() == 3)]
    return df.set_index('IATA')[['Latitude', 'Longitude']].to_dict('index')
