import pandas as pd

#* Step 1: data cleaning  with the objective of ensuring we only process valid and consistent data

# load data
df = pd.read_csv('data/events.csv')

# delete exact duplicates (if the system have sent the same event twice we drop it)
df = df.drop_duplicates() 

# convert event_time to real datetime so we can manipulate dates
df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')

# filtering: delete nan
df = df.dropna(subset=['user_id', 'event_time'])

#* Step 2: feature engineering - creating new features from existing data

# get only the date part of event_time 
df['event_date'] = df['event_time'].dt.date

# group by date and count unique users per day and total events per day
df_daily = df.groupby('event_date').agg({'user_id': 'nunique', 'event_type': 'count'}).reset_index()

#* Step 3: preparing the final dataset for analysis or modeling

# rename columns for clarity
df_daily = df_daily.rename(columns={'user_id': 'unique_users', 'event_type': 'total_events'})

# order by date
df_daily = df_daily.sort_values(by='event_date')

# save the processed data (optional, and can be in different formats as needed)
df_daily.to_csv('data/daily_user_events.csv', index=False)