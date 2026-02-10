import pandas as pd
import numpy as np

# Load the datasets
fact_performance = pd.read_csv('../docs/fact_performance.csv')
dim_campaign = pd.read_csv('../docs/dim_campaign.csv')
dim_source = pd.read_csv('../docs/dim_source.csv')
dim_date = pd.read_csv('../docs/dim_date.csv')

# Verification
print(f"Fact Table Size: {fact_performance.shape}")
print(f"Campaign Dimension Size: {dim_campaign.shape}")
print(f"Source Dimension Size: {dim_source.shape}")
print(f"Date Dimension Size: {dim_date.shape}")

# Left join fact table with dimensions
df_master = fact_performance.merge(dim_campaign, on='campaign_id', how='left')
df_master = df_master.merge(dim_source, on='source_id', how='left')
df_master = df_master.merge(dim_date, on='date_key', how='left')

# CPM: (Spend / Impressions) * 1000
df_master['CPM'] = np.where(df_master['impressions'] > 0, 
                            (df_master['spend'] / df_master['impressions']) * 1000, 
                            0)

# CTR: (Clicks / Impressions) * 100
df_master['CTR'] = np.where(df_master['impressions'] > 0, 
                            (df_master['clicks'] / df_master['impressions']) * 100, 
                            0)

# CPC: Spend / Clicks
df_master['CPC'] = np.where(df_master['clicks'] > 0, 
                            df_master['spend'] / df_master['clicks'], 
                            0)

# Conversion Rate (Conversions / Clicks * 100)
df_master['Conversion_Rate'] = np.where(df_master['clicks'] > 0, 
                                        (df_master['conversions'] / df_master['clicks']) * 100, 
                                        0)

# Check for nulls or anomalies
print(df_master.info())

# Export to CSV
df_master.to_csv('../docs/marketing_analytics_master.csv', index=False)
print("Export complete: ../docs/marketing_analytics_master.csv")