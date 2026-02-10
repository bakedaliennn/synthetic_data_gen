import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_marketing_star_schema(output_dir='../docs'):
    """
    Generates a synthetic marketing dataset structured as a Star Schema 
    suitable for BI tools (Tableau, PowerBI, Looker).
    
    The schema consists of:
    1. FACT_PERFORMANCE: Daily metrics (Imps, Clicks, Spend, Conv)
    2. DIM_DATE: Calendar metadata
    3. DIM_SOURCE: Platform/Channel metadata
    4. DIM_CAMPAIGN: Campaign objectives and hierarchy

    Args:
        output_dir (str): The directory where CSV files will be saved. 
                          Defaults to current directory.
    """
    print(f"Initializing Data Generation... Target Directory: {os.path.abspath(output_dir)}")

    # ==========================================
    # I. GENERATE DIMENSIONS (Metadata)
    # ==========================================
    
    # --- DIM_DATE (15 Months: Jan 2023 - Mar 2024) ---
    # Generating a daily grain for time-series analysis
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(456)] 
    dim_date = pd.DataFrame({'date': dates})
    
    # Integer keys are preferred in data warehousing for performance
    dim_date['date_key'] = dim_date['date'].dt.strftime('%Y%m%d').astype(int)
    dim_date['year'] = dim_date['date'].dt.year
    dim_date['month'] = dim_date['date'].dt.month
    dim_date['month_name'] = dim_date['date'].dt.strftime('%b')
    dim_date['quarter'] = dim_date['date'].dt.quarter
    dim_date['is_weekend'] = dim_date['date'].dt.dayofweek >= 5

    # --- DIM_SOURCE (8 Sources, Mapped to Channels) ---
    sources_data = [
        (1, 'Amazon Ad Server', 'Programmatic'), 
        (2, 'StackAdapt', 'Programmatic'), 
        (3, 'DV360', 'Programmatic'), 
        (4, 'Search Ads 360', 'Paid Search'), 
        (5, 'Bing Ads', 'Paid Search'), 
        (6, 'Facebook', 'Paid Social'), 
        (7, 'LinkedIn Ads', 'Paid Social'), 
        (8, 'Organic Search', 'Organic')
    ]
    dim_source = pd.DataFrame(sources_data, columns=['source_id', 'source_name', 'channel'])

    # --- DIM_CAMPAIGN (Mapped to Channel & Objective) ---
    # Campaigns are scoped to specific channels to ensure logical consistency 
    # (e.g., No "Search" campaigns appearing on "Facebook")
    campaign_config = [
        ("Business-focused zero tolerance", "Programmatic", "Brand Awareness"), 
        ("Profound intangible policy", "Programmatic", "Brand Awareness"),
        ("Networked value-added time-frame", "Programmatic", "Consideration"), 
        ("Persistent 24/7 attitude", "Paid Social", "Lead Gen"), 
        ("Centralized modular throughput", "Paid Social", "Conversion"), 
        ("Integrated dedicated contingency", "Paid Search", "Conversion"), 
        ("Automated uniform software", "Paid Search", "Lead Gen"), 
        ("Cross-platform static hierarchy", "Organic", "Traffic")
    ]
    
    campaign_rows = []
    id_counter = 1
    
    for name, scope, objective in campaign_config:
        # Create 2 Ad Sets per Campaign to enable hierarchy testing in Tableau
        for tier in ['_Tier1', '_Tier2']:
            campaign_rows.append({
                'campaign_id': id_counter,
                'campaign_name': name, 
                'ad_set_name': f"{name}{tier}", 
                'channel_scope': scope, # Helper for generation only
                'objective': objective
            })
            id_counter += 1
            
    dim_campaign = pd.DataFrame(campaign_rows)

    # ==========================================
    # II. GENERATE FACT TABLE 
    # ==========================================
    
    # Step A: Link Campaigns to Sources (Logical Join)
    schema_link = dim_campaign.merge(dim_source, left_on='channel_scope', right_on='channel')
    
    # Step B: Cross Join with Date to create the full time-series skeleton
    # ~20k rows (16 campaigns * relevant sources * 456 days)
    df_fact = schema_link.merge(dim_date[['date_key', 'is_weekend', 'year', 'month']], how='cross')
    N = len(df_fact)

    # Step C: Vectorized Metric Logic
    # ---------------------------------------------------------
    # 1. Seasonality Mask: Apply a multiplier to simulate lower traffic on weekends
    seasonality = np.where(df_fact['is_weekend'], 0.7, 1.1)

    # 2. Initialize Metric Arrays
    base_imps = np.zeros(N)
    ctrs = np.zeros(N)
    cpcs = np.zeros(N)

    # 3. Apply Channel-Specific Distributions (The "Realism" Layer)
    # We use boolean masking for performance rather than iterating rows
    
    # Programmatic: High Volume, Low CTR, Low CPC
    mask_prog = df_fact['channel'] == 'Programmatic'
    s_prog = mask_prog.sum()
    base_imps[mask_prog] = np.random.randint(5000, 15000, s_prog)
    ctrs[mask_prog] = np.random.uniform(0.003, 0.007, s_prog) # ~0.5% CTR
    cpcs[mask_prog] = np.random.uniform(0.30, 0.90, s_prog)

    # Search: Low Volume, High CTR, High CPC
    mask_search = df_fact['channel'] == 'Paid Search'
    s_search = mask_search.sum()
    base_imps[mask_search] = np.random.randint(300, 1200, s_search)
    ctrs[mask_search] = np.random.uniform(0.08, 0.12, s_search) # ~10% CTR
    cpcs[mask_search] = np.random.uniform(2.50, 6.00, s_search)

    # Social: Mid Volume, Mid CTR, Mid CPC
    mask_social = df_fact['channel'] == 'Paid Social'
    s_social = mask_social.sum()
    base_imps[mask_social] = np.random.randint(1000, 4000, s_social)
    ctrs[mask_social] = np.random.uniform(0.015, 0.035, s_social) # ~2.5% CTR
    cpcs[mask_social] = np.random.uniform(1.50, 3.50, s_social)

    # Organic: No Cost
    mask_org = df_fact['channel'] == 'Organic'
    s_org = mask_org.sum()
    base_imps[mask_org] = np.random.randint(1000, 3000, s_org)
    ctrs[mask_org] = np.random.uniform(0.05, 0.08, s_org)
    cpcs[mask_org] = 0.0
  
    # 4. Inject Strategic Insights (The "Story" Layer)
    final_imps = base_imps * seasonality
    
    # INSIGHT 1: "The August Spike" - Programmatic impressions triple in Aug '23
    mask_spike = (df_fact['year'] == 2023) & (df_fact['month'] == 8) & (df_fact['channel'] == 'Programmatic')
    final_imps[mask_spike] *= 3.0
    
    # INSIGHT 2: "December Efficiency" - Search CPC drops 30% in Dec '23 (Bid Optimization)
    mask_effic = (df_fact['year'] == 2023) & (df_fact['month'] == 12) & (df_fact['channel'] == 'Paid Search')
    cpcs[mask_effic] *= 0.7

    # 5. Final Calculations
    df_fact['impressions'] = final_imps.astype(int)
    df_fact['clicks'] = (df_fact['impressions'] * ctrs).astype(int)
    df_fact['spend'] = (df_fact['clicks'] * cpcs).round(2)
    
    # Conversion Rate (Randomized between 5% and 15%)
    conv_rates = np.random.uniform(0.05, 0.15, N)
    df_fact['conversions'] = (df_fact['clicks'] * conv_rates).astype(int)
    
    # Video Views (Only relevant for Display/Social channels)
    df_fact['video_views'] = 0
    mask_video = df_fact['channel'].isin(['Programmatic', 'Paid Social'])
    
# Logic: ~40% of impressions are video, with some variance
    df_fact.loc[mask_video, 'video_views'] = (
        df_fact.loc[mask_video, 'impressions'] * 0.40 * np.random.uniform(0.8, 1.2, mask_video.sum())
    ).astype(int)

    # ==========================================
    # III. EXPORT (4 Clean CSVs)
    # ==========================================
    
    # Select only keys and metrics for the Fact Table (Star Schema Best Practice)
    fact_columns = ['date_key', 'source_id', 'campaign_id', 'impressions', 'clicks', 'spend', 'conversions', 'video_views']
    fact_final = df_fact[fact_columns]

    # Clean up dimensions (remove helper columns used for generation logic)
    dim_campaign_final = dim_campaign.drop(columns=['channel_scope'])

    # Ensure output directory exists
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV
    dim_date.to_csv(os.path.join(output_dir, 'dim_date.csv'), index=False)
    dim_source.to_csv(os.path.join(output_dir, 'dim_source.csv'), index=False)
    dim_campaign_final.to_csv(os.path.join(output_dir, 'dim_campaign.csv'), index=False)
    fact_final.to_csv(os.path.join(output_dir, 'fact_performance.csv'), index=False)
    
    print("SUCCESS: 4 Star Schema files generated!")
    print(f"Fact Table rows: {len(fact_final)}")
    print(f"Files saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    # Execute generation in current directory
    generate_marketing_star_schema()