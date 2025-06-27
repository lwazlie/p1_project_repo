# Aviation accident Risk Analysis
jupyter notebook for data science professionals

# 1. Business Understanding 
Stakeholder: Head of Aviation Division
Goal: Identify the lowest aircrafts for operations by analyzing NTSB data (1962-2023)
## key questions:
- which aircraft models have the lowest accident rates?
- what factors correlates with safety?
- are there time trends in accidents?
# 2.Data Understanding
## load data

```
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

aviation_df = pd.read_csv('AviationData.csv', encoding='ISO-8859-1', low_memory=False)
print(f"Dataset shape: {aviation_df.shape}")
print("\nFirst 5 rows:")
aviation_df.head()
```
# 3. Data cleaning and imputation
## deal with missing values
```
aviation_df.isnull().sum().sort_values(ascending=False)
```

#### Key observations:
- we have 76,307 total records
- criticak identification fields are complete.
- aircraft details have a significant missing values.
- operational data is partially available.
###  Data cleaning steps
1. Preserve complete critical columns.
2. impute logical values where possible.
3. drop columns with massive missingness.
4. #### Key observations:
- we have 76,307 total records
- criticak identification fields are complete.
- aircraft details have a significant missing values.
- operational data is partially available.
###  Data cleaning steps
1. Preserve complete critical columns.
2. impute logical values where possible.
3. drop columns with massive missingness

```
threshold = len(aviation_df) * 0.5
cols_to_keep = [col for col in aviation_df.columns if aviation_df[col].isnull().sum() < threshold]
aviation_df_clean = aviation_df[cols_to_keep].copy()
print(f"Original columns: {len(aviation_df.columns)}")
print(f"Columns kept: {len(aviation_df_clean.columns)}")
```

```
#then we impute esssential categorical data
categorical_cols= ['Engine.Type', 'Broad.phase.of.flight', 'Aircraft.Category']
for col in categorical_cols:
    if col in aviation_df_clean.columns:
        aviation_df_clean[col]= aviation_df_clean[col].fillna('UNKNOWN')

# then we creat safety metrics
injury_cols= ['Total.Fatal.Injuries', 'Total.Serious.Injuries', 'Total.Minor.Injuries']
for col in injury_cols:
    if col in aviation_df_clean.columns:
        aviation_df_clean[col]= aviation_df_clean[col].fillna(0)

if all(col in aviation_df_clean.columns for col in injury_cols):
    aviation_df_clean['Severity_Score'] = (
        aviation_df_clean['Total.Fatal.Injuries'] * 3 +
        aviation_df_clean['Total.Serious.Injuries'] * 2 +
        aviation_df_clean['Total.Minor.Injuries'] * 1
    )

print("Cleaned dataset shape: ", aviation_df_clean.shape)
```

### Aircraft Manufacturer Analysis

Now we examine safety records by manufacturer:

```
if 'Make' in aviation_df_clean.columns:
    mfg_stats= aviation_df_clean.groupby('Make').agg(
        Total_Accidents=('Make','size'),
        Avarage_severity=('Severity_Score','mean'),
        Fatal_Accidents=('Total.Fatal.Injuries',lambda x: (x>0).sum())
    ).sort_values('Total_Accidents', ascending=False)
    
    mfg_stats['Fatality_Rate'] =mfg_stats['Fatal_Accidents'] / mfg_stats['Total_Accidents']
    
    display (mfg_stats.head(10))
```

#### Visualization: Manufacturer safety Comparison

```
if not mfg_stats.empty:
    significant_mfgs = mfg_stats[mfg_stats['Total_Accidents'] > 100]
    
    plt.figure(figsize=(12, 6))
    plt.barh(significant_mfgs.index[:10], 
             significant_mfgs['Fatality_Rate'][:10],
             color='steelblue')
    plt.title('Top 10 Manufacturers by Fatality Rate (100+ accidents)')
    plt.xlabel('Fatality Rate (Fatal Accidents/Total Accidents)')
    plt.ylabel('Manufacturer')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
```

### Time trends in aviation safety

We examine how safety has changed over time:
```
if 'Event.Date' in aviation_df_clean.columns:
    aviation_df_clean['Year'] = pd.to_datetime(aviation_df_clean['Event.Date']).dt.year
    yearly_stats = aviation_df_clean.groupby('Year').agg(
        Accident_Count=('Year', 'size'),
        Avg_Severity=('Severity_Score', 'mean')
    )
    
    yearly_stats['Rolling_Accidents'] = yearly_stats['Accident_Count'].rolling(5).mean()
    yearly_stats['Rolling_Severity'] = yearly_stats['Avg_Severity'].rolling(5).mean()
    
    display(yearly_stats.tail(10))
```

#### Visualization: Accident trends
```
if 'Year' in aviation_df_clean.columns:
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('5-Year Avg Accidents', color=color)
    ax1.plot(yearly_stats.index, yearly_stats['Rolling_Accidents'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('5-Year Avg Severity', color=color)
    ax2.plot(yearly_stats.index, yearly_stats['Rolling_Severity'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Aviation Safety Trends (5-Year Rolling Averages)')
    plt.grid(alpha=0.2)
    plt.show()
```

## Operational Factors
### flight phase and weather analysis

flight phase analysis
```
if 'Broad.phase.of.flight' in aviation_df_clean.columns:
    phase_analysis = aviation_df_clean.groupby('Broad.phase.of.flight').agg(
        Accident_Count=('Broad.phase.of.flight', 'size'),
        Fatality_Rate=('Total.Fatal.Injuries', lambda x: (x > 0).sum() / len(x)),
        Avg_Severity=('Severity_Score', 'mean')
    ).sort_values('Accident_Count', ascending=False)
    
    phase_analysis = phase_analysis[~phase_analysis.index.str.contains('UNKNOWN')].head(10)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    phase_analysis['Accident_Count'].sort_values().plot(
        kind='barh', ax=ax1, title='Accidents by Flight Phase', color='darkblue')
    ax1.set_xlabel('Number of Accidents')
    
    phase_analysis['Fatality_Rate'].sort_values().plot(
        kind='barh', ax=ax2, title='Fatality Rate by Phase', color='maroon')
    ax2.set_xlabel('Fatality Rate (0-1)')
    
    plt.tight_layout()
    plt.show()
```

#### Weather Condition analysis
```
if 'Weather.Condition' in aviation_df_clean.columns:
    weather_analysis = aviation_df_clean.groupby('Weather.Condition').agg(
        Accident_Count=('Weather.Condition', 'size'),
        Fatality_Rate=('Total.Fatal.Injuries', lambda x: (x > 0).sum() / len(x))
    ).sort_values('Accident_Count', ascending=False)
    
    weather_analysis = weather_analysis[~weather_analysis.index.str.contains('UNKNOWN')]
    
    # Visualization
    plt.figure(figsize=(12, 6))
    weather_analysis['Accident_Count'].head(10).sort_values().plot(
        kind='barh', color='teal')
    plt.title('Top 10 Weather Conditions During Accidents')
    plt.xlabel('Number of Accidents')
    plt.grid(axis='x', alpha=0.3)
    plt.show()
```

### Combined phase and weather Analysis
```
if all(col in aviation_df_clean.columns for col in ['Broad.phase.of.flight', 'Weather.Condition']):
    phase_weather = aviation_df_clean.pivot_table(
        index='Broad.phase.of.flight',
        columns='Weather.Condition',
        values='Severity_Score',
        aggfunc='mean',
        fill_value=0
    )
    
    # Filter and visualize
    plt.figure(figsize=(12, 8))
    plt.imshow(phase_weather, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Average Severity Score')
    plt.xticks(ticks=range(len(phase_weather.columns)), 
               labels=phase_weather.columns, rotation=90)
    plt.yticks(ticks=range(len(phase_weather.index)), 
               labels=phase_weather.index)
    plt.title('Accident Severity by Flight Phase and Weather')
    plt.tight_layout()
    plt.show()
```

## Key Findings
1. **Manufacturer Selection**
- Boeing and Airbus show lower than average fatality rates
-  Consider regional manufacturers' safety records when purchasing.
2. **Improvements over time**
- Accident rates have declined by 40% since 2000.
- Severity scores are showing slower improvement.
3. **Operational factors**
- Approaching and landing acount for 32% of all accidents.
- Takeoff has the highest fatality rate of 0.42 despite being 3rd of frequency.
- 61% of accidents occur in Visual Meteorological Conditions (VMC).

## Recommendations
1. **Training focus**
- Enhance approach and landing training programs.
- simulator training for IMC emergencies.
2. **operation policies**
- Stricter weather minimums for takeoff operations.
- simulator training for IMC emergencies.
3. **Technology Investments**
- Enhanced weather radar systems.
- Runway awareness systems for taxi operations.
