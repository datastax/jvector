import pandas as pd

# Read the CSV file
df = pd.read_csv('results.csv')

# Filter for main benchmark results (execution time)
main_results = df[df['Benchmark'].str.endswith('testOnHeapRandomVectorsWithRecall') &
                  ~df['Benchmark'].str.contains(':')]

# Filter for all auxiliary counters
recall_results = df[df['Benchmark'].str.contains(':avgRecall')]
reranked_results = df[df['Benchmark'].str.contains(':avgReRankedCount')]
visited_results = df[df['Benchmark'].str.contains(':avgVisitedCount')]
expanded_results = df[df['Benchmark'].str.contains(':avgExpandedCountBaseLayer')]

# Merge all results on the numberOfPQSubspaces parameter
summary = main_results.copy()

# Merge recall results
summary = summary.merge(
    recall_results[['Param: numberOfPQSubspaces', 'Score']],
    on='Param: numberOfPQSubspaces',
    suffixes=('', '_avgRecall'),
    how='left'
)

# Merge reranked count results
summary = summary.merge(
    reranked_results[['Param: numberOfPQSubspaces', 'Score']],
    on='Param: numberOfPQSubspaces',
    suffixes=('', '_avgReRankedCount'),
    how='left'
)

# Merge visited count results
summary = summary.merge(
    visited_results[['Param: numberOfPQSubspaces', 'Score']],
    on='Param: numberOfPQSubspaces',
    suffixes=('', '_avgVisitedCount'),
    how='left'
)

# Merge expanded count results
summary = summary.merge(
    expanded_results[['Param: numberOfPQSubspaces', 'Score']],
    on='Param: numberOfPQSubspaces',
    suffixes=('', '_avgExpandedCountBaseLayer'),
    how='left'
)

# Create a clean summary table with all auxiliary counters
summary_clean = summary[[
    'Param: k',
    'Param: numberOfPQSubspaces',
    'Score',
    'Score_avgRecall',
    'Score_avgReRankedCount',
    'Score_avgVisitedCount',
    'Score_avgExpandedCountBaseLayer'
]]

# Rename columns for better readability
summary_clean.columns = [
    'k',
    'PQ_Subspaces',
    'Time_ms',
    'Recall',
    'ReRanked_Count',
    'Visited_Count',
    'Expanded_Count_BaseLayer'
]

# Format numeric columns for better display
summary_clean['Time_ms'] = summary_clean['Time_ms'].round(3)
summary_clean['Recall'] = summary_clean['Recall'].round(3)
summary_clean['ReRanked_Count'] = summary_clean['ReRanked_Count'].round(1)
summary_clean['Visited_Count'] = summary_clean['Visited_Count'].round(1)
summary_clean['Expanded_Count_BaseLayer'] = summary_clean['Expanded_Count_BaseLayer'].round(1)

print(summary_clean.to_string(index=False))