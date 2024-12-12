import pandas as pd

# Load the CSV file
file_path = r''
df = pd.read_csv(file_path)

# Use the correct column names
species_col = 'rank_species'       # Species column
condition_col = df.columns[9]      # 10th column for condition
media_col = df.columns[10]          # 11th column for media
chemostat_col = df.columns[12]      # 13th column for chemostat
atmosphere_col = df.columns[13]     # 14th column for atmosphere

# Remove duplicates to ensure uniqueness in species, condition, media, chemostat, and atmosphere
df_unique = df.drop_duplicates(subset=[species_col, condition_col, media_col, chemostat_col, atmosphere_col])

# Initialize variables
ranking = []
remaining_species = set(df_unique[species_col])

# Debug print to check initial counts
print("Initial count of remaining species:", len(remaining_species))

while remaining_species:
    # Count unique species for each condition-media-chemostat-atmosphere combination
    filtered_df = df_unique[df_unique[species_col].isin(remaining_species)]
    combination_species_count = filtered_df.groupby([condition_col, media_col, chemostat_col, atmosphere_col])[species_col].nunique()

    # Debug print to check combination counts
    print("Filtered DataFrame:", filtered_df.head())
    print("Combination Species Count:", combination_species_count)

    if not combination_species_count.empty:
        # Rank combinations by the number of unique species
        top_combination = combination_species_count.idxmax()
        ranking.append((top_combination, combination_species_count[top_combination]))

        # Remove species produced by the top-ranked combination
        species_to_remove = set(df_unique[
            (df_unique[condition_col] == top_combination[0]) & 
            (df_unique[media_col] == top_combination[1]) &
            (df_unique[chemostat_col] == top_combination[2]) &
            (df_unique[atmosphere_col] == top_combination[3])
        ][species_col])
        remaining_species -= species_to_remove

        # Debug print to check remaining species
        print("Remaining species after removal:", len(remaining_species))
    else:
        print("No combination species count available. Exiting loop.")
        break

# Convert the ranking to a DataFrame for easy viewing
ranking_df = pd.DataFrame(ranking, columns=['Condition-Media-Chemostat-Atmosphere Combination', 'Unique Species Count'])

# Extract each component into separate columns
ranking_df[['Condition', 'Media', 'Chemostat', 'Atmosphere']] = pd.DataFrame(ranking_df['Condition-Media-Chemostat-Atmosphere Combination'].tolist(), index=ranking_df.index)
ranking_df = ranking_df.drop(columns=['Condition-Media-Chemostat-Atmosphere Combination'])

# Sort the DataFrame by 'Unique Species Count' (descending), 'Condition', 'Media', 'Chemostat', and 'Atmosphere' (ascending)
ranking_df = ranking_df.sort_values(by=['Unique Species Count', 'Condition', 'Media', 'Chemostat', 'Atmosphere'], ascending=[False, True, True, True, True])

# Assign ranks ensuring no gaps in ranking
ranking_df['Rank'] = range(1, len(ranking_df) + 1)

# Add a row for the total count
total_unique_species = df_unique[species_col].nunique()
total_row = pd.DataFrame([['Total', None, None, None, total_unique_species]], columns=['Condition', 'Media', 'Chemostat', 'Atmosphere', 'Unique Species Count'])
total_row['Rank'] = None  # No rank for the total row
ranking_df = pd.concat([ranking_df, total_row], ignore_index=True)

# Save the result to a new Excel file
output_file_path = r''
ranking_df.to_excel(output_file_path, index=False)

print(f"Ranking has been saved to {output_file_path}")
print(ranking_df)

# Print unique combinations not included in ranking
all_combinations = set(df[[condition_col, media_col, chemostat_col, atmosphere_col]].apply(tuple, axis=1))
ranked_combinations = set(zip(ranking_df['Condition'], ranking_df['Media'], ranking_df['Chemostat'], ranking_df['Atmosphere']))
missing_combinations = all_combinations - ranked_combinations

print("Missing combinations that were not included in the ranking:")
print(missing_combinations)
