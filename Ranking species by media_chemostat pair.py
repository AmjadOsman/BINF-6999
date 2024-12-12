import pandas as pd

# Load the CSV file
file_path = r''
df = pd.read_csv(file_path)

# Use the correct column names
species_col = 'rank_species'   # Species column
media_col = df.columns[10]     # 11th column for media
chemostat_col = df.columns[12]  # 13th column for chemostat

# Remove duplicates to ensure uniqueness in species, media, and chemostat
df_unique = df.drop_duplicates(subset=[species_col, media_col, chemostat_col])

# Initialize variables
ranking = []
remaining_species = set(df_unique[species_col])

# Debug print to check initial counts
print("Initial count of remaining species:", len(remaining_species))

while remaining_species:
    # Count unique species for each media-chemostat pair
    filtered_df = df_unique[df_unique[species_col].isin(remaining_species)]
    media_chemostat_species_count = filtered_df.groupby([media_col, chemostat_col])[species_col].nunique()

    # Debug print to check media-chemostat counts
    print("Filtered DataFrame:", filtered_df.head())
    print("Media-Chemostat Species Count:", media_chemostat_species_count)

    if not media_chemostat_species_count.empty:
        # Rank media-chemostat pairs by the number of unique species
        top_media_chemostat = media_chemostat_species_count.idxmax()
        ranking.append((top_media_chemostat, media_chemostat_species_count[top_media_chemostat]))

        # Remove species produced by the top-ranked media-chemostat pair
        species_to_remove = set(df_unique[(df_unique[media_col] == top_media_chemostat[0]) & 
                                          (df_unique[chemostat_col] == top_media_chemostat[1])][species_col])
        remaining_species -= species_to_remove

        # Debug print to check remaining species
        print("Remaining species after removal:", len(remaining_species))
    else:
        print("No media-chemostat species count available. Exiting loop.")
        break

# Convert the ranking to a DataFrame for easy viewing
ranking_df = pd.DataFrame(ranking, columns=['Media-Chemostat Pair', 'Unique Species Count'])

# Extract media and chemostat into separate columns
ranking_df['Media'] = ranking_df['Media-Chemostat Pair'].apply(lambda x: x[0])
ranking_df['Chemostat'] = ranking_df['Media-Chemostat Pair'].apply(lambda x: x[1])
ranking_df = ranking_df.drop(columns=['Media-Chemostat Pair'])

# Sort the DataFrame by 'Unique Species Count' (descending), 'Media' (ascending), and 'Chemostat' (ascending)
ranking_df = ranking_df.sort_values(by=['Unique Species Count', 'Media', 'Chemostat'], ascending=[False, True, True])

# Assign ranks ensuring no gaps in ranking
ranking_df['Rank'] = range(1, len(ranking_df) + 1)

# Add a row for the total count
total_unique_species = df_unique[species_col].nunique()
total_row = pd.DataFrame([['Total', None, None, total_unique_species]], columns=['Media', 'Chemostat', 'Unique Species Count', 'Rank'])
total_row['Rank'] = None  # No rank for the total row
ranking_df = pd.concat([ranking_df, total_row], ignore_index=True)

# Save the result to a new Excel file
output_file_path = r''
ranking_df.to_excel(output_file_path, index=False)

print(f"Ranking has been saved to {output_file_path}")
print(ranking_df)

# Print unique media-chemostat pairs not included in ranking
all_media_chemostat = set(df[[media_col, chemostat_col]].apply(tuple, axis=1))
ranked_media_chemostat = set(zip(ranking_df['Media'], ranking_df['Chemostat']))
missing_media_chemostat = all_media_chemostat - ranked_media_chemostat

print("Missing media-chemostat pairs that were not included in the ranking:")
print(missing_media_chemostat)
