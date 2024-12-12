import pandas as pd

# Load the CSV file
file_path = r''
df = pd.read_csv(file_path)

# Use the correct column names
species_col = 'rank_species'   # Species column
media_col = df.columns[10]     # 11th column for media
atmosphere_col = df.columns[13]  # 14th column for atmosphere

# Remove duplicates to ensure uniqueness in species, media, and atmosphere
df_unique = df.drop_duplicates(subset=[species_col, media_col, atmosphere_col])

# Initialize variables
ranking = []
remaining_species = set(df_unique[species_col])

# Debug print to check initial counts
print("Initial count of remaining species:", len(remaining_species))

while remaining_species:
    # Count unique species for each media-atmosphere pair
    filtered_df = df_unique[df_unique[species_col].isin(remaining_species)]
    media_atmosphere_species_count = filtered_df.groupby([media_col, atmosphere_col])[species_col].nunique()

    # Debug print to check media-atmosphere counts
    print("Filtered DataFrame:", filtered_df.head())
    print("Media-Atmosphere Species Count:", media_atmosphere_species_count)

    if not media_atmosphere_species_count.empty:
        # Rank media-atmosphere pairs by the number of unique species
        top_media_atmosphere = media_atmosphere_species_count.idxmax()
        ranking.append((top_media_atmosphere, media_atmosphere_species_count[top_media_atmosphere]))

        # Remove species produced by the top-ranked media-atmosphere pair
        species_to_remove = set(df_unique[(df_unique[media_col] == top_media_atmosphere[0]) & 
                                          (df_unique[atmosphere_col] == top_media_atmosphere[1])][species_col])
        remaining_species -= species_to_remove

        # Debug print to check remaining species
        print("Remaining species after removal:", len(remaining_species))
    else:
        print("No media-atmosphere species count available. Exiting loop.")
        break

# Convert the ranking to a DataFrame for easy viewing
ranking_df = pd.DataFrame(ranking, columns=['Media-Atmosphere Pair', 'Unique Species Count'])

# Extract media and atmosphere into separate columns
ranking_df['Media'] = ranking_df['Media-Atmosphere Pair'].apply(lambda x: x[0])
ranking_df['Atmosphere'] = ranking_df['Media-Atmosphere Pair'].apply(lambda x: x[1])
ranking_df = ranking_df.drop(columns=['Media-Atmosphere Pair'])

# Sort the DataFrame by 'Unique Species Count' (descending), 'Media' (ascending), and 'Atmosphere' (ascending)
ranking_df = ranking_df.sort_values(by=['Unique Species Count', 'Media', 'Atmosphere'], ascending=[False, True, True])

# Assign ranks ensuring no gaps in ranking
ranking_df['Rank'] = range(1, len(ranking_df) + 1)

# Add a row for the total count
total_unique_species = df_unique[species_col].nunique()
total_row = pd.DataFrame([['Total', None, None, total_unique_species]], columns=['Media', 'Atmosphere', 'Unique Species Count', 'Rank'])
total_row['Rank'] = None  # No rank for the total row
ranking_df = pd.concat([ranking_df, total_row], ignore_index=True)

# Save the result to a new Excel file
output_file_path = r''
ranking_df.to_excel(output_file_path, index=False)

print(f"Ranking has been saved to {output_file_path}")
print(ranking_df)

# Print unique media-atmosphere pairs not included in ranking
all_media_atmosphere = set(df[[media_col, atmosphere_col]].apply(tuple, axis=1))
ranked_media_atmosphere = set(zip(ranking_df['Media'], ranking_df['Atmosphere']))
missing_media_atmosphere = all_media_atmosphere - ranked_media_atmosphere

print("Missing media-atmosphere pairs that were not included in the ranking:")
print(missing_media_atmosphere)
