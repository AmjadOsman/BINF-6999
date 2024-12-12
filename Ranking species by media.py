
import pandas as pd

# Load the CSV file
file_path = r''
df = pd.read_csv(file_path)

# Extract the relevant columns
species_col = df.columns[7]  # 8th column for genus and species
media_col = df.columns[10]   # 11th column for media

# Remove duplicates to ensure uniqueness in species
df_unique = df.drop_duplicates(subset=[species_col, media_col])

# Initialize variables
ranking = []
remaining_species = set(df_unique[species_col])

# Debug print to check initial counts
print("Initial count of remaining species:", len(remaining_species))

while remaining_species:
    # Count unique species for each media
    filtered_df = df_unique[df_unique[species_col].isin(remaining_species)]
    media_species_count = filtered_df.groupby(media_col)[species_col].nunique()

    # Debug print to check media counts
    print("Filtered DataFrame:", filtered_df.head())
    print("Media Species Count:", media_species_count)

    if not media_species_count.empty:
        # Rank media by the number of unique species
        top_media = media_species_count.idxmax()
        ranking.append((top_media, media_species_count[top_media]))

        # Remove species produced by the top-ranked media
        species_to_remove = set(df_unique[df_unique[media_col] == top_media][species_col])
        remaining_species -= species_to_remove

        # Debug print to check remaining species
        print("Remaining species after removal:", len(remaining_species))
    else:
        print("No media species count available. Exiting loop.")
        break

# Convert the ranking to a DataFrame for easy viewing
ranking_df = pd.DataFrame(ranking, columns=['Media', 'Unique Species Count'])

# Sort the DataFrame by 'Unique Species Count' (descending) and 'Media' (ascending)
ranking_df = ranking_df.sort_values(by=['Unique Species Count', 'Media'], ascending=[False, True])

# Assign ranks ensuring no gaps in ranking
ranking_df['Rank'] = range(1, len(ranking_df) + 1)

# Add a row for the total count
total_unique_species = df_unique[species_col].nunique()
total_row = pd.DataFrame([['Total', total_unique_species]], columns=['Media', 'Unique Species Count'])
total_row['Rank'] = None  # No rank for the total row
ranking_df = pd.concat([ranking_df, total_row], ignore_index=True)

# Save the result to a new Excel file
output_file_path = r''
ranking_df.to_excel(output_file_path, index=False)

print(f"Ranking has been saved to {output_file_path}")
print(ranking_df)

# Print unique media not included in ranking
all_media = set(df[media_col])
ranked_media = set(ranking_df['Media'])
missing_media = all_media - ranked_media

print("Missing media that were not included in the ranking:")
print(missing_media)

