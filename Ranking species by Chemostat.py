import pandas as pd

# Load the CSV file
file_path = r''
df = pd.read_csv(file_path)

# Extract the relevant columns
species_col = df.columns[7]  # 8th column for genus and species
chemostat_col = df.columns[12]  # 13th column for chemostat

# Remove duplicates to ensure uniqueness in species and chemostat
df_unique = df.drop_duplicates(subset=[species_col, chemostat_col])

# Initialize variables
ranking = []
remaining_species = set(df_unique[species_col])

# Debug print to check initial counts
print("Initial count of remaining species:", len(remaining_species))

while remaining_species:
    # Count unique species for each chemostat
    filtered_df = df_unique[df_unique[species_col].isin(remaining_species)]
    chemostat_species_count = filtered_df.groupby(chemostat_col)[species_col].nunique()

    # Debug print to check chemostat counts
    print("Filtered DataFrame:", filtered_df.head())
    print("Chemostat Species Count:", chemostat_species_count)

    if not chemostat_species_count.empty:
        # Rank chemostat by the number of unique species
        top_chemostat = chemostat_species_count.idxmax()
        ranking.append((top_chemostat, chemostat_species_count[top_chemostat]))

        # Remove species produced by the top-ranked chemostat
        species_to_remove = set(df_unique[df_unique[chemostat_col] == top_chemostat][species_col])
        remaining_species -= species_to_remove

        # Debug print to check remaining species
        print("Remaining species after removal:", len(remaining_species))
    else:
        print("No chemostat species count available. Exiting loop.")
        break

# Convert the ranking to a DataFrame for easy viewing
ranking_df = pd.DataFrame(ranking, columns=['Chemostat', 'Unique Species Count'])

# Sort the DataFrame by 'Unique Species Count' (descending) and 'Chemostat' (ascending)
ranking_df = ranking_df.sort_values(by=['Unique Species Count', 'Chemostat'], ascending=[False, True])

# Assign ranks ensuring no gaps in ranking
ranking_df['Rank'] = range(1, len(ranking_df) + 1)

# Add a row for the total count
total_unique_species = df_unique[species_col].nunique()
total_row = pd.DataFrame([['Total', total_unique_species]], columns=['Chemostat', 'Unique Species Count'])
total_row['Rank'] = None  # No rank for the total row
ranking_df = pd.concat([ranking_df, total_row], ignore_index=True)

# Save the result to a new Excel file
output_file_path = r''
ranking_df.to_excel(output_file_path, index=False)

print(f"Ranking has been saved to {output_file_path}")
print(ranking_df)

# Print unique chemostat not included in ranking
all_chemostat = set(df[chemostat_col])
ranked_chemostat = set(ranking_df['Chemostat'])
missing_chemostat = all_chemostat - ranked_chemostat

print("Missing chemostat types that were not included in the ranking:")
print(missing_chemostat)
