import pandas as pd

# Load the CSV file
file_path = r''
df = pd.read_csv(file_path)

# Extract the relevant columns
species_col = df.columns[7]  # 8th column for genus and species
atmosphere_col = df.columns[13]  # 14th column for atmosphere

# Remove duplicates to ensure uniqueness in species and atmosphere
df_unique = df.drop_duplicates(subset=[species_col, atmosphere_col])

# Initialize variables
ranking = []
remaining_species = set(df_unique[species_col])

# Debug print to check initial counts
print("Initial count of remaining species:", len(remaining_species))

while remaining_species:
    # Count unique species for each atmosphere
    filtered_df = df_unique[df_unique[species_col].isin(remaining_species)]
    atmosphere_species_count = filtered_df.groupby(atmosphere_col)[species_col].nunique()

    # Debug print to check atmosphere counts
    print("Filtered DataFrame:", filtered_df.head())
    print("Atmosphere Species Count:", atmosphere_species_count)

    if not atmosphere_species_count.empty:
        # Rank atmosphere by the number of unique species
        top_atmosphere = atmosphere_species_count.idxmax()
        ranking.append((top_atmosphere, atmosphere_species_count[top_atmosphere]))

        # Remove species produced by the top-ranked atmosphere
        species_to_remove = set(df_unique[df_unique[atmosphere_col] == top_atmosphere][species_col])
        remaining_species -= species_to_remove

        # Debug print to check remaining species
        print("Remaining species after removal:", len(remaining_species))
    else:
        print("No atmosphere species count available. Exiting loop.")
        break

# Convert the ranking to a DataFrame for easy viewing
ranking_df = pd.DataFrame(ranking, columns=['Atmosphere', 'Unique Species Count'])

# Sort the DataFrame by 'Unique Species Count' (descending) and 'Atmosphere' (ascending)
ranking_df = ranking_df.sort_values(by=['Unique Species Count', 'Atmosphere'], ascending=[False, True])

# Assign ranks ensuring no gaps in ranking
ranking_df['Rank'] = range(1, len(ranking_df) + 1)

# Add a row for the total count
total_unique_species = df_unique[species_col].nunique()
total_row = pd.DataFrame([['Total', total_unique_species]], columns=['Atmosphere', 'Unique Species Count'])
total_row['Rank'] = None  # No rank for the total row
ranking_df = pd.concat([ranking_df, total_row], ignore_index=True)

# Save the result to a new Excel file
output_file_path = r''
ranking_df.to_excel(output_file_path, index=False)

print(f"Ranking has been saved to {output_file_path}")
print(ranking_df)

# Print unique atmosphere not included in ranking
all_atmosphere = set(df[atmosphere_col])
ranked_atmosphere = set(ranking_df['Atmosphere'])
missing_atmosphere = all_atmosphere - ranked_atmosphere

print("Missing atmosphere types that were not included in the ranking:")
print(missing_atmosphere)
