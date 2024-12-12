import pandas as pd

# Specify the file paths - Species in File 1 not in File 2
file1_path = r''  # Excel file
file2_path = r''  # CSV file

# Load the Excel and CSV files
sheet1 = pd.read_excel(file1_path)
sheet2 = pd.read_csv(file2_path)

# Extract species names from the specified columns
species_file1 = sheet1.iloc[:, 0].dropna().str.strip().str.lower()  # Column 1 from file 1
species_file2 = sheet2.iloc[:, 14].dropna().str.strip().str.lower()  # Column 15 from file 2

# Total count of species in File 1
total_species_file1 = len(species_file1)

# Find species in file1 that are not in file2
species_not_in_file2 = species_file1[~species_file1.isin(species_file2)]
count_species_not_in_file2 = len(species_not_in_file2)

# Create DataFrame for the results
not_found_df = pd.DataFrame({'Species in File 1 not in File 2': species_not_in_file2})

# Save the results to a new Excel file
output_path = r''  # Output file path
not_found_df.to_excel(output_path, index=False)

# Print the counts
print(f"Total species in File 1: {total_species_file1}")
print(f"Species in File 1 not in File 2: {count_species_not_in_file2}")
print(f"Results saved to {output_path}.")
