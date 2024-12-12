import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from lxml import etree

# Set up Selenium WebDriver using webdriver-manager
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# The base URL of the DSMZ website and the page with the list of species
base_url = 'https://www.dsmz.de'
species_list_url = 'https://www.dsmz.de/collection/catalogue/microorganisms/special-groups-of-organisms/human-gut-microbiota-sanger'

# Navigate to the main page with the list of species
driver.get(species_list_url)

# Add an explicit wait to ensure the page loads completely
wait = WebDriverWait(driver, 10)  # Waits up to 10 seconds

# Extract species links using Selenium
def get_species_links():
    try:
        # Wait for the main content to load
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'a')))
        
        # Extract species links
        species_links = []
        elements = driver.find_elements(By.TAG_NAME, 'a')

        # Find all 'a' tags with href and filter for relevant links
        for element in elements:
            href = element.get_attribute('href')
            if href:
                # Check if the href is a relative link (starts with '/') and ensure we only append the base_url when needed
                if href.startswith('/'):
                    full_url = base_url + href
                elif href.startswith('http'):
                    full_url = href
                else:
                    continue
                
                # Further filter the links to ensure we are getting species links (containing "catalogue/details/culture/")
                if '/catalogue/details/culture/' in full_url:
                    species_links.append(full_url)

        print(f"Found {len(species_links)} species links.")
        return species_links
    except Exception as e:
        print(f"Error fetching the species list page: {e}")
        return []

# Function to extract name, target information from summary, and 16S rRNA accession number using Selenium
def extract_species_data(species_url):
    try:
        # Navigate to the species page using Selenium
        driver.get(species_url)
        
        # Wait for the page content to load
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'h1')))
        
        # Parse the page source with lxml
        tree = etree.HTML(driver.page_source)
        
        # Extract the species name (usually in <h1> tag)
        try:
            name = tree.xpath('//h1/text()')[0].strip()
            print(f"Species Name: {name}")
        except IndexError:
            name = 'N/A'

        # Extract the entire "Summary and additional information" section
        extracted_info = 'N/A'
        try:
            # Generalized XPath to find the summary section by looking for labels or markers
            summary_section = tree.xpath('//div[contains(@class, "label") and contains(text(), "Summary")]/following-sibling::div[contains(@class, "value")]')
            if summary_section:
                # Extract the entire text from this div and its children
                summary_text = ''.join(summary_section[0].itertext()).strip()
                # Split the text into lines and find the line with the word 'medium'
                summary_lines = [line.strip() for line in summary_text.splitlines() if line.strip()]
                for line in summary_lines:
                    if 'medium' in line.lower():  # Case-insensitive search
                        # Find the last period before 'medium'
                        last_period_pos = line.rfind('.', 0, line.lower().find('medium'))
                        if last_period_pos != -1:
                            # Extract text to the right of this position
                            extracted_info = line[last_period_pos + 1:].strip()
                        else:
                            # If no period, take the whole line
                            extracted_info = line.strip()
                        break
            else:
                print(f"Summary and additional information section not found for {species_url}")
        except Exception as e:
            print(f"Error extracting summary information from {species_url}: {e}")

        # Extract the "16S rRNA gene" Genbank accession number or whole genome shotgun sequence
        accession_number = 'N/A'
        try:
            # Generalized XPath to find the accession number based on its label
            accession_section = tree.xpath('//div[contains(@class, "label") and contains(text(), "Genbank accession numbers")]/following-sibling::div[contains(@class, "value")]')
            if accession_section:
                accession_text = ''.join(accession_section[0].itertext()).strip()
                # Look for the "16S rRNA gene:" line
                if '16S rRNA gene:' in accession_text:
                    for line in accession_text.splitlines():
                        if '16S rRNA gene:' in line:
                            # Include "16S rRNA gene:" and the text after it
                            accession_number = line.strip()
                            break
            # If not found, check the alternative generalized XPath
            if accession_number == 'N/A':
                # Try the generalized fallback XPath (which you specified directly)
                fallback_section = tree.xpath('/html/body/div[1]/div[2]/main/div[2]/div/div[1]/div/div/div[3]/div[10]/div[2]')
                if fallback_section:
                    # Join all text content within the fallback section to ensure capturing everything
                    accession_number = ''.join(fallback_section[0].itertext()).strip()
                    
        except Exception as e:
            print(f"Error extracting 16S rRNA gene or whole genome shotgun sequence from {species_url}: {e}")

        return {
            'Species': name,
            'Extracted Information': extracted_info,
            '16S rRNA Accession Number': accession_number
        }
    except Exception as e:
        print(f"Error processing {species_url}: {e}")
        return {
            'Species': 'N/A',
            'Extracted Information': 'N/A',
            '16S rRNA Accession Number': 'N/A'
        }

# Main script execution
def main():
    # Get all species links
    species_links = get_species_links()

    # List to hold the extracted data
    extracted_data = []

    # Loop through each species link and extract the data
    for species_link in species_links:
        data = extract_species_data(species_link)
        extracted_data.append(data)

    # Convert the extracted data to a DataFrame
    df = pd.DataFrame(extracted_data)

    # Save the DataFrame to an Excel file (make sure to install openpyxl)
    df.to_excel(r'', index=False)

    print("Data extraction complete. Saved to species_extracted_information_sanger2.xlsx.")

    # Close the Selenium WebDriver
    driver.quit()

if __name__ == "__main__":
    main()
