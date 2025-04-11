import requests
from bs4 import BeautifulSoup
import csv

def scrape_contract_data(url, output_filename):
    # Request webpage
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching webpage: Status Code {response.status_code}")
        return
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find table with the contract data.
    table = soup.find('table', id = 'player-contracts')
    if table is None:
        print("Error: Could not find Contract table.")
        return
    
    # Check which is the correct header
    thead = table.find('thead')
    """for i, tr in enumerate(thead.find_all('tr')):
        row_headers = [th.get_text(strip=True) for th in tr.find_all(['th', 'td'])]
        print(f"Header row {i}:", row_headers)
    
    
    if headers and headers[0] == '':
        headers = headers[1]
        """
    
    # Extract Appropriate Header
    header_rows = thead.find_all('tr')
    header_tr = header_rows[-1] # Use second header row
    headers = [th.get_text(strip=True) for th in header_tr.find_all('th')]
    
    # Extract all rows from <tbody> section
    rows = []
    for row in table.find('tbody').find_all('tr'):
        cells = row.find_all(['th', 'td'])
        row_data = [cell.get_text(strip=True) for cell in cells]
        rows.append(row_data)
        
    # Fill in the CSV File
    with open(output_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        
    print(f"Data successfully written to {output_filename}")
    
if __name__ == '__main__':
    url = 'https://www.basketball-reference.com/contracts/players.html'
    output_filename = 'contracts.csv'
    scrape_contract_data(url, output_filename)
    
