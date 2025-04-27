import requests
import pdfplumber
import os
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time
import json

visited_urls = set()  # Store visited URLs
extracted_data = []  # List to store content of all pages

def fetch_content(url):
    """Fetches the content from the given URL."""
    try:
        # Add a user-agent header to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=10, headers=headers)  # Fetch page
        response.raise_for_status()  # Check for errors
        return response.text
    except requests.RequestException as e:
        print(f"Failed to access {url}: {e}")
        return None

def get_page_type(url):
    """Determines the type of content (HTML, PDF, or Plain Text) from the URL."""
    if url.lower().endswith(".pdf"):
        return "PDF"
    elif url.lower().endswith(".txt"):
        return "TEXT"
    else:
        return "HTML"

def download_pdf(url):
    """Downloads the PDF file to the local machine."""
    try:
        # Add a user-agent header to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        filename = url.split("/")[-1]  # Extract the filename from the URL
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename
    except requests.RequestException as e:
        print(f"Failed to download PDF: {e}")
        return None

def extract_text_from_html(soup):
    """Extracts and returns text content from an HTML page."""
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    return soup.get_text(separator='\n', strip=True)

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF document (using pdfplumber)."""
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
            return text
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""
    finally:
        # Ensure file is removed even if an exception occurs
        if os.path.exists(file_path):
            try:
                os.remove(file_path)  # Clean up by removing the downloaded PDF file
            except Exception as e:
                print(f"Error removing PDF file: {e}")

def extract_text_from_plain_text(url):
    """Extracts text content from a plain text document."""
    try:
        # Add a user-agent header to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Failed to fetch plain text file: {e}")
        return ""

def get_all_pages(base_url, url, level=1, max_depth=4):
    """Fetches the content of pages up to the specified depth."""
    if url in visited_urls:
        return
    
    # Skip non-FDIC URLs and certain file types we don't want to process
    parsed_url = urlparse(url)
    if "fdic.gov" not in parsed_url.netloc or any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.svg', '.css', '.js']):
        return
    
    print(f"Crawling (Level {level}/{max_depth}): {url}")
    visited_urls.add(url)  # Mark as visited
    
    # Fetch the content from the page
    page_content = fetch_content(url)
    if not page_content:
        return

    # Determine the type of content
    page_type = get_page_type(url)
    
    # Extract text content based on the page type
    if page_type == 'HTML':
        soup = BeautifulSoup(page_content, 'html.parser')
        extracted_content = extract_text_from_html(soup)
        
        # If we haven't reached max depth, collect links for further crawling
        if level < max_depth:
            for link in soup.find_all('a', href=True):
                next_url = urljoin(base_url, link['href'])
                parsed_next = urlparse(next_url)
                
                # Only follow links to the same domain and avoid anchors
                if parsed_next.netloc == parsed_url.netloc and "#" not in parsed_next.path:
                    # Process this URL in the next iteration
                    get_all_pages(base_url, next_url, level+1, max_depth)

    elif page_type == 'PDF':
        pdf_filename = download_pdf(url)  # Download the PDF to local
        if pdf_filename:
            extracted_content = extract_text_from_pdf(pdf_filename)  # Extract text from the downloaded PDF
        else:
            extracted_content = "Failed to download PDF"
    elif page_type == 'TEXT':
        extracted_content = extract_text_from_plain_text(url)
    else:
        extracted_content = "Unsupported content type"
    
    # Prepare the data structure to store
    data = {
        "url": url,
        "type": page_type,
        "level": level,
        "content": extracted_content
    }

    # Store the extracted content in the list
    extracted_data.append(data)
    
    # Save after each page to avoid losing data if the script crashes
    save_to_json()
    
    # Be polite and avoid server overload
    time.sleep(1)  

def save_to_json():
    """Save the extracted content to a JSON file."""
    with open('aml_data.json', 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)

# Main execution
if __name__ == "__main__":
    start_url = "https://www.fdic.gov/banker-resource-center/anti-money-laundering-countering-financing-terrorism-amlcft"
    
    # Reset global variables
    visited_urls = set()
    extracted_data = []
    
    print(f"Starting crawl from {start_url} with max depth of 4")
    get_all_pages(start_url, start_url, level=1, max_depth=4)
    
    # Final save
    save_to_json()
    
    print(f"Crawling complete. Visited {len(visited_urls)} URLs.")
    print(f"Extracted data stored in 'fdic_extracted_data.json'")