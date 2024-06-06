import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
from skimage.metrics import structural_similarity as ssim
from skimage import io
import numpy as np
from io import BytesIO

cookies = {'PHPSESSID': 'u6h1scn6knqdj77di40e2n6uo3'}

# Directory to save images
img_dir = "scraped_images"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

keinbild_path = "keinbild.jpg"
keinbild_image = io.imread(keinbild_path)

def images_are_same(image1, image2):
    if image1.shape != image2.shape:
        return False
    

    # Compute SSIM between the two images
    similarity_index = ssim(image1, image2, multichannel=True)

    # Adjust the threshold as needed
    threshold = 0.9  # Example threshold, adjust as necessary

    # Check if similarity index is above the threshold
    return similarity_index > threshold

for i in range(2388):
    print(i)
    # URL of the webpage to scrape
    url = f"https://www.dhm.de/datenbank/ccp/dhm_ccp.php?seite=8&current={i * 20}"
    print(url)

    try:
        # Send a GET request to the webpage with cookies
        response = requests.get(url, cookies=cookies, timeout=10)
        response.raise_for_status()  # Check if the request was successful

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all image tags
        img_tags = soup.find_all('img')

        for img in img_tags:
            img_url = img.get('src')
            if img_url and 'displayimg' in img_url:
                # Construct full URL
                img_url = urljoin(url, img_url)

                try:
                    # Get the image content with cookies
                    img_response = requests.get(img_url, cookies=cookies, timeout=10)
                    img_response.raise_for_status()

                    # Read the image as an array
                    img_array = io.imread(BytesIO(img_response.content))

                    if not images_are_same(img_array, keinbild_image):

                        # Extract image filename
                        img_name = os.path.join(img_dir, os.path.basename(img_url))

                        # Ensure the file has a .jpg extension
                        if not img_name.endswith('.jpg'):
                            img_name += '.jpg'

                        # Save the image
                        with open(img_name, 'wb') as img_file:
                            img_file.write(img_response.content)

                except requests.exceptions.RequestException as img_err:
                    print(f"Failed to download image {img_url}: {img_err}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")