# Fearless-erfgoed
Code voor het project Finding owners of stolen and lost art of World War II through AI image recognition voor het Ministerie van Onderwijs, Cultuur en Wetenschap  Rijksdienst voor het Cultureel Erfgoed.

# Datasets
We have lots of different folders for images explained below:

image_categories.csv - All pre classified classes for the MCCP dataset.

keinbild.jpg - Needed for scraping_munchen to compare with and filter it out.

Our testset:
nk_testset - 5 handpicked images from NK-collection we use to test our models
munich_testset - 5 handpicked images from MCCP-collection we use to test our models

nk_testset_no_back - same as nk_testset with background removal
munich_testset_no_back - same as munich_testset with background removal

Large datasets:
scraped_images_grayscaled_big - MCCP dataset after grayscaling and removing small images.

nk_collection_paintings_cleaned - NK dataset that contains all grayscaled paintings
nk_collection_paintings - NK dataset of paintings

nk_collection_furniture_cleaned - NK dataset that contains all grayscaled furniture
nk_collection_furniture - NK dataset of furniture


# Programs
background_removal.ipynb - Takes a folder containing images as input, removes the background and makes a new directory.

CNN_final - Final deliverable, can turn image classification on or off.

sift.ipynb - File containing our SIFT implementation/testing and images generated for the report.

orb.ipynb - File containing our ORB implementation and parameter optimization.

scraping_munchen.py - File to scrape the images from the MCCP website and grayscaling function.

scraping_nk.py - File to scrape the images from the NK API.