#%%
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

tqdm.pandas()

#%%
listings = pd.read_pickle("listings.pkl")
listings = pd.DataFrame(listings)

GET_FRONT_PICTURES = False

#%%
# urls of webpages with only (usually 5) front pictures for each apartment
picture_pages = listings["listing_url"]

#%%
# SECTION: Get Pictures from Front Page
def get_url(picture: str) -> str:
    return picture.find(class_="_6tbg2q")["src"]


def get_url_list(apartment_url: str) -> list[str]:
    response = requests.get(apartment_url)
    soup = BeautifulSoup(response.text, "html.parser")
    pictures = soup.find_all("picture")

    url_list = []
    for picture in pictures:
        try:
            url = get_url(picture)
        except:
            url = pd.NA
        url_list.append(url)
    return url_list


#%%
if GET_FRONT_PICTURES:
    # takes about 45 minutes on my cpu
    url_lists = picture_pages.progress_apply(get_url_list)

    # transform list in each row to long format, such that each row only contains one
    # picture url and the correct index-value mapping is maintained
    front_page_pictures = url_lists.explode()
    front_page_pictures.to_pickle("front_page_pictures.pkl")

#%%
# most apartments have 5 pictures on front page
front_page_pictures.reset_index().groupby("id").count().value_counts()


#%%
# SECTION: Get Pictures from extended picture page when clicking "Alle Fotos anzeigen"
def add_photos(url: str) -> str:
    return url + "/photos"


#%%
# urls of webpages with all pictures for each apartment
all_picture_pages = picture_pages.apply(add_photos)

#%%
get_url_list(all_picture_pages.iloc[0])

#%%
response = requests.get(all_picture_pages.iloc[0])
soup = BeautifulSoup(response.text, "html.parser")

# TODO: Does not find all pictures on page but only the first 5
pictures = soup.find_all("picture")
