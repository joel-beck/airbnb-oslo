#%%
import pandas as pd

# conda/mamba install requests
import requests

# conda/mamba install bs4
from bs4 import BeautifulSoup

# conda/mamba install tqdm
from tqdm import tqdm

import requests
from io import BytesIO
from PIL import Image

tqdm.pandas()

#%%
listings = pd.read_pickle("listings.pkl")
listings = pd.DataFrame(listings)

GET_FRONT_URLS = False
GET_RESPONSES = True
GET_ALL_URLS = False

#%%
# urls of webpages with only (usually 5) front pictures for each apartment
picture_pages = listings["listing_url"]

#%%
# SECTION: Get Pictures from Front Page
def get_url(picture: str) -> str:
    """
    Extracts URL from the first Picture on an Apartment's Front Page
    """

    return picture.find(class_="_6tbg2q")["src"]


def get_url_list(apartment_url: str) -> list[str]:
    """
    Collects URLs of all Pictures on an Apartment's Front Page in a List
    """

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
if GET_FRONT_URLS:
    # takes about 45 minutes on my cpu
    url_lists = picture_pages.progress_apply(get_url_list)

    # transform list in each row to long format, such that each row only contains one
    # picture url and the correct index-value mapping is maintained
    front_page_urls = url_lists.explode()
    front_page_urls.to_pickle("front_page_urls.pkl")

#%%
# most apartments have 5 pictures on front page
front_page_urls = pd.read_pickle("front_page_urls.pkl")
front_page_urls.groupby(front_page_urls.index).count().value_counts()

#%%
def get_response(url: str) -> Image:
    try:
        response = requests.get(url)
    except:
        response = pd.NA
    return response


#%%
if GET_RESPONSES:
    # takes about 1 hour on my cpu
    front_page_responses = front_page_urls.progress_apply(get_response)
    front_page_responses.to_pickle("front_page_responses.pkl")

#%%
# SECTION: Get Pictures from extended picture page when clicking "Alle Fotos anzeigen"
if GET_ALL_URLS:

    def add_photos(url: str) -> str:
        """
        returns URL of the Webpage containing all Pictures (not only the first 5) of an Apartment
        """

        return url + "/photos"

    # urls of webpages with all pictures for each apartment
    all_picture_pages = picture_pages.apply(add_photos)

    # NOTE: Extended Page is likely rendered by JavaScript => BeautifulSoup does not work
    # https://stackoverflow.com/questions/50014456/beautiful-soup-can-not-find-all-image-tags-in-html-stops-exactly-at-5

    # Possible Solutions:
    # - Selenium: quite complicated
    # - requests-html: simpler but does not work right now, https://docs.python-requests.org/projects/requests-html/en/latest/

    # BOOKMARK: Experimental
    # conda/mamba install requests-html
    from requests_html import AsyncHTMLSession

    asession = AsyncHTMLSession()

    # r = await asession.get(
    #     "https://www.airbnb.de/rooms/42932/photos?_set_bev_on_new_domain=1638088059_YzdiYWU0ZmVmNjBk&source_impression_id=p3_1640558986_v9VqTe%2B8Ep%2FDNpjX"
    # )

    # await r.html.arender()

    # pictures = r.html.find("img")

    # still only finds first 5 images
    # for picture in pictures:
    #     print(picture.attrs["src"])
    #     print()
