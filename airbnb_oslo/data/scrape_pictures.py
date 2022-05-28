import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm

tqdm.pandas()


def get_url(picture: str) -> str:
    """
    Extracts URL from the first Picture on an Apartment's Front Page.
    """

    return picture.find(class_="_6tbg2q")["src"]


def get_url_list(apartment_url: str) -> list[str]:
    """
    Collects URLs of all Pictures on an Apartment's Front Page in a List.
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


def get_response(url: str) -> Image:
    try:
        response = requests.get(url)
    except:
        response = pd.NA
    return response


def main():
    listings = pd.read_pickle("../../data/clean/listings.pkl")
    listings = pd.DataFrame(listings)

    GET_FRONT_URLS = True
    GET_RESPONSES = True

    # urls of webpages with only (usually 5) front pictures for each apartment
    picture_pages = listings["listing_url"]

    # SECTION: Get Pictures from Front Page
    if GET_FRONT_URLS:
        # takes about 45 minutes on my cpu
        url_lists = picture_pages.progress_apply(get_url_list)

        # transform list in each row to long format, such that each row only contains one
        # picture url and the correct index-value mapping is maintained
        front_page_urls = url_lists.explode()
        front_page_urls.to_pickle("../../data/clean/front_page_urls.pkl")

    # most apartments have 5 pictures on front page
    front_page_urls = pd.read_pickle("../../data/clean/front_page_urls.pkl")
    front_page_urls.groupby(front_page_urls.index).count().value_counts()

    if GET_RESPONSES:
        # takes about 1 hour on my cpu
        front_page_responses = front_page_urls.progress_apply(get_response)
        front_page_responses.to_pickle("../../data/clean/front_page_responses.pkl")


if __name__ == "__main__":
    main()
