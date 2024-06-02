import os
import re
import time

import numpy as np
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

_DIR = os.path.dirname(os.path.abspath(__file__))


class MacmillanDictionary:
    def __init__(self, save_dir, http_proxy="http://127.0.0.1:2340"):
        self.save_dir = save_dir
        self.headers = {"user-agent": "my-app/0.0.1"}
        self.proxies = {"http": http_proxy, "https": http_proxy}
        self.http_proxy = http_proxy
        self.url_prefix = "https://www.macmillandictionary.com/us/dictionary/american/"
        os.makedirs(os.path.join(self.save_dir, "success"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "failure"), exist_ok=True)

    def download(self, word):
        save_path = os.path.join(self.save_dir, "success", word)
        fail_path = os.path.join(self.save_dir, "failure", word)
        if os.path.exists(save_path + ".mp3"):
            return True
        if os.path.exists(fail_path):
            return False

        # time.sleep(np.random.uniform(2.0, 5.0))

        # * url and page
        try:
            url = self.url_prefix + word
            page = requests.get(url, headers=self.headers, proxies=self.proxies)
            soup = BeautifulSoup(page.content, "html.parser")
        except:
            print("Network error!")
            return False

        # * fail to find
        if soup.text.find("Sorry, no search result for") >= 0:
            with open(fail_path, "w"):
                pass
            return False

        node = soup.find("span", class_="sound audio_play_button dflex middle-xs")
        if node is None:
            return False
        data_src = node.attrs["data-src-mp3"]
        node = soup.find("span", class_="PRON")
        if node is None:
            return False
        phonemes = node.text.strip()
        with open(save_path + ".pron", "w") as fp:
            fp.write(phonemes)

        ret = os.popen(f"curl -I -s '{data_src}' --proxy '{self.http_proxy}' | grep HTTP").read()
        assert ret.split()[1] == "200"

        res = os.system(f"curl -L '{data_src}' -o '{save_path}.mp3' -s")
        return res == 0 and os.path.exists(save_path + ".mp3")


class CambridgeDictionary:
    def __init__(self, save_dir, http_proxy="http://127.0.0.1:2340"):
        self.root_url = "https://dictionary.cambridge.org"
        self.save_dir = save_dir
        self.headers = {"user-agent": "my-app/0.0.1"}
        self.proxies = {"http": http_proxy, "https": http_proxy}
        self.http_proxy = http_proxy
        os.makedirs(os.path.join(self.save_dir, "success"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "failure"), exist_ok=True)

    def download(self, word):
        word = word.lower()
        save_path = os.path.join(self.save_dir, "success", word)
        fail_path = os.path.join(self.save_dir, "failure", word)
        if os.path.exists(save_path + ".mp3"):
            return True
        if os.path.exists(fail_path):
            return False

        # * url and page
        try:
            url = self.root_url + f"/us/dictionary/english/{word}"
            page = requests.get(url, headers=self.headers, proxies=self.proxies)
            soup = BeautifulSoup(page.content, "html.parser")
        except:
            print("Network error!")
            return False

        # * find all header div
        failure = True
        span_list = []
        title = None
        div_list = soup.find_all("div", class_="pos-header dpos-h")
        for div in div_list:
            span_list = div.find_all("span", class_="us dpron-i")
            if len(span_list) == 0:
                continue
            # * the first div contains "us dpron-i"
            failure = False
            title = div.find("div", class_="di-title").text.lower()
            while title[0] == "-":
                title = title[1:]
            while title[-1] == "-":
                title = title[:-1]
            if word.find(".") < 0:
                title = re.sub(r"\.", "", title)
            break

        # * fail to find
        if failure or len(span_list) == 0 or title is None:
            with open(fail_path, "w"):
                pass
            return False

        # * get source from the first span
        node = None
        for span in span_list:
            node = span.find("source")
            if node is not None:
                break
        if node is None:
            return False
        data_src = self.root_url + node.attrs["src"]

        # * fix the word
        if title != word:
            print("Only {} instead of {}".format(title, word))
            failure = True
            with open(fail_path, "w"):
                pass
        word = title

        ret = os.popen(f"curl -I -s '{data_src}' --proxy '{self.http_proxy}' | grep HTTP").read()
        assert ret.split()[1] == "200"

        save_path = os.path.join(self.save_dir, "success", word)
        res = os.system(f"curl -L '{data_src}' -o '{save_path}.mp3' -s")
        failure |= res != 0 or not os.path.exists(save_path + ".mp3")

        return not failure


def read_words_lists(md_fpath):
    words = []
    with open(md_fpath) as fp:
        for line in fp:
            line = line.strip()
            if len(line) == 0 or line[0] != "|":
                continue
            line = [x.strip() for x in line.split("|")[1:-1]]
            if line[0].startswith("0") or line[0].startswith("1"):
                continue
            if line[0].startswith(":") or line[0].startswith("-"):
                continue
            words.extend(line)
    return [x.lower() for x in words]


if __name__ == "__main__":
    words = []
    words.extend(read_words_lists(os.path.join(_DIR, "words-NU-6.md")))
    words.extend(read_words_lists(os.path.join(_DIR, "words-CID-W-22.md")))
    words.extend(read_words_lists(os.path.join(_DIR, "words-Maryland-CNC.md")))
    words.extend(read_words_lists(os.path.join(_DIR, "words-PAL-PB50.md")))
    words = ["cars"] + sorted(list(set(words)))
    # words = ["deemed", "of", "cars", "tell"]

    for dictionary in [
        CambridgeDictionary(os.path.join(_DIR, "cambridgedictionary")),
        # MacmillanDictionary(os.path.join(_DIR, "macmillandictionary")),
    ]:
        pbar = tqdm(words)
        for i, word in enumerate(pbar):
            pbar.set_description(word)
            if not dictionary.download(word):
                print("Failed to download '{}'".format(word))
