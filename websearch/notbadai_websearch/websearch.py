from typing import List
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from collections import defaultdict

from ddgs import DDGS


class WebSearch:
    def __init__(self,
                 query: str,
                 num_results: int = 10,
                 discard_urls: List[str] = None,
                 user_agent: str = "*"
                 ):
        self.query = query
        self.num_results = num_results
        self.discard_urls = discard_urls if discard_urls is not None else ["youtube.com", "britannica.com", "vimeo.com"]
        self.user_agent = user_agent

    def filter_urls_by_robots_txt(self, urls: List[str]) -> List[str]:
        robot_urls = defaultdict(list)
        for url in urls:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            robot_urls[robots_url].append(url)

        allowed_urls = []
        for robots_url, url_list in robot_urls.items():
            try:
                rp = RobotFileParser(robots_url)
                rp.read()

                for url in url_list:
                    if rp.can_fetch(self.user_agent, url):
                        allowed_urls.append(url)
            except Exception as e:
                allowed_urls.extend(url_list)

        return allowed_urls

    def search(self) -> List[str]:
        search_term = self.query
        for url in self.discard_urls:
            search_term += f" -site:{url}"

        results = DDGS().text(search_term, max_results=self.num_results)
        results = [result["href"] for result in results]

        return self.filter_urls_by_robots_txt(results)
