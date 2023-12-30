from __future__ import annotations

import io
import json
import os
import random
import subprocess
from functools import lru_cache
from time import sleep
from urllib.parse import urljoin

import brotli
import pandas as pd
import requests
from pydantic import BaseModel
from pydantic import Field


class KaggleCompetitionDownloader:
    def __init__(self, competition_name: str) -> None:
        self.competition_name = competition_name

    def list_kernels(self):
        current_page = 1
        finished = False
        output_frames = []
        while not finished:
            print(f"Downloading page {current_page}")
            res = subprocess.run(
                [
                    "kaggle",
                    "kernels",
                    "list",
                    "-v",
                    "--competition",
                    self.competition_name,
                    "-p",
                    str(current_page),
                    "--page-size",
                    "100",
                ],
                capture_output=True,
            )
            output = res.stdout.decode("utf-8")
            if "Not found" in output:
                finished = True
            else:
                output_frames.append(pd.read_csv(io.StringIO(output)))
                current_page += 1
        if len(output_frames) == 0:
            raise ValueError("No kernels found")
        return pd.concat(output_frames)

    def download_kernel(self, kernels: pd.DataFrame, path: str):
        for _, row in kernels.iterrows():
            subprocess.run(["kaggle", "kernels", "pull", row["ref"]], cwd=str(path))

    def download_all_kernels(self, path: str):
        kernels = self.list_kernels()
        self.download_kernel(kernels, path)

    def convert_all_kernels(self, input_path: str, output_path: str):
        kernels = os.listdir(input_path)
        for kernel in kernels:
            print(f"Converting {kernel}")
            notebook_path = os.path.join(input_path, kernel)
            res = subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "markdown",
                    "--output-dir",
                    output_path,
                    notebook_path,
                ],
                capture_output=True,
            )
            if res.returncode != 0:
                print(f"Failed to convert {kernel}")
                print(res.stderr.decode("utf-8"))


class CompetitionPost(BaseModel):
    """Post model for competition API"""

    competitionIdOrName: str = Field(..., alias="id")

    def __hash__(self) -> int:
        return hash(self.competitionIdOrName)


class DiscussionTopicPost(BaseModel):
    """Post model for discussion topic API"""

    page: int = Field(1, alias="page")
    forumId: int = Field(..., alias="forumId")

    def __hash__(self) -> int:
        return hash((self.page, self.forumId))


class DiscussionPost(BaseModel):
    """Post model for discussion API"""

    forumTopicId: int = Field(..., alias="forumTopicId")
    includeComments: bool = Field(True, alias="includeComments")

    def __hash__(self) -> int:
        return hash((self.forumTopicId, self.includeComments))


class KaggleCompetitionCrawler:
    """Crawler for Kaggle competition discussion"""

    BASE_URL = "https://www.kaggle.com"
    COMPETITION_API_URL = (
        "/api/i/competitions.legacy.LegacyCompetitionService/GetCompetition"
    )
    DISCUSSION_TOPIC_URL = "/api/i/discussions.DiscussionsService/GetTopicListByForumId"
    DISCUSSION_URL = "/api/i/discussions.DiscussionsService/GetForumTopicById"
    RANDOM_DELAY = 5

    def __init__(self, competition_name: str):
        self.competition_name = competition_name
        # open session
        self.session = requests.Session()
        self.session.get(
            f"https://www.kaggle.com/competitions/{competition_name}/discussion/",
        )
        self.headers = {
            "accept-encoding": "gzip, deflate, br",
            "x-xsrf-token": self.session.cookies.get_dict()["XSRF-TOKEN"],
        }

    def get_discussion_id(self):
        """Get discussion id from competition name"""
        return self._post_to_api(
            urljoin(self.BASE_URL, self.COMPETITION_API_URL),
            CompetitionPost(id=self.competition_name),
        )["discussion"]["id"]

    def get_topic_ids(self):
        """Get topic ids from discussion id"""
        discussion_id = self.get_discussion_id()
        # iterate over pages
        topics_exhausted = False
        current_page = 1
        output = []
        while not topics_exhausted:
            print(f"Getting topics from page {current_page}...")
            parsed = self._post_to_api(
                urljoin(self.BASE_URL, self.DISCUSSION_TOPIC_URL),
                data=DiscussionTopicPost(
                    forumId=discussion_id,
                    page=current_page,
                ),
            )
            if "topics" in parsed:
                # parse id
                output.extend([i["id"] for i in parsed["topics"]])
            else:
                topics_exhausted = True
            current_page += 1
        return output

    def get_discussions(self):
        """Get discussions from topic ids"""
        topic_ids = self.get_topic_ids()
        output = []
        for topic_id in topic_ids:
            print(f"Getting discussion {topic_id}...")
            parsed = self._post_to_api(
                urljoin(self.BASE_URL, self.DISCUSSION_URL),
                data=DiscussionPost(forumTopicId=topic_id),
            )
            output.append(parsed)
        return output

    def _get_from_api(self, url):
        response = self.session.get(url, headers=self.headers)
        response.raise_for_status()
        return self._parse_response(response)

    @lru_cache(maxsize=5000)
    def _post_to_api(self, url, data: BaseModel):
        # add random delay
        sleep(self.RANDOM_DELAY * random.random())
        response = self.session.post(
            url,
            headers=self.headers,
            json=data.model_dump(),
        )
        response.raise_for_status()
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response):
        return json.loads(brotli.decompress(response.content).decode("utf-8"))
