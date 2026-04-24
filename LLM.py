from typing import List, Union
import base64
import requests
from io import BytesIO

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()


class ImageQAAgent:
    """
    A class that uses OpenAI Vision models via LangChain
    to answer questions based on multiple images.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        """
        Initialize the LLM.

        Args:
            model (str): OpenAI vision-capable model name.
            api_key (str): OpenAI API key (optional if set in env).
        """
        self._api_key = os.getenv("OPENAI_API_KEY")

        self.llm = ChatOpenAI(
            model=model,
            api_key=self._api_key
        )


    def _image_to_base64(self, image_path: str) -> str:
        """
        Convert a local image file to base64 string.
        """
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def _url_to_base64(self, url: str) -> str:
        """
        Convert image URL to base64 string.
        """
        response = requests.get(url)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")

    def _prepare_image(self, image: Union[str, dict]) -> dict:
        """
        Prepare image in OpenAI vision format.

        Supports:
        - URL
        - local file path
        - pre-encoded base64 (as dict)
        """
        if isinstance(image, dict) and "base64" in image:
            b64 = image["base64"]
        elif image.startswith("http"):
            b64 = self._url_to_base64(image)
        else:
            b64 = self._image_to_base64(image)

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }
        }

    def ask(self, question: str, images: List[Union[str, dict]]) -> str:
        """
        Ask a question based on multiple images.

        Args:
            question (str): User question
            images (List): List of image paths, URLs, or base64 dicts

        Returns:
            str: Model response
        """

        image_parts = [self._prepare_image(img) for img in images]

        message = HumanMessage(
            content=[
                {"type": "text", "text": question},
                *image_parts
            ]
        )

        response = self.llm.invoke([message])

        return response.content
