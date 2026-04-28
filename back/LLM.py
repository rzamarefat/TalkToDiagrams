from typing import List, Union
import base64
import requests
import mimetypes
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


class LLM:
    """
    Stateful Vision Chat LLM using LangChain + OpenAI.
    Supports multi-turn conversations with memory.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = None,
        system_prompt: str = "You are a helpful assistant."
    ):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")

        self.llm = ChatOpenAI(
            model=model,
            api_key=self._api_key
        )

        # 🔑 Conversation memory
        self.messages = [
            SystemMessage(content=system_prompt)
        ]

    # -------------------------
    # Image Handling
    # -------------------------
    def _image_to_base64(self, image_path: str):
        mime, _ = mimetypes.guess_type(image_path)
        mime = mime or "image/jpeg"

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        return b64, mime

    def _url_to_base64(self, url: str):
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        b64 = base64.b64encode(response.content).decode("utf-8")
        mime = response.headers.get("Content-Type", "image/jpeg")

        return b64, mime

    def _prepare_image(self, image: Union[str, dict]) -> dict:
        if isinstance(image, dict) and "base64" in image:
            b64 = image["base64"]
            mime = image.get("mime_type", "image/jpeg")

        elif isinstance(image, str) and image.startswith("http"):
            b64, mime = self._url_to_base64(image)

        else:
            b64, mime = self._image_to_base64(image)

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{b64}"
            }
        }

    # -------------------------
    # Chat Methods
    # -------------------------
    def chat(self, question: str, images: List[Union[str, dict]] = None) -> str:
        """
        Stateful chat call (non-streaming).
        """
        images = images or []
        image_parts = [self._prepare_image(img) for img in images]

        user_message = HumanMessage(
            content=[
                {"type": "text", "text": question},
                *image_parts
            ]
        )

        # Add user message to memory
        self.messages.append(user_message)

        # Call model with full history
        response = self.llm.invoke(self.messages)

        # Store assistant reply
        self.messages.append(AIMessage(content=response.content))

        return response.content

    def chat_stream(self, question: str, images: List[Union[str, dict]] = None):
        """
        Stateful streaming chat.
        """
        images = images or []
        image_parts = [self._prepare_image(img) for img in images]

        user_message = HumanMessage(
            content=[
                {"type": "text", "text": question},
                *image_parts
            ]
        )

        self.messages.append(user_message)

        stream = self.llm.stream(self.messages)

        full_response = ""

        for chunk in stream:
            if chunk.content:
                full_response += chunk.content
                yield chunk.content

        # Save full assistant response after streaming completes
        self.messages.append(AIMessage(content=full_response))

    # -------------------------
    # Memory Control
    # -------------------------
    def reset(self):
        """Clear conversation history."""
        self.messages = [self.messages[0]]  # keep system prompt

    def get_history(self):
        """Return full conversation history."""
        return self.messages