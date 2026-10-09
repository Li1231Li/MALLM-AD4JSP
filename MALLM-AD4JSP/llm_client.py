import os
import re
import json
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

try:
    import requests
except ModuleNotFoundError:
    requests = None


class LLMTools:
    def __init__(self):
        self.api_endpoint = os.environ.get("ARK_API_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
        self.api_key = os.environ.get("ARK_API_KEY")
        self.model_id = os.environ.get("ARK_MODEL_ID", "deepseek-v4-flash-260425")

    def chat(self, system_prompt: str, user_prompt: str,
             history: list = None, temperature: float = 0.7,
             max_tokens: int = 4000, max_retries: int = 3) -> str:
        if not self.api_key:
            raise RuntimeError("Set ARK_API_KEY before calling the LLM API.")

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_err = None
        for attempt in range(max_retries):
            try:
                if requests is not None:
                    resp = requests.post(
                        self.api_endpoint, headers=headers,
                        data=json.dumps(payload), timeout=120
                    )
                    resp.raise_for_status()
                    data = resp.json()
                else:
                    req = Request(
                        self.api_endpoint,
                        data=json.dumps(payload).encode("utf-8"),
                        headers=headers,
                        method="POST",
                    )
                    with urlopen(req, timeout=120) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except (Exception, HTTPError, URLError) as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after {max_retries} attempts: {last_err}")


def extract_code_block(text: str, lang: str = "python") -> str:
    pattern = rf"```{lang}\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    matches = re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    return text.strip()
