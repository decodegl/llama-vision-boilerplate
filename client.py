import base64
import requests
import argparse

from rich import print
import os


class LlamaVisionClient:
    def __init__(self, api_url):
        self.api_url = api_url

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_mime_type(self, image_path):
        extension = image_path.split('.')[-1].lower()
        if extension in ['jpg', 'jpeg']:
            return 'image/jpeg'
        elif extension == 'png':
            return 'image/png'
        elif extension == 'gif':
            return 'image/gif'
        else:
            raise ValueError(f"Unsupported image extension: {extension}")

    def get_encoded_image_with_mime_type(self, image_path):
        encoded_image = self.encode_image(image_path)
        mime_type = self.get_mime_type(image_path)
        return f"data:{mime_type};base64,{encoded_image}"

    def create_payload(self, encoded_image_mime, query, max_tokens=50, temperature=0.2):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {"url": encoded_image_mime},
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def send_request(self, payload):
        response = requests.post(self.api_url, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]
    
    
def main():
    parser = argparse.ArgumentParser(description="LlamaVisionClient Command Line Interface")
    parser.add_argument("--image", type=str, help="Path to the image file")
    parser.add_argument("--query", type=str, help="Query to ask about the image")
    parser.add_argument("--port", type=int, default=10111, help="Port number of the API server (default: 10111)")
    
    args = parser.parse_args()
    
    # Resolve the image path if it's relative
    image_path = os.path.abspath(args.image)
    
    api_url = f"http://localhost:{args.port}/v1/chat/completions"
    client = LlamaVisionClient(api_url)
    
    encoded_image_mime = client.get_encoded_image_with_mime_type(image_path)
    payload = client.create_payload(encoded_image_mime, args.query)
    response = client.send_request(payload)
    
    print(response)

if __name__ == "__main__":
    main()
