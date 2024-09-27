import litserve as ls
import argparse

from model import Llama3

class Llama3VisionAPI(ls.LitAPI):
    def setup(self, device):
        self.model = Llama3(device)

    def decode_request(self, request):
        return self.model.apply_chat_template(request.messages)

    def predict(self, inputs, context):
        yield self.model(inputs)

    def encode_response(self, outputs):
        for output in outputs:
            yield {"role": "assistant", "content": self.model.decode_tokens(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the llama3 vision server")
    parser.add_argument("--port", type=int, default=10111, help="Port number to run the server on")
    args = parser.parse_args()
    api = Llama3VisionAPI()
    server = ls.LitServer(api, accelerator="auto", spec=ls.OpenAISpec())
    server.run(port=args.port)