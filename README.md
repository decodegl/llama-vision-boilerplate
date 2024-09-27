# LLamaVision boilerplate  


This repo sets a LLamaVision server based on `LitServe`.

# Instructions

1. Install [poetry](https://python-poetry.org/docs/) on your machine globally.
2. Go to [HuggingFace and get accepted by Meta](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision) to download the model. You will receive an email once accepted.
3. Get a fresh [HuggingFace token](https://huggingface.co/settings/tokens), `read only` permission is enough
4. Clone this repo
5. Run `poetry install`

# Usage

1. Activate poetry shell in the repo

```
poetry shell
```

2. Start the server 

```
python server.py
```

You can change the port number with `--port XXXX`

3. You can use the test client (in another shell). Active poetry shell again then:

```
python client.py --image PATH --query "Describe this image" [--port XXXX]
```
