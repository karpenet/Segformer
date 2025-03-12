import safetensors
import requests


{"B0": "nvidia/segformer-b0-finetuned-ade-512-512"}

def _check_safetensor(file_path: str):
    pass

def download_safetensor_hf(file_path: str):
    url = "https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/main/model.safetensors" 

    # Send HTTP GET request to download the file
    response = requests.get(url, stream=True)

    # Save the content to the new file name
    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

print("Download complete.")

def load_weights():
    pass


if __name__ == "__main__":
    download_safetensor_hf("weights/test.safetensors")