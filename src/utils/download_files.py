import io
import imageio
import gzip
import os
import requests

class DownloadFile:
    def __init__(self, from_url, to_folder, file_name):
        self.from_url = from_url
        self.to_folder = to_folder
        self.file_name = file_name
        if not os.path.exists(self.to_folder):
            os.makedirs(self.to_folder)

    def as_image(self, format='png'):
        request = requests.get(self.from_url)
        byte_file = io.BytesIO(request.content)
        read_file = gzip.open(byte_file).read()
        image = imageio.imread(read_file)
        imageio.imwrite(f'{self.to_folder}/{self.file_name}.{format}', image)

# def download_image(from_url, to_path):
#     request = requests.get(from_url)
#     byte_file = io.BytesIO(request.content)
#     read_file = gzip.open(byte_file).read()
#     image = imageio.imread(read_file)
#     imageio.imwrite(to_path, image)