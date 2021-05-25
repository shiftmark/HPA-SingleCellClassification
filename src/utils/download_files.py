import io
import cv2
import gzip
import numpy as np
import os
import requests

class DownloadFile:
    """ 
    Dowload files using a link and specifying a filename and location.
    Args:
        from_url (string) - The url;
        to_folder (string) - The location to save the file to. If the folder doesn't exist, it will be created.
        file_name (string) - The file name. When calling 'as_image' method, the extension is specified in the 'format' argument.
    """

    def __init__(self, from_url, to_folder, file_name):
        self.from_url = from_url
        self.to_folder = to_folder
        self.file_name = file_name
        if not os.path.exists(self.to_folder):
            os.makedirs(self.to_folder)

    def as_image(self, file_format='.png'):
        """
        Save file as image.
        Args:
            format (string) - The image format to save as. Default is '.png'. The dot is required.
        Returns:
            None. Writes the file on disk.
        """
        request = requests.get(self.from_url)
        byte_file = io.BytesIO(request.content)
        read_file = gzip.open(byte_file).read()
        arr = np.asarray(bytearray(read_file), dtype='uint8')
        image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(f'{self.to_folder}/{self.file_name}{file_format}', image)

    def as_is(self):
        """
        Opens the file (bytes) and writes it on disk. Specify the extenstion in the 'file_name' class argument.
        Args: None
        Returns: None
        """
        request = requests.get(self.from_url)
        with open(self.file_name, 'w') as output:
            output.write(request.content)
