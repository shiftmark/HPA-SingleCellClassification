import io
import imageio
import gzip
import os
import requests

class DownloadFile:
    """ 
    Dowload files using a link and specifying a filename and location.
    Args:
        from_url (string) - the url;
        to_folder (string) - the location to save the file to. If the folder doesn't exist, it will be created.
        file_name (string) - the file name. When calling 'as_image' method, the extension is specified in the 'format' argument.
    """

    def __init__(self, from_url, to_folder, file_name):
        self.from_url = from_url
        self.to_folder = to_folder
        self.file_name = file_name
        if not os.path.exists(self.to_folder):
            os.makedirs(self.to_folder)

    def as_image(self, format='png'):
        """
        Save file as image.
        Args:
            format (string) - the image format to save as. Default is 'png'.
        Returns:
            None. Writes the file on disk.
        """
        request = requests.get(self.from_url)
        byte_file = io.BytesIO(request.content)
        read_file = gzip.open(byte_file).read()
        image = imageio.imread(read_file)
        imageio.imwrite(f'{self.to_folder}/{self.file_name}.{format}', image)

    def as_is(self):
        """
        Opens the file (bytes) and writes it on disk. Specify the extenstion in the 'file_name' class argument.
        Args: None
        Returns: None
        """
        request = requests.get(self.from_url)
        with open(self.file_name, 'w') as output:
            output.write(request.content)
