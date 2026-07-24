import os
import hashlib
import urllib.request

class WeightManager:
    """
    Centralized model asset handling.
    Resolves caching, downloads, and checksum validation.
    """
    def __init__(self, cache_dir: str = "./.cache/omniid/weights"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _verify_checksum(self, path: str, expected_sha256: str) -> bool:
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest() == expected_sha256

    def load_weight(self, url: str, expected_sha256: str = None) -> str:
        """
        Locates or downloads a weight file. Returns the local path.
        """
        filename = url.split("/")[-1]
        local_path = os.path.join(self.cache_dir, filename)

        if os.path.exists(local_path):
            if expected_sha256 and not self._verify_checksum(local_path, expected_sha256):
                print(f"Checksum mismatch for {filename}. Redownloading...")
                os.remove(local_path)
            else:
                return local_path

        print(f"Downloading {filename}...")
        # Mock download for the sake of the framework right now
        # urllib.request.urlretrieve(url, local_path)
        with open(local_path, "w") as f:
            f.write("mock_weight_data")

        return local_path
