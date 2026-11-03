"""
Helper class for reading images from LMDB files.

This module provides a reader class that can be integrated into PyTorch dataloaders
to efficiently load images from LMDB databases without reading entire archives.
"""

import lmdb
import io
from PIL import Image
from typing import Dict
from pathlib import Path


class LMDBImageReader:
    """
    Reader for images stored in LMDB files.

    Opens LMDB environments lazily and caches them for efficient access.
    Each image can be accessed individually without loading the entire database.
    """

    def __init__(self, lmdb_root: str):
        """
        Initialize the LMDB reader.

        Args:
            lmdb_root: Root directory containing LMDB databases
        """
        self.lmdb_root = Path(lmdb_root)
        self.envs: Dict[str, lmdb.Environment] = {}

    def __del__(self):
        """Close all LMDB environments on cleanup."""
        self.close_all()

    def close_all(self):
        """Close all open LMDB environments."""
        for env in self.envs.values():
            env.close()
        self.envs.clear()

    def _get_env(self, lmdb_name: str) -> lmdb.Environment:
        """
        Get or create an LMDB environment.

        Args:
            lmdb_name: Name of the LMDB database (e.g., 'llava_imgs_00000')

        Returns:
            LMDB environment handle
        """
        if lmdb_name not in self.envs:
            lmdb_path = str(self.lmdb_root / lmdb_name)
            # Open in read-only mode with lock=False for multi-process access
            self.envs[lmdb_name] = lmdb.open(
                lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
        return self.envs[lmdb_name]

    def read_image(self, lmdb_name: str, image_key: str) -> Image.Image:
        """
        Read an image from LMDB.

        Args:
            lmdb_name: Name of the LMDB database containing the image
            image_key: Key for the image (usually the relative path like '00454/004547755.jpg')

        Returns:
            PIL Image object

        Raises:
            KeyError: If the image key is not found in the database
        """
        env = self._get_env(lmdb_name)

        with env.begin(write=False) as txn:
            img_bytes = txn.get(image_key.encode('utf-8'))

            if img_bytes is None:
                raise KeyError(f"Image key '{image_key}' not found in LMDB '{lmdb_name}'")

            # Convert bytes to PIL Image
            img = Image.open(io.BytesIO(img_bytes))
            # Convert to RGB if needed (handles RGBA, L, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img

    def read_image_bytes(self, lmdb_name: str, image_key: str) -> bytes:
        """
        Read raw image bytes from LMDB.

        Args:
            lmdb_name: Name of the LMDB database containing the image
            image_key: Key for the image

        Returns:
            Raw image bytes

        Raises:
            KeyError: If the image key is not found in the database
        """
        env = self._get_env(lmdb_name)

        with env.begin(write=False) as txn:
            img_bytes = txn.get(image_key.encode('utf-8'))

            if img_bytes is None:
                raise KeyError(f"Image key '{image_key}' not found in LMDB '{lmdb_name}'")

            return img_bytes
