import os
import orjson

from utils import find_nearest_neighbors, find_nearest_neighbors_simple


class PickleDB:
    """
    A barebones orjson-based key-value store with essential methods:
    set, get, save, remove, purge, and all.
    """

    def __init__(self, location):
        """
        Initialize the PickleDB object.

        Args:
            location (str): Path to the JSON file.
        """
        self.location = os.path.expanduser(location)
        self._load()

    def __setitem__(self, key, value):
        """
        Wraps the `set` method to allow `db[key] = value`. See `set`
        method for details.
        """
        return self.set(key, value)

    def __getitem__(self, key):
        """
        Wraps the `get` method to allow `value = db[key]`. See `get`
        method for details.
        """
        return self.get(key)

    def _load(self):
        """
        Load data from the JSON file if it exists, or initialize an empty
        database.
        """
        if (os.path.exists(self.location) and
                os.path.getsize(self.location) > 0):
            try:
                with open(self.location, "rb") as f:
                    self.db = orjson.loads(f.read())
            except Exception as e:
                raise RuntimeError(f"{e}\nFailed to load database.")
        else:
            self.db = {}

    def save(self, option=0):
        """
        Save the database to the file using an atomic save.

        Args:
            options (int): `orjson.OPT_*` flags to configure
                           serialization behavior.

        Behavior:
            - Writes to a temporary file and replaces the
              original file only after the write is successful,
              ensuring data integrity.

        Returns:
            bool: True if save was successful, False if not.
        """
        temp_location = f"{self.location}.tmp"
        try:
            with open(temp_location, "wb") as temp_file:
                temp_file.write(orjson.dumps(self.db, option=option))
            os.replace(temp_location, self.location)
            return True
        except Exception as e:
            print(f"Failed to save database: {e}")
            return False

    def set(self, key, value):
        """
        Add or update a key-value pair in the database.

        Args:
            key (any): The key to set. If the key is not a string, it
                       will be converted to a string.
            value (any): The value to associate with the key.

        Behavior:
            - If the key already exists, its value will be updated.
            - If the key does not exist, it will be added to the
              database.

        Returns:
            bool: True if the operation succeeds.
        """
        key = str(key) if not isinstance(key, str) else key
        self.db[key] = value
        return True

    def remove(self, key):
        """
        Remove a key and its value from the database.

        Args:
            key (any): The key to delete. If the key is not a string,
                       it will be converted to a string.

        Returns:
            bool: True if the key was deleted, False if the key does
                  not exist.
        """
        key = str(key) if not isinstance(key, str) else key
        if key in self.db:
            del self.db[key]
            return True
        return False

    def purge(self):
        """
        Clear all keys from the database.

        Returns:
            bool: True if the operation succeeds.
        """
        self.db.clear()
        return True

    def get(self, key):
        """
        Get the value associated with a key.

        Args:
            key (any): The key to retrieve. If the key is not a
                       string, it will be converted to a string.

        Returns:
            any: The value associated with the key, or None if the
            key does not exist.
        """
        key = str(key) if not isinstance(key, str) else key
        return self.db.get(key)

    def all(self):
        """
        Get a list of all keys in the database.

        Returns:
            list: A list of all keys.
        """
        return list(self.db.keys())
    
    def find_nearest_embeddings_parallel(self, input_embedding, n, metric='euclidean'):
        """
        Find n nearest embeddings using multiprocessing (brute-force) 

        Args:
            input_embedding (list): Embedding input to compare.
            n (int): Number of nearest embeddings need to be found.
            metric (str): 'euclidean' or 'cosine'.

        Returns:
            list: List n nearest embeddings including: (key, value và distance).
        """
        if not isinstance(input_embedding, list):
            raise ValueError("Input embedding must be a list.")
        
        if len(self.db) == 0:
            raise ValueError("Database is empty. Please add embeddings first.")
        
        nearest = find_nearest_neighbors(self.db, input_embedding, n, metric)
        return nearest
    
    def find_nearest_embeddings(self, input_embedding, n, metric='euclidean'):
        """
        Find n nearest embeddings (brute-force) 

        Args:
            input_embedding (list): Embedding input to compare.
            n (int): Number of nearest embeddings need to be found.
            metric (str): 'euclidean' or 'cosine'.

        Returns:
            list: List n nearest embeddings including: (key, value và distance).
        """
        if not isinstance(input_embedding, list):
            raise ValueError("Input embedding must be a list.")
        
        if len(self.db) == 0:
            raise ValueError("Database is empty. Please add embeddings first.")
        
        nearest = find_nearest_neighbors_simple(self.db, input_embedding, n, metric)
        return nearest