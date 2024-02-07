import os
import glob


def clear_cache() -> None:
    """
    Remove all pickled SOM from cache directory. For some browser (for example
    Firefox) beforeunload event doesn't work while closing whole window instead
    of single card, that's why we need to do a cleanage.
    """
    cache_path = os.path.join("som-app", "cache")
    if os.path.exists(cache_path):
        # List pickled SOM
        pickle_pattern = os.path.join(cache_path, "*.pickle")
        files_to_rm = glob.glob(
            pickle_pattern
        )

        for file_path in files_to_rm:
            os.remove(file_path)


if __name__ == "__main__":
    clear_cache()
