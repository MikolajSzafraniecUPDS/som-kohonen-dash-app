"""
Functions reused in different scripts
"""

import os
import pickle

from SOM.SOM import SelfOrganizingMap

CACHE_DIR = "./cache"


def write_som_to_cache(session_id: str, som: SelfOrganizingMap) -> None:
    file_name = os.path.join(CACHE_DIR, "som_{0}.pickle".format(session_id))
    with open(file_name, 'wb') as f:
        pickle.dump(som, f)


def get_som_from_cache(session_id: str) -> SelfOrganizingMap:
    file_name = os.path.join(CACHE_DIR, "som_{0}.pickle".format(session_id))
    with open(file_name, 'rb') as f:
        som = pickle.load(f)

    return som


def rm_som_from_cache(session_id: str) -> SelfOrganizingMap:
    file_name = os.path.join(CACHE_DIR, "som_{0}.pickle".format(session_id))
    os.remove(file_name)
