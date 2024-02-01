"""
Functions reused in different scripts
"""

import os
import pickle

from SOM.SOM import SelfOrganizingMap

CACHE_DIR = "./cache"


def store_som_in_cache(session_id: str, som: SelfOrganizingMap) -> None:
    """
    Store SelfOrganizingMap object in disc cache - we need separated
    som per each user's session.

    :param session_id: id of user session stored in the UI Div
    :param som: SOM to store
    """
    file_name = os.path.join(CACHE_DIR, "som_{0}.pickle".format(session_id))
    with open(file_name, 'wb') as f:
        pickle.dump(som, f)


def get_som_from_cache(session_id: str) -> SelfOrganizingMap:
    """
    Retrieve SOM object from cache. If object doesn't exist (for
    example it was removed during the cache clean) new SOM is created
    and save in cache.

    :param session_id: id of user's session
    :return: som object
    """
    file_name = os.path.join(CACHE_DIR, "som_{0}.pickle".format(session_id))
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            som = pickle.load(f)
    else:
        som = SelfOrganizingMap()
        store_som_in_cache(session_id, som)

    return som


def rm_som_from_cache(session_id: str) -> None:
    """
    Remove som object when session is closed

    :param session_id: id of user's session
    """
    file_name = os.path.join(CACHE_DIR, "som_{0}.pickle".format(session_id))
    os.remove(file_name)
