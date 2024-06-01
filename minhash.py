from collections import defaultdict
import pickle
from typing import Hashable, Iterable, Generator, Mapping, Set

import numpy as np
import numpy.typing as npt

from index import DBIndex


NestedArray = Iterable[Iterable[Hashable]]
MinHashSignature = npt.NDArray[np.floating]


def extract_distinct(
    nested_list: NestedArray,
) -> Set[Hashable]:
    """Return list of unique elements from a nested list"""
    return set(token for tokens in nested_list for token in tokens)


class MinHasher:
    """Hashes strings of text using the MinHash algorithm

    Uses a given "corpus" represented by a list containing lists of
    k-length "shingles" of strings to generate a set of hash functions.
    The length of each hash will be the given band_length times n_bands.

    Args:
        shingled_corpus (Iterable[Iterable[Hashable]]): List of lists in
            which each sublist contains each substring of some string of
            text
        band_length (int): The length of each array that each hash will
            be separated into
        n_bands (int): Number of arrays to separate each hash into

    Attributes:
        mapping (Mapping[str, int]): Mapping of each unique substring
            from the given shingled corpus to a number
        band_length (int): The length of each array that each hash will
            be separated into
        n_bands (int): Number of arrays to separate each hash into
        hash_functions (np.ArrayLike[float]): Array which will be used
            to compute the minhash on any new text.
    """

    def __init__(self, shingled_corpus: NestedArray, band_length: int, n_bands: int):
        self.mapping = self._create_mapping(shingled_corpus)
        self.band_length = band_length
        self.n_bands = n_bands
        self.hash_functions = self._init_functions()

    def _create_mapping(self, corpus: NestedArray) -> Mapping[str, int]:
        return {element: i for i, element in enumerate(extract_distinct(corpus))}

    def _init_functions(self) -> npt.NDArray:
        n_dimensions = len(self.mapping)
        hash_size = self.band_length * self.n_bands
        return np.random.randint(n_dimensions, size=[hash_size, 2])

    def _permute(self, feature: npt.ArrayLike) -> npt.NDArray:
        feature = np.array([1, feature])
        return np.dot(self.hash_functions, feature) % len(self.mapping)

    def create_signature(self, array: Iterable[str]) -> MinHashSignature:
        """Computes the minhash for a given array of strings"""
        sig = np.full(self.hash_functions.shape[0], np.inf)
        for element in array:
            if not element in self.mapping.keys():
                continue
            orig_row = self.mapping[element]
            curr_col = self._permute(orig_row)
            sig = np.minimum(sig, curr_col)
        return sig

    def hash_arrays(
        self, arrays: Iterable[npt.ArrayLike]
    ) -> Generator[MinHashSignature, None, None]:
        """Computes minhash for each array in arrays"""
        for array in arrays:
            yield self.create_signature(array)


class MinHashLsh:
    """Builds & queries index of hashes computed using MinHash algorithm

    Stores an index object which points to a table in a database
    containing MinHash bands and their corresponding values and the
    hash functions used to generate those hashes.

    Args:
        connection_string (str): String to be used by the DBIndex object
            to connect to a sqlite database.
        table_name (str): Name of the table in the database to be build
            and/or queried

    Attributes:
        hasher (None | MinHasher): Initialized as None. Set to the
            MinHasher object initialized while building the index
        index (DBIndex): Object used to interact with table in database
    """

    def __init__(self, connection_string: str, table_name: str):
        self.hasher = None
        self.index = DBIndex(connection_string, table_name)

    def hash_to_bytes(self, corpus: NestedArray) -> Generator[bytes, None, None]:
        """Compute Minhashes, returning each band of array as a bytestring

        Given a list containing lists of strings, compute the MinHash
        using the hasher generated while building the LSH object's
        index. Lists of strings should be of same length "k" as those
        used to build the index. Each MinHash is separated into bands
        based on the band length and number of bands given when the
        index was built. Each band is then returned as a bytestring.

        Args:
            corpus (NestedArray): The new data to be hashed.

        Yields:
            Generator[bytes, None, None]: A MinHash represented by a bytestring
        """
        bands, length = self.hasher.n_bands, self.hasher.band_length
        for hashed_row in self.hasher.hash_arrays(corpus):
            for band in hashed_row.reshape(bands, length):
                yield band.tobytes()

    def _check_overwrite(self, overwrite: bool) -> None:
        if overwrite is False and self.index.row_count() > 0:
            raise ValueError(
                f"Table {self.index.table} is already populated. Set overwrite to True \
                to build a new index and overwrite the data"
            )
        if overwrite is True:
            self.index.clear_index()

    def build_index(
        self,
        corpus: NestedArray,
        band_length: int,
        n_bands: int,
        overwrite: bool = False,
    ) -> int:
        """Create or clear the index and populate it with the given data

        The given data will be used to create a new index for searching.
        If the table exists with data and the overwrite argument is not
        set to True, a ValueError will be raised. A MinHasher will be
        initialized and used to hash the given data, then populate the
        index with rows of hashes and the indices of data hashed to them,
        both stored as bytestrings of numpy arrays.

        Args:
            corpus (NestedArray): List of shingled texts to hash and build
                the index with
            band_length (int): Length of each band in a hash
            n_bands (int): Number of bands to include in a hash. Each hash
                will be of size n_bands x band_length
            overwrite (bool, optional): Set to True to wipe and replace a
                table which has already been populated. Defaults to False.

        Returns:
            int: Count of rows inserted
        """
        self._check_overwrite(overwrite)
        self.hasher = MinHasher(corpus, band_length, n_bands)
        hash_map = defaultdict(list)
        for index, vector in enumerate(self.hasher.hash_arrays(corpus)):
            banded_vector = vector.reshape(self.hasher.n_bands, self.hasher.band_length)
            for band in banded_vector:
                hash_map[band.tobytes()].append(index)
        pairs = [
            (bytestring, np.array(index_list).tobytes())
            for bytestring, index_list in hash_map.items()
        ]
        return self.index.batch_insert(pairs)

    def query(self, banded_vector: NestedArray) -> Iterable[int]:
        """Return list of all values retrieved from index for each band.

        Iterates over the banded vector, encoding each band, accessing
        the value associated with the encoded band, decoding that value,
        and adding its contents to a list. The banded vector should be
        created from a list of strings passed to the create_signature()
        method of this object's hasher.

        Args:
            banded_vector (NestedArray): 2-dimensional array of integers
                hashed using the MinHashLsh object's hasher

        Returns:
            Iterable[int]: List of each unique value retrieved from the
                object's index
        """
        candidates = []
        for band in banded_vector:
            match = self.index[band.tobytes()]
            if not match:
                continue
            candidates.extend(np.frombuffer(match, dtype=int) + 1)
        return list(set(candidates))

    def find_candidates(self, data: NestedArray) -> Mapping[int, Iterable[Hashable]]:
        """Create a mapping of indices in array to candidate matches

        Data should be a list of lists of k-length shingles where k is
        equal to the length of shingles used when building the index.
        Each list of shingles is hashed and separated into bands using
        the same hash functions, band length, and number of bands used
        to build the index. Each band is queried and results are added
        to a list mapped to the index of the list of shingles.

        Args:
            data (NestedArray): Shingled strings to hash and find similar
                items for.

        Returns:
            Mapping[int, Iterable[Hashable]]: Dictionary mapping indices
                from to_query to candidate matches
        """
        candidates = defaultdict(list)
        for i, vector in enumerate(self.hasher.hash_arrays(data)):
            banded_vector = vector.reshape(self.hasher.n_bands, self.hasher.band_length)
            candidates[i] = self.query(banded_vector)
        return candidates

    @classmethod
    def load(cls, object_path: str):
        """Initialize from a previously built MinHash LSH index"""
        with open(object_path, "rb") as pfile:
            lsh = pickle.load(pfile)
        return lsh

    def save(self, object_path: str) -> None:
        """Save the LSH object by pickling it to object_path"""
        with open(object_path, "wb") as pfile:
            pickle.dump(self, pfile)
