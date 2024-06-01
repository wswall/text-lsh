from collections import defaultdict
import pickle
from typing import Any, Generator, Hashable, Iterable, Mapping, Callable, Tuple

import numpy as np
import numpy.typing as npt
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer

from .index import DBIndex


SimilarityFunction = Callable[[npt.NDArray, npt.ArrayLike], npt.NDArray]


def ceiling_division(numerator, divisor):
    "Return numerator divided by divisor, always rounded up"
    return -(-numerator // divisor)


def chunk_sparse_matrix(matrix, chunk_size):
    "Iterate over a sparse matrix, yielding chunk_size arrays"
    chunks = ceiling_division(matrix.shape[0], chunk_size)
    for i in range(chunks):
        yield matrix[i * chunk_size : (i + 1) * chunk_size]


def most_similar(similarity_matrix) -> npt.NDArray:
    "Return indices where value is equal to the maximum value"
    return np.where(similarity_matrix == similarity_matrix.max())[0]


def hamming_sim(matrix, vector) -> int:
    """Return count of positions at which vectors are not equal"""
    return 1 - np.count_nonzero(matrix != vector, axis=1) / vector.size


def cosine_sim(vector1: npt.NDArray[Any], vector2: npt.NDArray[Any]) -> float:
    """Compute similarity of two vectors based angle"""
    dot = np.dot(vector1, vector2)
    axis = 1 if len(vector1.shape) > 1 else 0
    norm = np.linalg.norm(vector1, axis=axis) * np.linalg.norm(vector2)
    return np.nan_to_num(1 - np.arccos(dot / norm) / np.pi)


def euclidean_sim(
    vector1: npt.NDArray[Any], vector2: npt.NDArray[Any]
) -> npt.NDArray[Any]:
    """Return inverse of distance between two vectors"""
    return 1 / (1 + np.linalg.norm(vector1 - vector2))


class RandomProjectionHasher:
    """Hashes strings of text using random projections

    Fits a count vectorizer on the given corpus and creates a hash_size number
    of planes of size equal to the number of unique tokens in the corpus.
    Hashes can then be computed using create_signature on vectorized texts.

    Args:
        shingled_corpus (Iterable[str]): List of strings to fit the vectorizer
        hash_size (int): Number of planes to generate. This will be the length
            of each hash output by create_signature.

    Attributes:
        CHUNKSIZE: Number of rows to read into dense matrixes from a sparse
            matrix. Note that each dense matrix will be of size
            CHUNKSIZE x vocab length.
        vectorizer (CountVectorizer): Vectorizer fit on the corpus passed to
            the __init__() function.
        hash_functions (np.ArrayLike[float]): Array which will be used to
            compute the hashes on any new text.
    """

    CHUNKSIZE = 1000

    def __init__(self, corpus: Iterable[str], hash_size: int):
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(corpus)
        dimensions = len(self.vectorizer.vocabulary_)
        self.hash_functions = (
            np.random.rand(hash_size, dimensions).astype("float32") - 0.5
        )

    def create_signature(self, array: npt.NDArray) -> npt.NDArray:
        """Return the hash using the random projection method.

        Calculates the dot product of each plane generated in
        self.hash_functions and the array, returning array of length
        hash_size where value is 1 if dot product of that plane and
        vector were positive, 0 if not.

        Args:
            array (npt.NDArray): The vector to be hashed

        Returns:
            npt.NDArray: The hashed vector
        """
        dot = np.einsum(
            "...i,ij", array, self.hash_functions.T, optimize="optimal"
        )
        return np.where(dot > 0, 1, 0).astype(np.byte)

    def vectorize(self, data: Iterable[str]) -> spmatrix:
        """Transform text to array of counts of each word in the vocabulary"""
        return self.vectorizer.transform(data).astype("float32")

    def hash_arrays(
        self, arrays: Iterable[npt.ArrayLike]
    ) -> Generator[npt.NDArray, None, None]:
        """Hash arrays in chunks, yielding hashes one row at a time"""
        for matrix in chunk_sparse_matrix(
            self.vectorize(arrays), self.CHUNKSIZE
        ):
            hashes = self.create_signature(matrix.toarray())
            for hashed_row in hashes:
                yield hashed_row


class RandomProjectionLsh:
    """Builds & queries index of hashes computed using random projections

    Stores an index object which points to a table in a database containing
    hashes and their corresponding values in addition to the hash functions
    used to generate those hashes. Provides a method for searching for
    nearest neighbors of one or more texts. If the index_name passed does
    not exist in the database it will be created. The table can be populated
    using build_index and queried with find_candidates.

    Args:
        connection_string (str): String to be used by the DBIndex object
            to connect to a sqlite database.
        table_name (str): Name of the table in the database to be build
            and/or queried

    Attributes:
        hasher (None | RandomProjectionHasher): Initialized as None. Set
            to the RandomProjectionHasher object initialized while building
            the index
        index (DBIndex): Object used to interact with table in database
    """

    _METRICS = {
        "hamming": hamming_sim,
        "euclidean": euclidean_sim,
        "cosine": cosine_sim,
    }

    def __init__(self, db_path: str, index_name: str):
        self.hasher = None
        self.index = DBIndex(db_path, index_name)
        self.buckets = None

    def _get_buckets(self) -> npt.NDArray:
        hashes = self.index.get_hashes()
        return np.array(
            [np.frombuffer(bucket[0], dtype=np.byte) for bucket in hashes]
        )

    def _init_hasher(self, corpus: Iterable[str], hash_size: int) -> None:
        self.hasher = RandomProjectionHasher(corpus, hash_size)

    def _check_overwrite(self, overwrite: bool) -> None:
        if overwrite is False and self.index.row_count() > 0:
            raise ValueError(
                f"Table {self.index.table} already populated. Set overwrite to True replace"
            )
        if overwrite is True:
            self.index.clear_index()

    def _build_hashmap(self, data):
        # Build mapping of hashes to indices
        hash_map = defaultdict(list)
        for i, vect in enumerate(self.hasher.hash_arrays(data)):
            hash_map[vect.tobytes()].append(i)
        return hash_map

    def _get_hash_index_pairs(self, data: npt.ArrayLike) -> Iterable[Tuple[bytes, bytes]]:
        hash_map = self._build_hashmap(data)
        return [(k, np.array(v).tobytes()) for k, v in hash_map.items()]

    def build_index(
        self, data: Iterable[str], hash_size: int, overwrite: bool = False
    ) -> int:
        """Create or clear the index and populate it with the given data

        The given data will be used to create a new index for searching.
        If the table exists with data and the overwrite argument is not
        set to True, a ValueError will be raised. A RandomProjectionHasher
        will be initialized and used to hash the given data, then populate
        the index with rows of hashes and the indices of data hashed to them,
        both stored as bytestrings of numpy arrays.

        Args:
            data (Iterable[str]): List of texts to build the index with
            hash_size (int): Length of each hash
            overwrite (bool, optional): Set to True to wipe and replace a
                table which has already been populated. Defaults to False.

        Returns:
            int: Count of rows inserted

        Raises:
            ValueError
        """
        self._check_overwrite(overwrite)
        self._init_hasher(data, hash_size)
        pairs = self._get_hash_index_pairs(data)
        count = self.index.batch_insert(pairs)
        self.buckets = self._get_buckets()
        return count


    def _build_query_matrix(self, to_query: npt.ArrayLike):
        # Return the hashed query strings as an array
        return np.array([row for row in self.hasher.hash_arrays(to_query)])

    def _get_bucket_values(self, bucket_index: int) -> npt.NDArray:
        # Get the values associated with a hash and return them as an array
        bucket = self.buckets[bucket_index].tobytes()
        return np.frombuffer(self.index[bucket], dtype=int)

    def _get_most_similar(
        self, metric: SimilarityFunction, query_vector: npt.ArrayLike,
    ):
        # Score buckets against the query_vector and return a list of all
        # values in each bucket with score equal to the maximum
        candidates = []
        similarity_matrix = metric(self.buckets, query_vector)
        for bucket_index in most_similar(similarity_matrix):
            bucket_values = self._get_bucket_values(bucket_index)
            candidates.extend(bucket_values)
        return candidates

    def find_candidates(
        self, to_query: Iterable[str], metric: str = "hamming"
    ) -> Mapping[int, Iterable[Hashable]]:
        """Create a mapping of indices in array to candidate matches

        Compiles a matrix with from the hashes of each string in to_query
        with shape equal to the hash length x length of to_query. For each
        hash in the matrix, the metric is computed for all buckets in the
        index. Values associated with the bucket(s) with the highest score
        are combined into a list and mapped to the position of the query
        in to_query.

        Args:
            to_query (Iterable[str]): Texts to look for similar
                items to
            metric (str, optional): Metric to compute most simliar items.
                Can be one of hamming, euclidean, or cosine. Defaults to
                "hamming".

        Returns:
            Mapping[int, Iterable[Hashable]]: Dictionary mapping indices
                from to_query to candidate matches
        """
        metric = self._METRICS[metric]
        candidate_dict = defaultdict(list)
        query_matrix = self._build_query_matrix(to_query)
        for i, query_vector in enumerate(query_matrix):
            if not to_query[i]:
                candidate_dict[i] = []
                continue
            # Each result is the index of a bucket with highest similarity
            candidates = self._get_most_similar(metric, query_vector)
            candidate_dict[i].extend(candidates)
        return candidate_dict

    @staticmethod
    def load(object_path: str) -> None:
        """Initialize from a previously built LSH index"""
        with open(object_path, "rb") as pfile:
            lsh = pickle.load(pfile)
        return lsh

    def save(self, object_path: str) -> None:
        """Save the LSH object by pickling it to object_path"""
        with open(object_path, "wb") as pfile:
            pickle.dump(self, pfile)
