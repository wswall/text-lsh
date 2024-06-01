import sqlite3
from typing import Iterable, Tuple


class DBIndex:
    """Object for managing queries to an index stored in a database

    Intended to be used as the backend/data source for MinhashLsh and
    RandomProjectionLsh classes. The connection string is expected to
    point to a sqlite database. If no table exists with the given name
    for argument "table_name", then a table will be created when the
    object is initialized

    Args:
        connection_string (str): String to use for creating a connection
            to a database
        table_name (str): Name of table to query in the database

    Attributes:
        connection_string (str): String used to create connection
        connection (sqlite3 Connection): The connection used to initialize
            cursors, commit quries, or roll back queries
        cursor (sqlite3 Cursor): Cursor object used to execute queries
        table (str): Name of table to query in the database

    """

    def __init__(
        self,
        connection_string: str,
        table_name: str,
    ):
        self.connection_string = connection_string
        self.connection = sqlite3.connect(self.connection_string)
        self.cursor = self.connection.cursor()
        self.table = table_name
        self._create_table()

    def _create_table(self) -> None:
        # Creates a table for the index in the connected database
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                hash_bytestring BLOB PRIMARY KEY,
                index_list BLOB NOT NULL
            ) WITHOUT ROWID;
        """)

    def __getstate__(self):
        # Can't pickle connection and cursor
        self.connection.commit()
        self.connection.close()
        state = self.__dict__.copy()
        del state["connection"]
        del state["cursor"]
        return state

    def __setstate__(self, state):
        # Add connection and cursor attributes back in when unpickling.
        self.__dict__.update(state)
        self.connection = sqlite3.connect(self.connection_string)
        self.cursor = self.connection.cursor()

    def __getitem__(self, bytestring):
        self.cursor.execute(
            f"SELECT index_list FROM {self.table} WHERE hash_bytestring=?", [bytestring]
        )
        res = self.cursor.fetchone()
        return res if res is None else res[0]

    def __setitem__(self, bytestring, index_list):
        self.cursor.execute(
            f"INSERT INTO {self.table} VALUES (?, ?)", (bytestring, index_list)
        )

    def get_hashes(self) -> Iterable[Tuple[str]]:
        """Get all hash_bytestring values from the index table"""
        self.cursor.execute(f"SELECT hash_bytestring FROM {self.table}")
        return self.cursor.fetchall()

    def row_count(self) -> int:
        """Return count of rows in index"""
        self.cursor.execute(f"SELECT COUNT (*) from {self.table}")
        return self.cursor.fetchone()[0]

    def batch_insert(self, pairs: Iterable[Tuple[str, str]]) -> int:
        """Insert each hash, value pair into index table, return row count"""
        self.cursor.executemany(
            f"INSERT INTO {self.table} values (?, ?)", pairs
        )
        self.connection.commit()
        return self.row_count()

    def clear_index(self) -> int:
        """Delete all rows from index table"""
        self.cursor.execute(f"DELETE FROM {self.table}")
        return self.row_count()
