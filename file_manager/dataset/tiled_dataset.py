import concurrent.futures
import os
from functools import partial

import httpx
import numpy as np
from tiled.client import from_uri
from tiled.client.array import ArrayClient
from tiled.client.cache import Cache

from file_manager.dataset.dataset import Dataset

# Check if a static tiled client has been set
STATIC_TILED_URI = os.getenv("STATIC_TILED_URI", None)
STATIC_TILED_API_KEY = os.getenv("STATIC_TILED_API_KEY", None)
print("in tiled_dataset")
if STATIC_TILED_URI:
    STATIC_TILED_CLIENT = from_uri(STATIC_TILED_URI, api_key=STATIC_TILED_API_KEY)

    cache_path = os.environ.get("TILED_CACHE_PATH")
    
    if cache_path:
        # Get capacity and max item size from environment or use defaults
        capacity = int(os.environ.get("TILED_CACHE_CAPACITY", 500_000_000))
        max_item_size = int(os.environ.get("TILED_CACHE_MAX_ITEM_SIZE", 500_000))
        
        # Create custom cache
        cache = Cache(
            capacity=capacity,
            max_item_size=max_item_size,
            filepath=cache_path,
            readonly=False,
        )
        
        # Create client with custom cache
    
        STATIC_TILED_CLIENT.context.cache = cache
        client_httpx = STATIC_TILED_CLIENT.context.http_client
        # Try bumping up the pool timeout until the error goes away.
        client_httpx.timeout = httpx.Timeout(200.0, connect=5.0, pool=5.0)
        client_httpx.limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
        print(f"https timeouts {client_httpx.timeout}")
        print(f"https limits {client_httpx.limits}")
else:
    STATIC_TILED_CLIENT = None


class TiledDataset(Dataset):
    def __init__(
        self,
        uri,
        cumulative_data_count,
    ):
        """
        Definition of a tiled data set
        """
        super().__init__(uri, cumulative_data_count)
        pass

    def to_dict(self):
        """
        Convert to dictionary
        Returns:
            Dictionary
        """
        return {
            "uri": self.uri,
            "cumulative_data_count": self.cumulative_data_count,
        }

    @classmethod
    def from_dict(cls, dataset_dict):
        """
        Create a new instance from dictionary
        Args:
            dataset_dict:           Dictionary
        Returns:
            New instance
        """
        return cls(dataset_dict["uri"], dataset_dict["cumulative_data_count"])

    @staticmethod
    def get_tiled_client(
        tiled_uri, api_key=None, static_tiled_client=STATIC_TILED_CLIENT
    ):
        """
        Get the tiled client
        Args:
            tiled_uri:              Tiled URI
            api_key:                Tiled API key
            static_tiled_client:    Static tiled client
        Returns:
            Tiled client
        """
        # Checks if a static tiled client has been set, otherwise creates a new one
        if static_tiled_client:
            return static_tiled_client
        else:
            client = from_uri(tiled_uri, api_key=api_key)
            return client

    def read_data(
        self,
        root_uri,
        indexes,
        export="base64",
        resize=True,
        log=False,
        api_key=None,
        downsample=False,
        just_uri=False,
        tiled_client=None,
        percentiles=[0, 100],
    ):
        """
        Read data set
        Args:
            root_uri:          Root URI from which data should be retrieved
            indexes:           Index or list of indexes of the images to retrieve
            export:            Export format, defaults to base64
            resize:            Resize image to 200x200, defaults to True
            log:               Apply log(1+x) to the image, defaults to False
            api_key:           Tiled API key
            downsample:        Downsample the image, defaults to False
            just_uri:          Return only the uri, defaults to False
            tiled_client:      Tiled client
            percentiles:       Percentiles to normalize the image, defaults to [0, 100]
        Returns:
            Base64/PIL image
            Dataset URI
        """
        if isinstance(indexes, int):
            indexes = [indexes]

        if tiled_client is None:
            tiled_client = self.get_tiled_client(root_uri, api_key)

        tiled_uris = self._get_tiled_uris(tiled_client, indexes)
        if just_uri:
            return tiled_uris

        tiled_data = tiled_client[self.uri]
        if downsample:
            if len(tiled_data.shape) == 4:
                block_data = tiled_data[indexes, :, ::10, ::10]
            elif len(tiled_data.shape) == 3:
                block_data = tiled_data[indexes, ::10, ::10]
            else:
                block_data = tiled_data[::10, ::10]
                block_data = np.expand_dims(block_data, axis=0)
        else:
            if len(tiled_data.shape) == 4:
                block_data = tiled_data[indexes]
            elif len(tiled_data.shape) == 3:
                block_data = tiled_data[indexes]
            else:
                block_data = tiled_data
                block_data = np.expand_dims(block_data, axis=0)

        if export == "raw":
            return block_data, tiled_uris

        # Check if there are 4 dimensions for a grayscale image
        if block_data.shape[1] == 1:
            block_data = np.squeeze(block_data, axis=1)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            data = list(
                executor.map(
                    self._read_data_point,
                    block_data,
                    [log] * len(indexes),
                    [resize] * len(indexes),
                    [export] * len(indexes),
                    [percentiles] * len(indexes),
                )
            )
        return data, tiled_uris

    def _read_data_point(self, image, log, resize, export, percentiles):
        """
        Read data point
        Args:
            image:          Image data
            log:            Apply log(masked(x)+threshold) to the image
            resize:         Resize image to 200x200
            export:         Export format
            percentiles:    Percentiles to normalize the image
            threshold:      Threshold for log
        """
        return self._process_image(image, log, resize, export, percentiles)

    def _get_tiled_uris(self, tiled_client, indexes):
        """
        Get tiled URIs
        Args:
            tiled_client:      Tiled client
            indexes:           List of indexes of the images to retrieve
        Returns:
            List of tiled URIs
        """
        tiled_metadata = tiled_client[self.uri]
        base_tiled_uri = tiled_metadata.uri
        if len(tiled_metadata.shape) > 2 and tiled_metadata.shape[0] > 1:
            base_tiled_uri.replace("/metadata/", "/array/full/")
            return [f"{base_tiled_uri}?slice={index}" for index in indexes]
        else:
            return [base_tiled_uri]

    def get_uri_index(self, uri):
        """
        Get index of the URI
        Args:
            uri:          URI of the image
        Returns:
            Index of the URI
        """
        if "slice=" not in uri:
            return 0
        return int(uri.split("slice=")[-1])

    def _check_node(tiled_client, sub_uri, node):
        """
        Checks if the sub_uri exists in the node and returns the URI
        Args:
            tiled_client:       Current tiled client, which is used when the method is run
            sub_uri:           sub_uri to filter the data
            node:               Node to process
        Returns:
            URI of the node
        """
        try:
            tiled_client[f"/{node}/{sub_uri}"]
            return f"/{node}/{sub_uri}"
        except Exception:
            return None

    @staticmethod
    def _get_node_size(tiled_client, node):
        tiled_array = tiled_client[node]
        array_shape = tiled_array.shape
        if len(array_shape) == 2:
            return 1
        else:
            return array_shape[0]

    @classmethod
    def _get_cumulative_data_count(cls, tiled_client, nodes):
        """
        Retrieve tiled data sets from list of tiled_uris
        Args:
            tiled_uris:         Tiled URIs from which data should be retrieved
            api_key:            Tiled API key
        Returns:
            Length of the data set
        """
        get_node_size_with_client = partial(cls._get_node_size, tiled_client)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            sizes = list(executor.map(get_node_size_with_client, nodes))

        cumulative_dataset_size = [sum(sizes[: i + 1]) for i in range(len(sizes))]
        return cumulative_dataset_size

    @classmethod
    def browse_data(
        cls,
        root_uri,
        api_key=None,
        sub_uri_template="",
        selected_sub_uris=[""],
    ):
        """
        Retrieve a list of nodes from tiled URI
        Args:
            root_uri:                Root URI from which data should be retrieved
            api_key:                 Tiled API key
            sub_uri_template:        Template for the sub URI
            selected_sub_uris:       List of selected sub URIs
        Returns:
            tiled_uris:              List of tiled URIs found in tiled client
            cumulative_data_counts:  Cumulative data count
        """
        tiled_client = cls.get_tiled_client(root_uri, api_key)
        if selected_sub_uris != [""]:
            # Check if the selected sub URIs are nodes
            tmp_sub_uris = []
            for sub_uri in selected_sub_uris:
                if type(tiled_client[sub_uri]) is ArrayClient:
                    tmp_sub_uris.append(sub_uri)
                else:
                    tmp_sub_uris += [
                        f"{sub_uri}/{node}" for node in tiled_client[sub_uri]
                    ]
            selected_sub_uris = tmp_sub_uris

            # Get sizes of the selected nodes
            cumulative_data_counts = cls._get_cumulative_data_count(
                tiled_client,
                selected_sub_uris,
            )
            return selected_sub_uris, cumulative_data_counts

        # Browse the tiled URI
        tiled_uris = []
        nodes = list(tiled_client)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_node = {
                executor.submit(
                    cls._check_node, tiled_client, sub_uri_template, node
                ): node
                for node in nodes
            }
            for future in concurrent.futures.as_completed(future_to_node):
                uri = future.result()
                if uri is not None:
                    tiled_uris.append(uri)
        return tiled_uris, [0] * len(tiled_uris)
