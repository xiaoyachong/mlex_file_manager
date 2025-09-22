import concurrent
import glob
import os
from concurrent.futures import ThreadPoolExecutor
from functools import reduce

import numpy as np
from PIL import Image

from file_manager.dataset.dataset import Dataset

# List of allowed and not allowed formats
FORMATS = [
    "**/*.[pP][nN][gG]",
    "**/*.[jJ][pP][gG]",
    "**/*.[jJ][pP][eE][gG]",
    "**/*.[tT][iI][fF]",
    "**/*.[tT][iI][fF][fF]",
]
NOT_ALLOWED_FORMATS = [
    "**/__pycache__/**",
    "**/.*",
    "cache/",
    "cache/**/",
    "cache/**",
    "tiled_local_copy/",
    "**/tiled_local_copy/**",
    "**/tiled_local_copy/**/",
    "mlexchange_store/**/",
    "mlexchange_store/**",
    "labelmaker_outputs/**/",
    "labelmaker_outputs/**",
]


class FileDataset(Dataset):
    def __init__(
        self,
        uri,
        cumulative_data_count,
        filenames=[],
    ):
        """
        Definition of a file data set
        """
        super().__init__(uri, cumulative_data_count)
        self.filenames = filenames
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
            "filenames": self.filenames,
        }

    @classmethod
    def from_dict(cls, dataset_dict):
        return cls(
            dataset_dict["uri"],
            dataset_dict["cumulative_data_count"],
            dataset_dict["filenames"],
        )

    @classmethod
    def _read_data_point(
        cls,
        root_uri,
        filename,
        export="base64",
        resize=True,
        log=False,
        percentiles=[0, 100],
    ):
        """
        Read data point
        Args:
            root_uri:          Root URI from which data should be retrieved
            filename:          Filename of the image to retrieve
            export:            Export format, defaults to base64
            resize:            Resize image to 200x200, defaults to True
            log:               Apply log to the images, defaults to False
            percentiles:       Percentiles for normalization, defaults to [0, 100]
        Returns:
            Base64/PIL image
            Dataset URI
        """
        file_path = os.path.join(root_uri, filename)
        img = Image.open(file_path)
        img = np.array(img, dtype=np.float32)
        img = cls._process_image(img, log, resize, export, percentiles)
        return img

    def read_data(
        self,
        root_uri,
        indices,
        export="base64",
        resize=True,
        log=False,
        just_uri=False,
        percentiles=[0, 100],
        **kwargs,
    ):
        """
        Read data set
        Args:
            root_uri:          Root URI from which data should be retrieved
            indices:           List of indexes of the images to retrieve
            export:            Export format, defaults to base64
            resize:            Resize images, defaults to True
            log:               Apply log to the images, defaults to False
            just_uri:          Return only the uri, defaults to False
            percentiles:       Percentiles for normalization, defaults to [0, 100]
        Returns:
            Base64/PIL image
            Dataset URI
        """
        results = []
        # Filter filenames to process based on indices, ensuring they are within bounds
        filenames_to_process = [
            self.uri + "/" + self.filenames[i]
            for i in indices
            if i < len(self.filenames)
        ]

        if just_uri:
            return [f"{root_uri}/{filename}" for filename in filenames_to_process]

        thread_indexes = []
        # Use ThreadPoolExecutor to read files in parallel
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_index = {
                executor.submit(
                    self._read_data_point,
                    root_uri,
                    filename,
                    export,
                    resize,
                    log,
                    percentiles=percentiles,
                ): index
                for index, filename in enumerate(filenames_to_process)
            }

            for future in concurrent.futures.as_completed(future_to_index):
                thread_index = future_to_index[future]
                result = future.result()
                results.append(result)
                thread_indexes.append(thread_index)

        ordered_results = [
            results[thread_indexes.index(i)] for i in range(len(indices))
        ]
        return ordered_results, [
            f"{root_uri}/{filename}" for filename in filenames_to_process
        ]

    def get_uri_index(self, uri):
        """
        Get index of the URI
        Args:
            uri:          URI of the image
        Returns:
            Index of the URI
        """
        filename = uri.split(self.uri, 1)[-1]
        return self.filenames.index(filename[1:])

    @staticmethod
    def filepaths_from_directory(
        directory,
        formats=FORMATS,
        selected_sub_uris=[""],
        sort=True,
    ):
        """
        Retrieve a list of filepaths from a given directory
        Args:
            directory:          Directory from which datapaths will be retrieved according to formats
            formats:            List of file formats/extensions of interest, defaults to FORMATS
            selected_sub_uris:  List of selected sub uris, defaults to None
            sort:               Sort output list of filepaths, defaults to True
        Returns:
            paths:              List of filepaths in directory
        """
        if type(formats) is str:  # If a single format was selected, adapt to list
            formats = [formats]

        cumulative_data_counts = []
        filenames_per_uri = []

        cumulative_dataset_size = 0
        for dataset in selected_sub_uris:
            dataset_path = os.path.join(directory, dataset)
            if os.path.isdir(dataset_path):
                # Find paths that match the format of interest
                all_paths = list(
                    reduce(
                        lambda list1, list2: list1 + list2,
                        (
                            [
                                os.path.relpath(path, start=dataset_path)
                                for path in glob.glob(
                                    str(dataset_path) + "/" + t, recursive=False
                                )
                            ]
                            for t in formats
                        ),
                    )
                )
                # Find paths that match the not allowed file/directory formats
                not_allowed_paths = list(
                    reduce(
                        lambda list1, list2: list1 + list2,
                        (
                            [
                                os.path.relpath(path, start=dataset_path)
                                for path in glob.glob(
                                    str(dataset_path) + "/" + t, recursive=False
                                )
                            ]
                            for t in NOT_ALLOWED_FORMATS
                        ),
                    )
                )
                # Remove not allowed filepaths from filepaths of interest
                paths = list(set(all_paths) - set(not_allowed_paths))
                if sort:
                    paths.sort()
                cumulative_dataset_size += len(paths)
                cumulative_data_counts.append(cumulative_dataset_size)
                filenames_per_uri.append(paths)

        if selected_sub_uris == [""]:
            return (
                filenames_per_uri[0] if len(filenames_per_uri) > 0 else [],
                [0] * len(filenames_per_uri[0]) if len(filenames_per_uri) > 0 else [],
                [[]] * len(filenames_per_uri[0]) if len(filenames_per_uri) > 0 else [],
            )
        else:
            return selected_sub_uris, cumulative_data_counts, filenames_per_uri
