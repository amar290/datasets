# coding=utf-8
# Copyright 2022 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base DatasetCollectionTestCase to test a DatasetCollection base class."""
import typing
from typing import List, Optional, Type

import pytest
from tensorflow_datasets.core import dataset_collection_builder
from tensorflow_datasets.core import registered


class TestDatasetCollectionBuilder:
  """Inherit this class to test your DatasetCollection class.

  You must set the following class attribute:
    * DATASET_COLLECTION_CLASS: class object of DatasetCollection to test.

  You may set the following class attribute:
    * VERSION: The version of the DatasetCollection used to run the test.
      Defaults to None (latest version).
    * DATASETS_TO_TEST: List containing the datasets to test existance for. If
      no dataset is spcified, the existance of all datasets in the collection
      will be tested.
    * CHECK_DATASETS_VERSION: Whether to check for the existance of the
      versioned datasets in the dataset collection, or for their default
      versions. Defaults to True.

  This test case will check for the following:
  - The dataset collection is correctly registered, i.e.
    `tfds.dataset_collection` works.
  - The datasets in the dataset collection exist, i.e. their builder is
    registered in tfds. If DATASETS_TO_TEST is not specified, all datasets in
    the collection will be checked.
  """
  DATASET_COLLECTION_CLASS = None
  VERSION: Optional[str] = None
  DATASETS_TO_TEST: Optional[List[str]] = None
  CHECK_DATASETS_VERSION: bool = True

  @pytest.fixture
  def dataset_collection(self):
    dataset_collection_cls = registered.imported_dataset_collection_cls(
        self.DATASET_COLLECTION_CLASS.name)
    dataset_collection_cls = typing.cast(
        Type[dataset_collection_builder.DatasetCollection],
        dataset_collection_cls)
    return dataset_collection_cls().get_collection(self.VERSION)

  @pytest.fixture
  def datasets(self, dataset_collection):
    return dataset_collection.get_collection(self.VERSION)

  def test_dataset_collection_is_registered(self, dataset_collection):
    """Checks that the dataset collection is registered."""
    assert registered.is_dataset_collection(dataset_collection.info.name)

  def test_dataset_collection_info(self, dataset_collection):
    """Checks that the collection's info is of type `DatasetCollectionInfo`."""
    assert isinstance(dataset_collection.info,
                      dataset_collection_builder.DatasetCollectionInfo)
