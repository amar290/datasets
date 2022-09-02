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

"""Tests for longt5."""
from tensorflow_datasets.dataset_collections.longt5 import longt5
from tensorflow_datasets.testing.dataset_collection_builder_testing import TestDatasetCollectionBuilder
# import tensorflow_datasets.public_api as tfds


# class TestLongt5(tfds.testing.TestDatasetCollectionBuilder):
class TestLongt5(TestDatasetCollectionBuilder):
  DATASET_COLLECTION_CLASS = longt5.Longt5


# class TestLongt5:
#   DATASET_COLLECTION_CLASS = longt5.Longt5
#   VERSION = None

#   @pytest.fixture
#   def dataset_collection(self):
#     dataset_collection_cls = registered.imported_dataset_collection_cls(
#         self.DATASET_COLLECTION_CLASS.name)
#     dataset_collection_cls = typing.cast(
#         Type[dataset_collection_builder.DatasetCollection],
#         dataset_collection_cls)
#     return dataset_collection_cls()

#   @pytest.fixture
#   def datasets(self, dataset_collection):
#     return dataset_collection.get_collection(self.VERSION)

#   def test_dataset_collection_is_registered(self, dataset_collection):
#     """Checks that the dataset collection is registered."""
#     assert registered.is_dataset_collection(dataset_collection.info.name)

#   def test_dataset_collection_info(self, dataset_collection):
#     """Checks that the collection's info is of type `DatasetCollectionInfo`."""
#     assert isinstance(dataset_collection.info,
#                       dataset_collection_builder.DatasetCollectionInfo)
