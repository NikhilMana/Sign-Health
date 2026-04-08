"""Tests for the shared.device_config module."""

import sys
import pytest
import numpy as np
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
from shared.device_config import DeviceManager


class TestDeviceManagerInitialize:
    """DeviceManager.initialize() should always return a valid strategy."""

    def test_initialize_returns_strategy(self):
        dm = DeviceManager(enable_mixed_precision=False, enable_xla=False)
        strategy = dm.initialize()
        assert isinstance(strategy, tf.distribute.Strategy)

    def test_repeated_initialize_returns_same_strategy(self):
        dm = DeviceManager(enable_mixed_precision=False, enable_xla=False)
        s1 = dm.initialize()
        s2 = dm.initialize()
        assert s1 is s2

    def test_device_tag_is_string(self):
        dm = DeviceManager(enable_mixed_precision=False, enable_xla=False)
        dm.initialize()
        assert dm.device_tag in ("GPU", "CPU")

    def test_num_gpus_is_non_negative(self):
        dm = DeviceManager(enable_mixed_precision=False, enable_xla=False)
        dm.initialize()
        assert dm.num_gpus >= 0


class TestBuildDataset:
    """DeviceManager.build_dataset() should create well-shaped tf.data pipelines."""

    def test_output_shapes_match_inputs(self):
        x = np.random.randn(100, 30, 1662).astype(np.float32)
        y = np.random.randn(100, 10).astype(np.float32)
        ds = DeviceManager.build_dataset(x, y, batch_size=16, training=False)

        for batch_x, batch_y in ds.take(1):
            assert batch_x.shape[1:] == (30, 1662)
            assert batch_y.shape[1:] == (10,)
            assert batch_x.shape[0] <= 16

    def test_training_dataset_is_shuffled(self):
        """Two iterations of a training dataset should have different orders."""
        x = np.arange(200).reshape(200, 1).astype(np.float32)
        y = np.zeros((200, 1), dtype=np.float32)
        ds = DeviceManager.build_dataset(x, y, batch_size=200, training=True)

        epoch1 = next(iter(ds))[0].numpy().flatten()
        epoch2 = next(iter(ds))[0].numpy().flatten()
        # It's astronomically unlikely that two shuffles produce the same order
        assert not np.array_equal(epoch1, epoch2)

    def test_validation_dataset_not_shuffled(self):
        """A non-training dataset should be deterministic."""
        x = np.arange(50).reshape(50, 1).astype(np.float32)
        y = np.zeros((50, 1), dtype=np.float32)
        ds = DeviceManager.build_dataset(x, y, batch_size=50, training=False)

        epoch1 = next(iter(ds))[0].numpy().flatten()
        epoch2 = next(iter(ds))[0].numpy().flatten()
        assert np.array_equal(epoch1, epoch2)

    def test_custom_batch_size(self):
        x = np.random.randn(33, 5).astype(np.float32)
        y = np.random.randn(33, 2).astype(np.float32)
        ds = DeviceManager.build_dataset(x, y, batch_size=8, training=False)
        batches = list(ds)
        # 33 / 8 = 4 full batches + 1 partial
        assert len(batches) == 5
        assert batches[-1][0].shape[0] == 1  # remainder


class TestCPUFallback:
    """When no GPU is present the manager should fall back gracefully."""

    def test_cpu_fallback_strategy(self):
        dm = DeviceManager(enable_mixed_precision=False, enable_xla=False)
        strategy = dm.initialize()
        # On a CPU-only machine this should be the default strategy
        if dm.num_gpus == 0:
            assert dm.device_tag == "CPU"
            assert isinstance(strategy, tf.distribute.Strategy)


class TestMixedPrecisionProperty:
    """is_mixed_precision_active should reflect the global policy."""

    def test_disabled_by_default_when_off(self):
        dm = DeviceManager(enable_mixed_precision=False, enable_xla=False)
        dm.initialize()
        assert dm.is_mixed_precision_active is False
