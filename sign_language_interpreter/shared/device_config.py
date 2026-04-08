"""
Centralized GPU / hardware-acceleration manager for the training pipeline.

Usage:
    from shared.device_config import DeviceManager

    dm = DeviceManager()
    strategy = dm.initialize()          # call once at program start

    with strategy.scope():
        model = create_model(...)
        model.compile(...)

    train_ds = dm.build_dataset(X_train, y_train, batch_size=64)
    val_ds   = dm.build_dataset(X_test,  y_test,  batch_size=64, training=False)

    model.fit(train_ds, validation_data=val_ds, ...)
"""

import logging
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


class DeviceManager:
    """Device-agnostic hardware configuration for TensorFlow training & inference.

    Handles:
    - Dynamic GPU detection with memory-growth configuration
    - Mixed-precision (float16 compute / float32 weights) for Tensor Core GPUs
    - XLA JIT compilation for graph-level kernel fusion
    - tf.distribute strategies for single-GPU, multi-GPU, or CPU fallback
    - tf.data pipeline construction with prefetching
    """

    def __init__(
        self,
        enable_mixed_precision: bool = True,
        enable_xla: bool = True,
        enable_memory_growth: bool = True,
    ):
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_xla = enable_xla
        self.enable_memory_growth = enable_memory_growth
        self._gpus: list = []
        self._strategy: tf.distribute.Strategy | None = None
        self._initialized = False

    # ── Public API ────────────────────────────────────────────

    def initialize(self) -> tf.distribute.Strategy:
        """Detect hardware, apply optimizations, and return a distribution strategy.

        This must be called **once** before model creation. Repeated calls
        are safe and return the cached strategy.
        """
        if self._initialized:
            return self._strategy

        self._gpus = tf.config.list_physical_devices("GPU")
        self._configure_memory_growth()
        self._configure_mixed_precision()
        self._configure_xla()
        self._strategy = self._build_strategy()
        self._initialized = True
        self._log_summary()
        return self._strategy

    @property
    def device_tag(self) -> str:
        """Human-readable label: 'GPU' or 'CPU'."""
        return "GPU" if self._gpus else "CPU"

    @property
    def num_gpus(self) -> int:
        return len(self._gpus)

    @property
    def is_mixed_precision_active(self) -> bool:
        return keras.mixed_precision.global_policy().name == "mixed_float16"

    # ── tf.data pipeline builder ──────────────────────────────

    @staticmethod
    def build_dataset(
        x,
        y,
        batch_size: int = 64,
        training: bool = True,
        shuffle_buffer: int | None = None,
    ) -> tf.data.Dataset:
        """Wrap NumPy arrays in an optimized ``tf.data.Dataset``.

        Args:
            x: Input features (NumPy array).
            y: Labels (NumPy array).
            batch_size: Mini-batch size.
            training: If True, shuffle the data each epoch.
            shuffle_buffer: Buffer size for shuffling (defaults to dataset length).

        Returns:
            A batched, prefetched ``tf.data.Dataset``.
        """
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            # Limit shuffle buffer to 1000 to prevent system OOM
            # If len(x) is huge, a full-dataset shuffle buffer duplicates it in RAM
            buffer = min(shuffle_buffer or len(x), 1000)
            ds = ds.shuffle(buffer, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=False)
        
        # Limit prefetch buffer to prevent arbitrary memory growth 
        # (AUTOTUNE can sometimes allocate too memory much on laptops)
        prefetch = 2 if tf.data.AUTOTUNE else 2 
        ds = ds.prefetch(prefetch)
        return ds

    # ── GPU memory diagnostics ────────────────────────────────

    def print_memory_usage(self):
        """Print current GPU memory allocation (if available)."""
        for gpu in self._gpus:
            try:
                info = tf.config.experimental.get_memory_info(gpu.name)
                current_mb = info["current"] / (1024 ** 2)
                peak_mb = info["peak"] / (1024 ** 2)
                print(f"  {gpu.name}  →  current: {current_mb:.0f} MB  |  peak: {peak_mb:.0f} MB")
            except Exception:
                pass  # not all backends support memory queries

    # ── Internal helpers ──────────────────────────────────────

    def _configure_memory_growth(self):
        """Enable incremental GPU memory allocation to allow co-resident processes."""
        if not (self.enable_memory_growth and self._gpus):
            return
        for gpu in self._gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                logger.warning("Could not set memory growth on %s: %s", gpu.name, exc)

    def _configure_mixed_precision(self):
        """Activate float16 compute + float32 accumulation when a GPU is present."""
        if not (self.enable_mixed_precision and self._gpus):
            return
        try:
            keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("Mixed precision enabled (mixed_float16)")
        except Exception as exc:
            logger.warning("Mixed precision setup failed: %s", exc)

    def _configure_xla(self):
        """Enable XLA just-in-time compilation for graph optimizations."""
        if not self.enable_xla:
            return
        try:
            tf.config.optimizer.set_jit(True)
            logger.info("XLA JIT compilation enabled")
        except Exception as exc:
            logger.warning("XLA setup failed: %s", exc)

    def _build_strategy(self) -> tf.distribute.Strategy:
        """Return the best-fit distribution strategy for the detected hardware."""
        if len(self._gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            logger.info("Using MirroredStrategy across %d GPUs", len(self._gpus))
        elif len(self._gpus) == 1:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            logger.info("Using OneDeviceStrategy on GPU:0")
        else:
            strategy = tf.distribute.get_strategy()  # default / CPU
            logger.info("No GPU detected — using default CPU strategy")
        return strategy

    def _log_summary(self):
        """Print a human-friendly banner summarizing the hardware config."""
        sep = "─" * 60
        lines = [
            "",
            sep,
            "  HARDWARE ACCELERATION SUMMARY",
            sep,
        ]
        if self._gpus:
            for i, gpu in enumerate(self._gpus):
                lines.append(f"  ✓ GPU {i}: {gpu.name}")
        else:
            lines.append("  ⚠ No GPU detected — training will run on CPU")

        lines.append(f"  • Mixed precision : {'ON ✓' if self.is_mixed_precision_active else 'OFF'}")
        lines.append(f"  • XLA JIT         : {'ON ✓' if self.enable_xla else 'OFF'}")
        lines.append(f"  • Strategy        : {type(self._strategy).__name__}")
        lines.append(sep)
        print("\n".join(lines))
