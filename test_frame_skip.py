"""Tests for frame skipping logic in video processing pipelines."""
import os
import tempfile

import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def _create_test_video(path, num_frames=90, fps=30.0, width=640, height=480):
    """Create a synthetic test video with known frame count."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    # Verify
    cap = cv2.VideoCapture(path)
    actual = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert actual == num_frames, f"Test video has {actual} frames, expected {num_frames}"


def _count_output_frames(path):
    """Count frames in an output video."""
    cap = cv2.VideoCapture(path)
    count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    return count


def _get_output_fps(path):
    """Get the FPS of a video."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


# Dummy stubs for Triton inference (avoid actual model calls)
_dummy_output = np.zeros((1, 0, 6), dtype=np.float32)


def _mock_triton_infer(client, model_name, img):
    return _dummy_output


def _mock_get_triton_client():
    return MagicMock()


class TestFrameSkipPPE:
    """Test frame skipping in _process_ppe_video."""

    @patch("main._get_triton_client", _mock_get_triton_client)
    @patch("main.triton_infer", _mock_triton_infer)
    def test_no_skip(self):
        """frame_skip=0 should produce the same number of frames as input."""
        from main import _process_ppe_video

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "output.mp4")
            _create_test_video(input_path, num_frames=90, fps=30.0)

            _process_ppe_video(input_path, output_path, frame_skip=0)

            assert _count_output_frames(output_path) == 90
            assert abs(_get_output_fps(output_path) - 30.0) < 0.5

    @patch("main._get_triton_client", _mock_get_triton_client)
    @patch("main.triton_infer", _mock_triton_infer)
    def test_skip_1(self):
        """frame_skip=1 should produce half the frames."""
        from main import _process_ppe_video

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "output.mp4")
            _create_test_video(input_path, num_frames=90, fps=30.0)

            _process_ppe_video(input_path, output_path, frame_skip=1)

            output_frames = _count_output_frames(output_path)
            # 90 frames, process every 2nd → frames 0,2,4,...,88 → 45 frames
            assert output_frames == 45
            # Output FPS should be halved to preserve duration
            assert abs(_get_output_fps(output_path) - 15.0) < 0.5

    @patch("main._get_triton_client", _mock_get_triton_client)
    @patch("main.triton_infer", _mock_triton_infer)
    def test_skip_2(self):
        """frame_skip=2 should produce 1/3 of the frames."""
        from main import _process_ppe_video

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "output.mp4")
            _create_test_video(input_path, num_frames=90, fps=30.0)

            _process_ppe_video(input_path, output_path, frame_skip=2)

            output_frames = _count_output_frames(output_path)
            # 90 frames, process every 3rd → frames 0,3,6,...,89 → 30 frames
            assert output_frames == 30
            assert abs(_get_output_fps(output_path) - 10.0) < 0.5


class TestFrameSkipTraffic:
    """Test frame skipping in _process_traffic_video."""

    @patch("main._get_triton_client", _mock_get_triton_client)
    @patch("main.triton_infer", _mock_triton_infer)
    def test_no_skip(self):
        """frame_skip=0 should produce the same number of frames as input."""
        from main import _process_traffic_video

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "output.mp4")
            _create_test_video(input_path, num_frames=90, fps=30.0)

            _process_traffic_video(input_path, output_path, frame_skip=0)

            assert _count_output_frames(output_path) == 90
            assert abs(_get_output_fps(output_path) - 30.0) < 0.5

    @patch("main._get_triton_client", _mock_get_triton_client)
    @patch("main.triton_infer", _mock_triton_infer)
    def test_skip_1(self):
        """frame_skip=1 should produce half the frames."""
        from main import _process_traffic_video

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "output.mp4")
            _create_test_video(input_path, num_frames=90, fps=30.0)

            _process_traffic_video(input_path, output_path, frame_skip=1)

            output_frames = _count_output_frames(output_path)
            assert output_frames == 45
            assert abs(_get_output_fps(output_path) - 15.0) < 0.5

    @patch("main._get_triton_client", _mock_get_triton_client)
    @patch("main.triton_infer", _mock_triton_infer)
    def test_skip_2(self):
        """frame_skip=2 should produce 1/3 of the frames."""
        from main import _process_traffic_video

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "output.mp4")
            _create_test_video(input_path, num_frames=90, fps=30.0)

            _process_traffic_video(input_path, output_path, frame_skip=2)

            output_frames = _count_output_frames(output_path)
            assert output_frames == 30
            assert abs(_get_output_fps(output_path) - 10.0) < 0.5
