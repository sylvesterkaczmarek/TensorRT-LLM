#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
from pathlib import Path, PureWindowsPath
from unittest import mock

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "legacy_utils.py"


@pytest.fixture()
def mod():
    spec = importlib.util.spec_from_file_location("legacy_utils_windows_paths", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_path_uses_posix_separator_for_windows_absolute_path(mod):
    repo_root = PureWindowsPath(r"C:\repo")
    filepath = r"C:\repo\tensorrt_llm\serve\tool_parser\base_tool_parser.py"

    with mock.patch.object(mod, "Path", PureWindowsPath):
        normalized = mod.normalize_path(filepath, repo_root)

    assert normalized == "tensorrt_llm/serve/tool_parser/base_tool_parser.py"


def test_normalize_path_uses_posix_separator_for_windows_relative_path(mod):
    filepath = r"tests\unittest\llmapi\apps\test_tool_parsers.py"

    with mock.patch.object(mod, "Path", PureWindowsPath):
        normalized = mod.normalize_path(filepath, PureWindowsPath(r"C:\repo"))

    assert normalized == "tests/unittest/llmapi/apps/test_tool_parsers.py"
