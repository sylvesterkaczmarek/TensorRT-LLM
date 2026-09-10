# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm.llmapi import utils


@pytest.fixture
def affinity(monkeypatch):
    state = SimpleNamespace(pid=1234, mask=list(range(8)))
    process = SimpleNamespace(cpu_affinity=lambda: list(state.mask))
    monkeypatch.setattr(utils, "psutil", SimpleNamespace(
        Process=lambda pid: process, cpu_count=lambda: 8))
    monkeypatch.setattr(utils, "os", SimpleNamespace(
        getpid=lambda: state.pid, environ={}))
    monkeypatch.setattr(utils, "logger", Mock())
    monkeypatch.setattr(utils, "_numa_affinity_state", None, raising=False)
    monkeypatch.setattr(utils, "get_numa_aware_cpu_affinity", Mock(
        side_effect=lambda device: list(range(device * 4, device * 4 + 4))))

    def bind(cpus):
        state.mask = list(cpus)
        return 3, 3

    monkeypatch.setattr(utils, "_set_affinity_all_threads", Mock(side_effect=bind))
    return state


def test_repeated_configuration_keeps_numa_affinity(affinity):
    utils.configure_cpu_affinity(0)
    utils.configure_cpu_affinity(0)
    assert affinity.mask == [0, 1, 2, 3]
    assert utils._set_affinity_all_threads.call_count == 2
    assert all(call.args[0] == [0, 1, 2, 3]
               for call in utils._set_affinity_all_threads.call_args_list)
    utils.logger.warning.assert_not_called()


def test_forced_configuration_is_recognized_without_override(affinity):
    affinity.mask = [5, 6]
    utils.os.environ["TLLM_NUMA_AWARE_WORKER_AFFINITY"] = "1"
    utils.configure_cpu_affinity(0)
    utils.os.environ.clear()
    utils.logger.reset_mock()
    utils.configure_cpu_affinity(0)
    assert affinity.mask == [0, 1, 2, 3]
    utils.logger.warning.assert_not_called()


def test_external_constraint_keeps_existing_default_policy(affinity):
    affinity.mask = [0, 2]
    utils.configure_cpu_affinity(0)
    assert affinity.mask == list(range(8))
    utils.get_numa_aware_cpu_affinity.assert_not_called()
    assert utils._numa_affinity_state is None


def test_external_change_invalidates_owned_mask(affinity):
    utils.configure_cpu_affinity(0)
    affinity.mask = [4, 6]
    utils.configure_cpu_affinity(0)
    assert affinity.mask == list(range(8))
    assert utils._numa_affinity_state is None
    assert utils.get_numa_aware_cpu_affinity.call_count == 1


def test_external_change_is_not_remembered_after_opt_out(affinity):
    utils.configure_cpu_affinity(0)
    utils.os.environ["TLLM_NUMA_AWARE_WORKER_AFFINITY"] = "0"
    affinity.mask = [4, 6]
    utils.configure_cpu_affinity(0)
    assert utils._numa_affinity_state is None
    affinity.mask = [0, 1, 2, 3]
    utils.os.environ.clear()
    utils.configure_cpu_affinity(0)
    assert affinity.mask == list(range(8))


def test_inherited_state_from_another_pid_is_not_owned(affinity):
    utils.configure_cpu_affinity(0)
    affinity.pid += 1
    utils.configure_cpu_affinity(0)
    assert affinity.mask == list(range(8))
    assert utils._numa_affinity_state is None


@pytest.mark.parametrize("setting", ["0", "false", "disabled"])
def test_opt_out_does_not_rebind_or_query_numa(affinity, setting):
    utils.configure_cpu_affinity(0)
    utils.os.environ["TLLM_NUMA_AWARE_WORKER_AFFINITY"] = setting
    utils.configure_cpu_affinity(0)
    assert affinity.mask == [0, 1, 2, 3]
    assert utils.get_numa_aware_cpu_affinity.call_count == 1
    assert utils._set_affinity_all_threads.call_count == 1


def test_sequential_devices_recompute_the_target(affinity):
    utils.configure_cpu_affinity(0)
    utils.configure_cpu_affinity(1)
    assert affinity.mask == [4, 5, 6, 7]
    assert utils._numa_affinity_state == (affinity.pid, frozenset([4, 5, 6, 7]))


def test_failed_binding_does_not_claim_ownership(affinity):
    utils._set_affinity_all_threads.side_effect = None
    utils._set_affinity_all_threads.return_value = (0, 3)
    utils.configure_cpu_affinity(0)
    assert utils._numa_affinity_state is None
    assert affinity.mask == list(range(8))


def test_failed_main_thread_binding_does_not_claim_ownership(affinity):
    utils._set_affinity_all_threads.side_effect = None
    utils._set_affinity_all_threads.return_value = (2, 3)
    utils.configure_cpu_affinity(0)
    assert utils._numa_affinity_state is None
    assert affinity.mask == list(range(8))


def test_effective_cpuset_intersection_is_remembered(affinity):
    def bind(cpus):
        affinity.mask = sorted(set(cpus) & {0, 2})
        return 3, 3

    utils._set_affinity_all_threads.side_effect = bind
    utils.configure_cpu_affinity(0)
    utils.configure_cpu_affinity(0)
    assert utils._numa_affinity_state == (affinity.pid, frozenset([0, 2]))
    assert all(call.args[0] == [0, 1, 2, 3]
               for call in utils._set_affinity_all_threads.call_args_list)


def test_mask_order_does_not_change_ownership(affinity):
    utils.configure_cpu_affinity(0)
    affinity.mask.reverse()
    utils.configure_cpu_affinity(0)
    assert affinity.mask == [0, 1, 2, 3]
    utils.logger.warning.assert_not_called()
