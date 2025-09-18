import torch
import pytest
import numpy as np
from numba import cuda
import math
from aster_invasion.filament_funcs import update_filament_states
from aster_invasion.filament_funcs import (
    change_filament_lengths,
    check_for_sorted_and_uniqueness,
    point_line_dist_sqr_device,
    get_catastrophe_rate_device,
    find_filaments_out_of_bounds,
    calc_inhib_func,
    HALF_PI,
)


def test_change_filament_lengths_grow_and_shrink():
    # Create a dummy filaments tensor: 5 filaments, 10 features each
    filaments = torch.zeros((5, 10), dtype=torch.float64)
    # Set initial lengths
    filaments[:, 7] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    # Indices to grow and shrink
    growing_idx = torch.tensor([0, 2])
    shrinking_idx = torch.tensor([1, 3])
    # Parameters
    params = {"vg": 0.5, "vs": 0.2, "dt": 2.0}
    vg = params["vg"]
    vs = params["vs"]
    dt = params["dt"]

    change_filament_lengths(filaments, growing_idx, shrinking_idx, params)

    # Check grown filaments
    assert filaments[0, 7] == pytest.approx(1.0 + vg * dt)
    assert filaments[2, 7] == pytest.approx(3.0 + vg * dt)
    # Check shrunk filaments
    assert filaments[1, 7] == pytest.approx(2.0 - vs * dt)
    assert filaments[3, 7] == pytest.approx(4.0 - vs * dt)
    # Unchanged filament
    assert filaments[4, 7] == pytest.approx(5.0)


def test_change_filament_lengths_empty_indices():
    filaments = torch.zeros((3, 10), dtype=torch.float64)
    filaments[:, 7] = torch.tensor([1.0, 2.0, 3.0])
    params = {"vg": 1.0, "vs": 1.0, "dt": 1.0}
    growing_idx = torch.tensor([], dtype=torch.int)
    shrinking_idx = torch.tensor([], dtype=torch.int)

    change_filament_lengths(filaments, growing_idx, shrinking_idx, params)

    # No changes expected
    assert torch.allclose(
        filaments[:, 7], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    )


def test_change_filament_lengths_overlap_indices():
    filaments = torch.zeros((2, 10), dtype=torch.float64)
    filaments[:, 7] = torch.tensor([5.0, 10.0])
    params = {"vg": 2.0, "vs": 1.0, "dt": 1.0}
    # Overlapping index: filament 0 will both grow and shrink
    growing_idx = torch.tensor([0])
    shrinking_idx = torch.tensor([0, 1])

    # Make sure error is raised
    with pytest.raises(ValueError):
        change_filament_lengths(filaments, growing_idx, shrinking_idx, params)


def test_check_for_sorted_and_uniqueness_sorted_unique():
    fil_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float64)
    # Should not raise
    check_for_sorted_and_uniqueness(fil_ids)


def test_check_for_sorted_and_uniqueness_not_sorted():
    fil_ids = torch.tensor([1, 3, 2, 4, 5], dtype=torch.float64)
    with pytest.raises(ValueError, match="Filament ids must be sorted and unique."):
        check_for_sorted_and_uniqueness(fil_ids)


def test_check_for_sorted_and_uniqueness_not_unique():
    fil_ids = torch.tensor([1, 2, 2, 3, 4], dtype=torch.float64)
    with pytest.raises(ValueError, match="Filament ids must be sorted and unique."):
        check_for_sorted_and_uniqueness(fil_ids)


def test_check_for_sorted_and_uniqueness_empty():
    fil_ids = torch.tensor([], dtype=torch.float64)
    # Should not raise for empty input
    check_for_sorted_and_uniqueness(fil_ids)


def test_check_for_sorted_and_uniqueness_single_element():
    fil_ids = torch.tensor([42], dtype=torch.float64)
    # Should not raise for single element
    check_for_sorted_and_uniqueness(fil_ids)
    # Import the device function from the module under test


# Helper CUDA kernel to call the device function and store result in output array
@cuda.jit
def call_point_line_dist_sqr_device_kernel(point, line_seg, out):
    # Only one thread needed
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        out[0] = point_line_dist_sqr_device(point, line_seg)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    "point,minus_end,direction,length,expected",
    [
        # Point projects onto the segment (side)
        (
            np.array([1.0, 0.0, 0.0], dtype=np.float64),  # point position
            np.array([0.0, 0.0, 0.0], dtype=np.float64),  # filament minus end
            np.array([1.0, 0.0, 0.0], dtype=np.float64),  # filament direction
            2.0,  # filament length
            0.0,  # point is on the segment
        ),
        # Point before the minus end (behind)
        (
            np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            2.0,
            1.0,  # distance squared from -1 to 0
        ),
        # Point after the plus end (in front)
        (
            np.array([3.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            2.0,
            1.0,  # distance squared from 3 to 2
        ),
        # Point off the line, projects onto the segment (side)
        (
            np.array([1.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            2.0,
            1.0,  # perpendicular distance squared is 1
        ),
        # Point off the line, before minus end (behind)
        (
            np.array([-1.0, 2.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            2.0,
            5.0,  # sqrt((1)^2 + (2)^2) = sqrt(5), squared is 5
        ),
        # Point off the line, after plus end (in front)
        (
            np.array([3.0, -2.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            2.0,
            5.0,  # sqrt((1)^2 + (2)^2) = sqrt(5), squared is 5
        ),
        # 3D case: point projects onto segment
        (
            np.array([0.0, 1.0, 1.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            2.0,
            1.0,  # perpendicular distance squared is 1 (z direction)
        ),
    ],
)
def test_point_line_dist_sqr_device_cuda(point, minus_end, direction, length, expected):
    # Prepare line_seg as [minus_end(3), direction(3), length]
    line_seg = np.concatenate(
        [minus_end, direction, np.array([length], dtype=np.float64)]
    ).astype(np.float64)
    # Allocate device arrays
    d_point = cuda.to_device(point)
    d_line_seg = cuda.to_device(line_seg)
    d_out = cuda.device_array(1, dtype=np.float64)
    # Launch kernel
    call_point_line_dist_sqr_device_kernel[1, 1](d_point, d_line_seg, d_out)
    out = d_out.copy_to_host()[0]
    assert math.isclose(out, expected, rel_tol=1e-12)


# Helper CUDA kernel to call the device function and store result in output array
@cuda.jit
def call_get_catastrophe_rate_device_kernel(
    tip_idx, filaments, inhib_range, inhib_strength, ang_senstv, out
):
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        out[0] = get_catastrophe_rate_device(
            tip_idx, filaments, inhib_range, inhib_strength, ang_senstv
        )


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is not available")
def test_get_catastrophe_rate_device_no_near_filaments():
    # 2 filaments, tip_idx=0, both far apart
    filaments = np.zeros((2, 10), dtype=np.float64)
    # Filament 0: minus end at (0,0,0), direction (1,0,0), length 5
    filaments[0, 1:4] = [0, 0, 0]
    filaments[0, 4:7] = [1, 0, 0]
    filaments[0, 7] = 5.0
    # Filament 1: minus end at (100,0,0), direction (1,0,0), length 5
    filaments[1, 1:4] = [100, 0, 0]
    filaments[1, 4:7] = [1, 0, 0]
    filaments[1, 7] = 5.0
    d_filaments = cuda.to_device(filaments)
    d_out = cuda.device_array(1, dtype=np.float64)
    # Check filament 0 (tip_idx=0) with inhib_range=1.0, inhib_strength=2.0, ang_senstv=1.0
    call_get_catastrophe_rate_device_kernel[1, 1](0, d_filaments, 1.0, 2.0, 1.0, d_out)
    out = d_out.copy_to_host()[0]
    assert out == pytest.approx(0.0)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is not available")
def test_get_catastrophe_rate_device_one_near_filament():
    # 2 filaments, tip_idx=0, filament 1 is close
    filaments = np.zeros((2, 10), dtype=np.float64)
    filaments[0, 1:4] = [0, 0, 0]
    filaments[0, 4:7] = [1, 0, 0]
    filaments[0, 7] = 5.0
    filaments[1, 1:4] = [8, 0, 0]  # close to tip of filament 0
    filaments[1, 4:7] = [-1, 0, 0]
    filaments[1, 7] = 5.0
    d_filaments = cuda.to_device(filaments)
    d_out = cuda.device_array(1, dtype=np.float64)
    call_get_catastrophe_rate_device_kernel[1, 1](0, d_filaments, 2.0, 2.0, 1.0, d_out)
    out = d_out.copy_to_host()[0]
    assert out > 0.0


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is not available")
def test_get_catastrophe_rate_device_tip_filament_too_short():
    # Filament length < 2*inhib_range, should return 0
    filaments = np.zeros((2, 10), dtype=np.float64)
    filaments[0, 1:4] = [0, 0, 0]
    filaments[0, 4:7] = [1, 0, 0]
    filaments[0, 7] = 1.0  # less than 2*inhib_range
    filaments[1, 1:4] = [0, 0, 0]
    filaments[1, 4:7] = [1, 0, 0]
    filaments[1, 7] = 5.0
    d_filaments = cuda.to_device(filaments)
    d_out = cuda.device_array(1, dtype=np.float64)
    call_get_catastrophe_rate_device_kernel[1, 1](0, d_filaments, 1.0, 2.0, 1.0, d_out)
    out = d_out.copy_to_host()[0]
    assert out == pytest.approx(0.0)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is not available")
def test_get_catastrophe_rate_device_multiple_near_filaments():
    # 3 filaments, tip_idx=0, filaments 1 and 2 are close
    filaments = np.zeros((3, 10), dtype=np.float64)
    filaments[0, 1:4] = [0, 0, 0]
    filaments[0, 4:7] = [1, 0, 0]
    filaments[0, 7] = 5.0

    filaments[1, 1:4] = [5.0, -3.0, 0]
    filaments[1, 4:7] = [0, 1, 0]
    filaments[1, 7] = 5.0

    filaments[2, 1:4] = [4.0, 0, -4.5]
    filaments[2, 4:7] = [0, 0, 1]
    filaments[2, 7] = 5.0
    d_filaments = cuda.to_device(filaments)
    d_out = cuda.device_array(1, dtype=np.float64)
    call_get_catastrophe_rate_device_kernel[1, 1](0, d_filaments, 2.0, 2.0, 1.0, d_out)
    out = d_out.copy_to_host()[0]
    assert out > 0.0


def test_find_filaments_out_of_bounds_none_out():
    # 3 filaments, all within bounds
    filaments = torch.zeros((3, 10), dtype=torch.float64)
    filaments[:, 0] = torch.tensor([10, 20, 30])  # filament ids
    filaments[:, 1:4] = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [-1, -1, -1]], dtype=torch.float64
    )
    filaments[:, 4:7] = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64
    )
    filaments[:, 7] = torch.tensor([1.0, 1.0, 1.0])
    bounds = [2.0, 2.0, 2.0]
    out_ids = find_filaments_out_of_bounds(filaments, bounds)
    assert out_ids.shape[0] == 0


def test_find_filaments_out_of_bounds_some_out():
    # 3 filaments, one out of bounds in x, one in y, one in z
    filaments = torch.zeros((3, 10), dtype=torch.float64)
    filaments[:, 0] = torch.tensor([101, 102, 103])
    filaments[:, 1:4] = torch.tensor(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float64
    )
    filaments[:, 4:7] = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64
    )
    filaments[:, 7] = torch.tensor([5.0, 5.0, 5.0])
    bounds = [2.0, 2.0, 2.0]
    out_ids = find_filaments_out_of_bounds(filaments, bounds)
    # All tips at (5,0,0), (0,5,0), (0,0,5) so all out of bounds
    assert set(out_ids.tolist()) == {101, 102, 103}


def test_find_filaments_out_of_bounds_mixed():
    filaments = torch.zeros((4, 10), dtype=torch.float64)
    filaments[:, 0] = torch.tensor([1, 2, 3, 4])
    filaments[:, 1:4] = torch.tensor(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float64
    )
    filaments[:, 4:7] = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=torch.float64
    )
    filaments[:, 7] = torch.tensor([1.0, 3.0, 0.5, 1.0])
    bounds = [2.0, 2.0, 2.0]
    # Tips: (1,0,0), (0,3,0), (0,0,0.5), (1,1,1)
    out_ids = find_filaments_out_of_bounds(filaments, bounds)
    # Only filament 2 (id=2) tip is out of bounds in y
    assert out_ids.tolist() == [2]


def test_find_filaments_out_of_bounds_negative_bounds():
    # Negative bounds mean no bounds, so nothing should be out
    filaments = torch.zeros((2, 10), dtype=torch.float64)
    filaments[:, 0] = torch.tensor([11, 12])
    filaments[:, 1:4] = torch.tensor([[100, 0, 0], [0, 100, 0]], dtype=torch.float64)
    filaments[:, 4:7] = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float64)
    filaments[:, 7] = torch.tensor([10.0, 10.0])
    bounds = [-1.0, -1.0, -1.0]
    out_ids = find_filaments_out_of_bounds(filaments, bounds)
    assert out_ids.shape[0] == 0


def test_find_filaments_out_of_bounds_zero_bounds():
    # Zero bounds mean no filament can be in bounds except at origin
    filaments = torch.zeros((2, 10), dtype=torch.float64)
    filaments[:, 0] = torch.tensor([21, 22])
    filaments[:, 1:4] = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float64)
    filaments[:, 4:7] = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float64)
    filaments[:, 7] = torch.tensor([0.0, 0.0])
    bounds = [0.0, 0.0, 0.0]
    out_ids = find_filaments_out_of_bounds(filaments, bounds)
    assert out_ids.shape[0] == 0


def test_calc_inhib_func_parallel():
    # Parallel vectors: dot_product = 1, acos(1) = 0
    dir_tip = np.array([1.0, 0.0, 0.0])
    dir_fil = np.array([1.0, 0.0, 0.0])
    ang_senstv = 1.0
    result = calc_inhib_func(dir_tip, dir_fil, ang_senstv)
    expected = 1.0 / (1.0 + math.exp(-ang_senstv * (0 - HALF_PI)))
    assert math.isclose(result, expected, rel_tol=1e-12)


def test_calc_inhib_func_antiparallel():
    # Antiparallel vectors: dot_product = -1, acos(-1) = pi
    dir_tip = np.array([1.0, 0.0, 0.0])
    dir_fil = np.array([-1.0, 0.0, 0.0])
    ang_senstv = 2.0
    result = calc_inhib_func(dir_tip, dir_fil, ang_senstv)
    expected = 1.0 / (1.0 + math.exp(-ang_senstv * (math.pi - HALF_PI)))
    assert math.isclose(result, expected, rel_tol=1e-12)


def test_calc_inhib_func_perpendicular():
    # Perpendicular vectors: dot_product = 0, acos(0) = HALF_PI
    dir_tip = np.array([1.0, 0.0, 0.0])
    dir_fil = np.array([0.0, 1.0, 0.0])
    ang_senstv = 0.5
    result = calc_inhib_func(dir_tip, dir_fil, ang_senstv)
    expected = 1.0 / (1.0 + math.exp(-ang_senstv * (HALF_PI - HALF_PI)))
    assert math.isclose(result, 0.5, rel_tol=1e-12)
    assert math.isclose(result, expected, rel_tol=1e-12)


def test_calc_inhib_func_zero_sensitivity():
    # ang_senstv = 0, should always return 0.5
    dir_tip = np.array([1.0, 0.0, 0.0])
    dir_fil = np.array([0.0, 1.0, 0.0])
    ang_senstv = 0.0
    result = calc_inhib_func(dir_tip, dir_fil, ang_senstv)
    assert math.isclose(result, 0.5, rel_tol=1e-12)


def test_calc_inhib_func_various_angles():
    # Test for a range of angles
    ang_senstv = 1.0
    for theta in [0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi]:
        dir_tip = np.array([1.0, 0.0, 0.0])
        dir_fil = np.array([math.cos(theta), math.sin(theta), 0.0])
        result = calc_inhib_func(dir_tip, dir_fil, ang_senstv)
        expected = 1.0 / (
            1.0
            + math.exp(
                -ang_senstv
                * (math.acos(np.clip(np.dot(dir_tip, dir_fil), -1, 1)) - HALF_PI)
            )
        )
        assert math.isclose(result, expected, rel_tol=1e-12)


def make_filaments(n, ids=None, lengths=None):
    filaments = torch.zeros((n, 10), dtype=torch.float64)
    if ids is not None:
        filaments[:, 0] = torch.tensor(ids, dtype=torch.float64)
    if lengths is not None:
        filaments[:, 7] = torch.tensor(lengths, dtype=torch.float64)
    return filaments


def test_update_filament_states_no_bounds_no_inhib():
    filaments = make_filaments(3, ids=[1, 2, 3], lengths=[1.0, 2.0, 3.0])
    growing_idx = torch.tensor([0, 1], dtype=torch.int64)
    paused_idx = torch.tensor([2], dtype=torch.int64)
    params = {"kc": 0.0, "dt": 1.0, "inhib_range": 0.0, "inhib_strength": 0.0}
    out_filaments, unbind_nuc = update_filament_states(
        filaments, growing_idx, paused_idx, params
    )
    # No catastrophe, no removal
    assert torch.all(out_filaments[:, 7] == torch.tensor([1.0, 2.0, 3.0]))
    assert unbind_nuc.shape[0] == 0


def test_update_filament_states_with_bounds_pausing():
    filaments = make_filaments(2, ids=[10, 20], lengths=[1.0, 1.0])
    filaments[0, 1:4] = torch.tensor([2.0, 0.0, 0.0])
    filaments[1, 1:4] = torch.tensor([0.0, 3.0, 0.0])
    filaments[0, 4:7] = torch.tensor([1.0, 0.0, 0.0])
    filaments[1, 4:7] = torch.tensor([0.0, 1.0, 0.0])
    growing_idx = torch.tensor([0, 1], dtype=torch.int64)
    paused_idx = torch.tensor([], dtype=torch.int64)
    params = {
        "kc": 0.0,
        "dt": 1.0,
        "inhib_range": 0.0,
        "inhib_strength": 0.0,
        "Lx": 2.0,
        "Ly": 2.0,
        "Lz": 2.0,
    }
    out_filaments, unbind_nuc = update_filament_states(
        filaments, growing_idx, paused_idx, params
    )
    # Both tips are out of bounds, so both should be paused (state=0)
    assert torch.all(out_filaments[:, 8] == 0)
    assert unbind_nuc.shape[0] == 0


def test_update_filament_states_catastrophe_removal():
    filaments = make_filaments(2, ids=[1, 2], lengths=[-0.5, 1.0])
    filaments[0, 8] = -1  # depolymerizing
    filaments[0, 9] = 42  # nucleator id
    growing_idx = torch.tensor([1], dtype=torch.int64)
    paused_idx = torch.tensor([], dtype=torch.int64)
    params = {"kc": 0.0, "dt": 1.0, "inhib_range": 0.0, "inhib_strength": 0.0}
    out_filaments, unbind_nuc = update_filament_states(
        filaments, growing_idx, paused_idx, params
    )
    # Filament 0 should be removed, nucleator id returned
    assert out_filaments.shape[0] == 1
    assert torch.all(out_filaments[:, 0] == 2)
    assert unbind_nuc.tolist() == [42]


def test_update_filament_states_aberrant_filament_warning(capfd):
    filaments = make_filaments(2, ids=[1, 2], lengths=[-1.0, 2.0])
    filaments[0, 8] = 1  # not depolymerizing
    filaments[0, 9] = 99
    growing_idx = torch.tensor([1], dtype=torch.int64)
    paused_idx = torch.tensor([], dtype=torch.int64)
    params = {"kc": 0.0, "dt": 1.0, "inhib_range": 0.0, "inhib_strength": 0.0}
    update_filament_states(filaments, growing_idx, paused_idx, params)
    out = capfd.readouterr()
    assert "Warning: Filament with length <0 that is not depolymerizing" in out.err


def test_update_filament_states_no_filaments():
    filaments = make_filaments(0)
    growing_idx = torch.tensor([], dtype=torch.int64)
    paused_idx = torch.tensor([], dtype=torch.int64)
    params = {"kc": 0.0, "dt": 1.0, "inhib_range": 0.0, "inhib_strength": 0.0}
    out_filaments, unbind_nuc = update_filament_states(
        filaments, growing_idx, paused_idx, params
    )
    assert out_filaments.shape[0] == 0
    assert unbind_nuc.shape[0] == 0


def test_update_filament_states_catastrophe_fast_path(monkeypatch):
    filaments = make_filaments(5, ids=[1, 2, 3, 4, 5], lengths=[1.0] * 5)
    growing_idx = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    paused_idx = torch.tensor([], dtype=torch.int64)
    params = {"kc": 10.0, "dt": 1.0, "inhib_range": 0.0, "inhib_strength": 0.0}
    # Patch torch.binomial to always return 2
    monkeypatch.setattr(torch, "binomial", lambda n, p: torch.tensor(2))
    # Patch torch.choose to select first two indices
    monkeypatch.setattr(torch, "randperm", lambda n: torch.arange(0, n))
    out_filaments, unbind_nuc = update_filament_states(
        filaments, growing_idx, paused_idx, params
    )
    # Filaments 0 and 1 should be set to -1 in state
    assert out_filaments[0, 8] == -1
    assert out_filaments[1, 8] == -1
