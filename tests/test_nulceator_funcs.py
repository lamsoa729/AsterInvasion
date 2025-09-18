import pytest
import torch
from aster_invasion.nulceator_funcs import (
    update_nucleators,
    rotation_matrix_from_vectors,
    gen_rand_dir_vecs,
    gen_rand_dir_vecs_2d,
)


@pytest.fixture
def filaments():
    # 3 filaments: id, pos(3), dir(3), length, state, nuc_id, cent_id
    # Only length (col 7) is relevant for update_nucleators
    arr = torch.tensor(
        [
            [0, 0, 0, 0, 1, 0, 0, 5.0, 1, -1, 0, 0],  # filament 0, length 5
            [1, 0, 0, 0, 1, 0, 0, 10.0, 1, -1, 0, 0],  # filament 1, length 10
            [2, 0, 0, 0, 1, 0, 0, 15.0, 1, -1, 0, 0],  # filament 2, length 15
        ],
        dtype=torch.float64,
    )
    return arr


@pytest.fixture
def nucleators():
    # 4 nucleators: id, fil_id, fil_nuc_id, bind_pos
    arr = torch.tensor(
        [
            [0, -1, -1, -1],  # unbound
            [1, -1, -1, -1],  # unbound
            [2, 1, -1, 2.0],  # bound to filament 1 at pos 2.0
            [3, 2, -1, 20.0],  # bound to filament 2 at pos 20.0 (should detach)
        ],
        dtype=torch.float64,
    )
    return arr


@pytest.fixture
def params():
    return {
        "kb": 0.5,
        "dt": 0.1,
    }


def test_update_nucleators_binds_unbound_nucleators(
    monkeypatch, filaments, nucleators, params
):
    # Patch np.random.binomial to always bind 1 nucleator
    # monkeypatch.setattr(np.random, "binomial", lambda n, p: min(n, 1))
    monkeypatch.setattr(torch, "binomial", lambda n, p: torch.tensor(1))
    # Patch torch.multinomial to always pick filament 0
    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda weights, num_samples, replacement: torch.zeros(
            num_samples, dtype=torch.long
        ),
    )
    # Patch torch.rand to always return 0.5
    monkeypatch.setattr(torch, "rand", lambda n: torch.full((n,), 0.5))

    nucleators_copy = nucleators.clone()
    update_nucleators(
        filaments, nucleators_copy, torch.empty((0,), dtype=torch.long), params
    )
    # One of the unbound nucleators should now be bound to filament 0
    bound = nucleators_copy[nucleators_copy[:, 1] == 0]
    assert bound.shape[0] == 1
    assert bound[0, 3] == pytest.approx(0.5 * filaments[0, 7])


def test_update_nucleators_unbinds_nucleators(filaments, nucleators, params):
    nucleators_copy = nucleators.clone()
    # Unbind nucleator at index 2
    update_nucleators(filaments, nucleators_copy, torch.tensor([2]), params)
    assert (nucleators_copy[2, 1:] == -1).all()


def test_update_nucleators_detaches_nucleators_when_filament_too_short(
    filaments, nucleators, params
):
    nucleators_copy = nucleators.clone()
    # nucleator at index 3 is bound to filament 2 at pos 20.0, but filament 2 length is 15.0
    update_nucleators(
        filaments, nucleators_copy, torch.empty((0,), dtype=torch.long), params
    )
    assert (nucleators_copy[3, 1:] == -1).all()


def test_update_nucleators_no_binding_when_no_unbound_nucleators(
    filaments, nucleators, params
):
    nucleators_copy = nucleators.clone()
    # Set all nucleators as bound
    nucleators_copy[:, 1] = 1
    nucleators_copy[-1, -1] = 10  # Set last nucleator to be bound at position 10
    update_nucleators(
        filaments, nucleators_copy, torch.empty((0,), dtype=torch.long), params
    )
    # No nucleators should be newly bound (all already bound)
    assert (nucleators_copy[:, 1] == 1).all()


def test_update_nucleators_raises_on_zero_total_length(nucleators, params):
    filaments = torch.zeros((2, 12), dtype=torch.float64)
    nucleators_copy = nucleators.clone()
    with pytest.raises(AssertionError):
        update_nucleators(
            filaments, nucleators_copy, torch.empty((0,), dtype=torch.long), params
        )


def test_rotation_matrix_identity():
    # Rotating a vector to itself should yield the identity matrix
    v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    rot = rotation_matrix_from_vectors(v, v)
    assert torch.allclose(rot, torch.eye(3, dtype=torch.float64), atol=1e-8)


def test_rotation_matrix_90_deg():
    # Rotating x-axis to y-axis should yield a 90 degree rotation about z
    v1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    v2 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    rot = rotation_matrix_from_vectors(v1, v2)
    result = rot @ v1
    assert torch.allclose(result, v2, atol=1e-8)


def test_rotation_matrix_reverse():
    # Rotating a vector to its negative should flip direction
    v1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    v2 = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64)
    rot = rotation_matrix_from_vectors(v1, v2)
    result = rot @ v1
    assert torch.allclose(result, v2, atol=1e-8)


def test_rotation_matrix_arbitrary():
    # Rotating between arbitrary vectors
    v1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    v2 = torch.tensor([-3.0, 2.0, 1.0], dtype=torch.float64)
    rot = rotation_matrix_from_vectors(v1, v2)
    result = rot @ (v1 / torch.norm(v1))
    expected = v2 / torch.norm(v2)
    assert torch.allclose(result, expected, atol=1e-8)
    return


def test_gen_rand_dir_vecs_output_shape_and_type(monkeypatch):
    # Patch torch.normal to return zeros for deterministic output
    monkeypatch.setattr(torch, "normal", lambda mean, std, size: torch.zeros(size))
    dir_vec = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    std_ang = 0.1
    result = gen_rand_dir_vecs(dir_vec, std_ang)
    assert result.shape == (2, 3)
    assert result.dtype == torch.float64
    # All vectors should be normalized
    norms = torch.norm(result, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-8)


def test_gen_rand_dir_vecs_rotates_z_to_dir(monkeypatch):
    # Patch torch.normal to return zeros for deterministic output
    monkeypatch.setattr(torch, "normal", lambda mean, std, size: torch.zeros(size))
    dir_vec = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    std_ang = 0.1
    result = gen_rand_dir_vecs(dir_vec, std_ang)
    # With zero angle perturbation and z direction, should return [0,0,1]
    assert torch.allclose(
        result[0], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), atol=1e-8
    )

    # Test with x direction
    dir_vec = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    result = gen_rand_dir_vecs(dir_vec, std_ang)
    # Should be normalized and point roughly in x direction
    assert torch.allclose(
        torch.norm(result[0]), torch.tensor(1.0).to(result[0].dtype), atol=1e-8
    )
    assert abs(result[0][0]) > 0.99  # x component should dominate


def test_gen_rand_dir_vecs_2d_output_shape_and_type(monkeypatch):
    # Patch torch.normal to return zeros for deterministic output
    monkeypatch.setattr(torch, "normal", lambda mean, std, size: torch.zeros(size))
    dir_vec = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    std_ang = 0.1
    result = gen_rand_dir_vecs_2d(dir_vec, std_ang)
    assert result.shape == (2, 3)
    assert result.dtype == torch.float64
    # All vectors should be normalized
    norms = torch.norm(result, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-8)


def test_gen_rand_dir_vecs_2d_direction(monkeypatch):
    # Patch torch.normal to return zeros for deterministic output
    monkeypatch.setattr(torch, "normal", lambda mean, std, size: torch.zeros(size))
    # x direction
    dir_vec = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    result = gen_rand_dir_vecs_2d(dir_vec, 0.1)
    # Should be [1,0,0] normalized
    assert torch.allclose(
        result[0], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-8
    )
    # y direction
    dir_vec = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    result = gen_rand_dir_vecs_2d(dir_vec, 0.1)
    assert torch.allclose(
        result[0], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-8
    )
    # -x direction
    dir_vec = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float64)
    result = gen_rand_dir_vecs_2d(dir_vec, 0.1)
    assert torch.allclose(
        result[0], torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-8
    )


def test_gen_rand_dir_vecs_2d_angle_perturbation(monkeypatch):
    # Patch torch.normal to return pi/2 for deterministic output
    monkeypatch.setattr(
        torch, "normal", lambda mean, std, size: torch.full(size, torch.pi / 2)
    )
    dir_vec = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    std_ang = 0.1
    result = gen_rand_dir_vecs_2d(dir_vec, std_ang)
    # Should rotate x direction by pi/2 to y direction
    print(result[0])
    assert torch.allclose(
        result[0], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-7
    )
