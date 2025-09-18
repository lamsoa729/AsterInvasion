import pytest
import torch
from aster_invasion.centrosome_funcs import nucleate_filaments_from_centrosomes


class DummyParams(dict):
    def get(self, key, default=None):
        return self[key] if key in self else default


@pytest.fixture
def basic_params():
    return DummyParams(
        {
            "nuc_rate": 1.0,
            "dt": 1.0,
            "diam": 2.0,
            "dimension": 3,
        }
    )


@pytest.fixture
def centrosomes():
    # columns: [id, x, y, z, radius]
    return torch.tensor(
        [
            [0, 0.0, 0.0, 0.0, 1.0],
            [1, 5.0, 5.0, 5.0, 2.0],
        ],
        dtype=torch.float64,
    )


@pytest.fixture
def filaments():
    # 11 columns, empty initially
    return torch.zeros((0, 11), dtype=torch.float64)


def test_no_filaments_nucleated(monkeypatch, filaments, centrosomes, basic_params):
    # Patch torch.poisson to always return 0
    monkeypatch.setattr(torch, "poisson", lambda x: torch.tensor(0.0))
    fils, max_id = nucleate_filaments_from_centrosomes(
        filaments, centrosomes, 0, basic_params
    )
    assert fils.shape[0] == 0
    assert max_id == 0


# Make assert function for testing radius and position
def assert_filament_positions(fils, centrosomes):
    cent_pos = centrosomes[fils[:, 10].long(), 1:4]
    cent_rads = centrosomes[fils[:, 10].long(), 4]
    new_fil_pos = fils[:, 1:4]
    assert torch.allclose(torch.norm(new_fil_pos - cent_pos, dim=1), cent_rads), (
        "Filament positions are not at the correct distance from centrosomes"
    )


def test_filaments_nucleated(monkeypatch, filaments, centrosomes, basic_params):
    # Patch torch.poisson to always return 2
    monkeypatch.setattr(torch, "poisson", lambda x: torch.tensor(2.0))
    fils, max_id = nucleate_filaments_from_centrosomes(
        filaments, centrosomes, 0, basic_params
    )
    assert fils.shape[0] == 2
    # Check IDs
    assert torch.all(fils[:, 0] == torch.tensor([1.0, 2.0]))
    # Check filament length
    assert torch.all(fils[:, 7] == basic_params["diam"])
    # Check filament state
    assert torch.all(fils[:, 8] == 1.0)
    # Check nucleator id
    assert torch.all(fils[:, 9] == -1.0)
    # Check centrosome id is valid
    assert torch.all((fils[:, 10] == 0.0) | (fils[:, 10] == 1.0))
    assert max_id == 2
    # Check minus ends are at centrosome radius away from centrosome position
    assert_filament_positions(fils, centrosomes)


def test_dimension_2(monkeypatch, filaments, centrosomes, basic_params):
    basic_params["dimension"] = 2
    monkeypatch.setattr(torch, "poisson", lambda x: torch.tensor(1.0))
    fils, max_id = nucleate_filaments_from_centrosomes(
        filaments, centrosomes, 5, basic_params
    )
    # Check direction z is zero
    assert fils[0, 6] == 0.0
    assert max_id == 6
    assert_filament_positions(fils, centrosomes)


def test_filaments_appended(monkeypatch, centrosomes, basic_params):
    monkeypatch.setattr(torch, "poisson", lambda x: torch.tensor(1.0))
    # Existing filaments
    filaments = torch.ones((2, 11), dtype=torch.float64)
    fils, max_id = nucleate_filaments_from_centrosomes(
        filaments, centrosomes, 2, basic_params
    )
    assert fils.shape[0] == 3
    assert torch.all(fils[:2] == 1.0)
    assert fils[2, 0] == 3.0  # New filament ID
    assert max_id == 3
    assert_filament_positions(fils[-1:], centrosomes)
