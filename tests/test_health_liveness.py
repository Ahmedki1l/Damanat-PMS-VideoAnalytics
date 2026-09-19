"""HTTP health reports API liveness, independent of engine/DB/camera checks."""
import pytest
from fastapi.testclient import TestClient

from src.api import create_app


@pytest.mark.parametrize('engine_status', ['degraded', 'unhealthy'])
def test_engine_failure_cannot_change_health_response(engine_status, tmp_path):
    app = create_app(
        get_engine_status=lambda: {
            'status': engine_status,
            'db_ok': False,
            'model_loaded': False,
            'camera_streams_delivering': 0,
        },
        snapshot_base_dir=str(tmp_path),
    )
    response = TestClient(app).get('/api/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
    assert 'db_ok' not in response.json()


def test_health_never_invokes_engine_dependency_checks(tmp_path):
    def unavailable_dependencies():
        raise AssertionError('Engine/database/camera checks must not run for liveness')

    app = create_app(
        get_engine_status=unavailable_dependencies,
        snapshot_base_dir=str(tmp_path),
    )
    response = TestClient(app).get('/api/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
