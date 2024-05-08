import sys

import grpc
import pytest
import pytest_asyncio

from vllm_tgis_adapter.cli import run
from vllm_tgis_adapter.grpc.pb import generation_pb2_grpc


@pytest.fixture(scope="session")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch

    m = MonkeyPatch()
    yield m
    m.undo()


@pytest.fixture(scope="session")
def grpc_server_details():
    host = "localhost"
    grpc_port = 8033
    server_address = f"{host}:{grpc_port}"

    return host, grpc_port, server_address


@pytest_asyncio.fixture(scope="session")
async def grpc_server(grpc_server_details, monkeypatch_session):
    """Spin up vllm_tgis_adapter grpc server."""
    host, grpc_port, server_address = grpc_server_details

    argv = [
        "__main__.py",
        f"--host={host}",
        f"--grpc-port={grpc_port}",
    ]
    monkeypatch_session.setattr(sys, "argv", argv)

    server = await run()

    channel = grpc.insecure_channel(server_address)
    try:
        grpc.channel_ready_future(channel).result(timeout=60)
    except grpc.FutureTimeoutError as exc:
        raise RuntimeError("Failed to start grpc server") from exc

    yield server

    await server.stop(30)
    await server.wait_for_termination()


@pytest.fixture()
def grpc_client(grpc_server, grpc_server_details):
    """Return the GenerationService stub."""
    _, _, server_address = grpc_server_details
    channel = grpc.insecure_channel(server_address)
    return generation_pb2_grpc.GenerationServiceStub(channel)


@pytest.mark.asyncio()
async def test_modelinfo(grpc_server, grpc_client):
    modelinfo_request = generation_pb2_grpc.vllm__tgis__adapter_dot_grpc_dot_pb_dot_generation__pb2.ModelInfoRequest( # noqa: E501
        model_id="facebook/opt-125m"
    )
    response = grpc_client.ModelInfo(modelinfo_request)
    assert response is not None
