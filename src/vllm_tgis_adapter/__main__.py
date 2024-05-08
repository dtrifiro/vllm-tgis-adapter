import asyncio

from vllm.logger import init_logger

from .cli import run

logger = init_logger(__name__)


async def main() -> None:
    server = await run()

    try:
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        logger.info("Gracefully stopping gRPC server")
        await server.stop(30)  # TODO configurable grace
        await server.wait_for_termination()
        logger.info("gRPC server stopped")


if __name__ == "__main__":
    asyncio.run(main())
