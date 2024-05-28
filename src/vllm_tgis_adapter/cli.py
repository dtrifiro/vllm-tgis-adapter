from argparse import Namespace

import grpc
import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

from ._version import __version__
from .grpc import start_grpc_server
from .tgis_utils.args import add_tgis_args, postprocess_tgis_args


from grpc.aio import Server

logger = init_logger(__name__)


def parse_args() -> tuple[Namespace, AsyncEngineArgs]:
    parser = make_arg_parser()
    parser = add_tgis_args(parser)
    args = postprocess_tgis_args(parser.parse_args())
    return args, AsyncEngineArgs.from_cli_args(args)  # type: ignore[return-value]


async def run() -> Server:
    args, engine_args = parse_args()
    engine = AsyncLLMEngine.from_engine_args(
        engine_args,  # type: ignore[arg-type]
        usage_context=UsageContext.OPENAI_API_SERVER,
    )

    logger.info("vLLM API grpc server version %s", __version__)
    logger.info("vLLM version %s", vllm.__version__)
    logger.info("args: %s", args)

    return await start_grpc_server(
        engine, args, disable_log_stats=engine_args.disable_log_stats
    )
