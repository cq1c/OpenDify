"""
OpenDify Lite 入口。实现已拆分到 opendify/ 包中，本文件仅负责启动 uvicorn。
"""

import os

from opendify import app
from opendify.config import SERVER_HOST, SERVER_PORT

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        access_log=False,
        server_header=False,
        date_header=False,
        loop="uvloop" if os.name != "nt" else "asyncio",
    )
