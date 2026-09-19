import itertools
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
import uvicorn

app = FastAPI()

BACKENDS = [
    "http://127.0.0.1:20001",
    "http://127.0.0.1:20002",
    "http://127.0.0.1:20003",
    "http://127.0.0.1:20004",
    "http://127.0.0.1:20005",
    "http://127.0.0.1:20006",
]

print(f"BACKENDS: {BACKENDS}")

backend_cycle = itertools.cycle(BACKENDS)

client = httpx.AsyncClient(
    timeout=None,
    limits=httpx.Limits(
        max_connections=2000,
        max_keepalive_connections=2000,
    ),
)


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"]
)
async def proxy(request: Request, path: str):
    target_backend = next(backend_cycle)
    url = f"{target_backend}/{path}"

    headers = dict(request.headers)
    headers.pop("host", None)

    req = client.build_request(
        method=request.method,
        url=url,
        headers=headers,
        params=request.query_params,
        content=await request.body(),
    )

    response = await client.send(req, stream=True)

    response_headers = dict(response.headers)
    response_headers.pop("content-length", None)

    return StreamingResponse(
        response.aiter_raw(),
        status_code=response.status_code,
        headers=response_headers,
        background=BackgroundTask(response.aclose),
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=20000,
        workers=1,
    )