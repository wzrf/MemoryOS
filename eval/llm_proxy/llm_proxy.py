import itertools
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
import uvicorn

app = FastAPI()

# 1. 替换为你实际的 4 个后端服务地址
BACKENDS = [
    "http://127.0.0.1:30005",
    "http://127.0.0.1:30006",
    "http://127.0.0.1:30007",
    "http://127.0.0.1:30008",
    "http://127.0.0.1:30009",
    "http://127.0.0.1:30010",
    "http://127.0.0.1:30011",
]

# 2. 轮询调度器 (Round-Robin)
backend_cycle = itertools.cycle(BACKENDS)

# 3. 创建异步 HTTP 客户端（禁用超时限制，防止大模型生成长文本超时）
client = httpx.AsyncClient(timeout=None)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
async def proxy(request: Request, path: str):
    # 选择下一个后端服务器
    target_backend = next(backend_cycle)
    url = f"{target_backend}/{path}"

    # 构建转发请求
    headers = dict(request.headers)
    headers.pop("host", None)  # 移除原本的 host 头，避免后端校验报错

    req = client.build_request(
        method=request.method,
        url=url,
        headers=headers,
        params=request.query_params,
        content=await request.body()
    )

    # 发送请求并开启流式响应
    response = await client.send(req, stream=True)

    # 过滤掉响应头中的 content-length，防止与流式传输冲突
    response_headers = dict(response.headers)
    response_headers.pop("content-length", None)

    # 透传后端响应（支持 SSE / 打字机输出）
    return StreamingResponse(
        response.aiter_raw(),
        status_code=response.status_code,
        headers=response_headers,
        # 【修复点】使用 BackgroundTask 正确包装实例的 aclose 方法
        background=BackgroundTask(response.aclose)
    )


if __name__ == "__main__":
    # 监听 127.0.0.1:40003
    uvicorn.run(app, host="127.0.0.1", port=30004)