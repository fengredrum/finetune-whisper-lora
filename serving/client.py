from jina.clients import Client
from docarray import DocList, BaseDoc
from docarray.typing import NdArray
from typing import Optional


class MyDoc(BaseDoc):
    text: str = ''
    tensor: Optional[NdArray] = None


client = Client(host='http://172.31.0.4:8080')

docs = client.post(
    '/bar',
    inputs=DocList[MyDoc](
        [MyDoc(text=f'This is document indexed number {i}') for i in range(100)]),
    return_type=DocList[MyDoc],
    request_size=10,
)

print(docs.tensor)
