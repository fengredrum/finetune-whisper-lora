# Model Serving

```bash
jina hub new
docker build -t my_containerized_executor .
```

Create a `flow.yml` file:

```yml
jtype: Flow
with:
  port: 8080
  protocol: http
executors:
  - name: encoder
    uses: docker://<user-id>/EncoderPrivate
    replicas: 2
  - name: indexer
    uses: docker://<user-id>/IndexerPrivate
    shards: 2
```

```bash
jina export docker-compose flow.yml docker-compose.yml
docker compose up -d
```
