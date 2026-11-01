from pinecone import Pinecone, ServerlessSpec

METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"


def ensure_index(pc: Pinecone, name: str, dimension: int) -> None:
    if pc.has_index(name):
        info = pc.describe_index(name)
        if info.dimension != dimension:
            raise ValueError(f"Index '{name}' has dimension {info.dimension}, expected {dimension}")
        if info.metric != METRIC:
            raise ValueError(f"Index '{name}' uses metric {info.metric}, expected {METRIC}")
    else:
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=METRIC,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
