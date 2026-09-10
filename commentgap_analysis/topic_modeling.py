"""Core topic-model fitting, persistence, and document/distribution preparation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TOPIC_COLUMN_RE = re.compile(r"^topic_(\d+)$")


TOPIC_MODEL_VERSION = 3


def _current_code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _frame_fingerprint(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    selected = list(columns or frame.columns)
    working = frame.loc[:, selected].copy()
    digest = hashlib.sha256()
    digest.update(json.dumps(selected, sort_keys=True, default=str).encode())
    digest.update(pd.util.hash_pandas_object(working, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _model_component_manifest(component: object) -> dict[str, object]:
    result = {
        "class": f"{type(component).__module__}.{type(component).__name__}",
    }
    if hasattr(component, "get_params"):
        try:
            result["params"] = {
                str(key): repr(value)
                for key, value in component.get_params(deep=False).items()
            }
        except Exception:
            result["params"] = "unavailable"
    return result


@dataclass(frozen=True)
class TopicModelConfig:
    """Configuration for the common article/comment topic model."""

    model_type: str = "bertopic"
    n_topics: int | None = None
    nr_topics: str | int | None = "auto"
    min_df: int | float = 5
    max_df: int | float = 0.95
    max_features: int | None = 100_000
    ngram_range: tuple[int, int] = (1, 2)
    random_state: int = 2025
    max_iter: int = 400
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_metric: str = "cosine"
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int | None = None
    calculate_probabilities: bool = False
    representation_stopwords: bool = False
    reduce_frequent_words: bool = False
    outlier_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.outlier_threshold is not None and not 0 < self.outlier_threshold <= 1:
            raise ValueError("outlier_threshold must be in (0, 1] or None")
        if self.model_type not in {"bertopic", "nmf", "lda"}:
            raise ValueError("model_type must be 'bertopic', 'nmf', or 'lda'")
        if self.model_type == "bertopic" and self.n_topics is not None:
            raise ValueError("BERTopic determines the topic count; use nr_topics instead")
        if self.model_type != "bertopic" and (self.n_topics is None or self.n_topics < 2):
            raise ValueError("n_topics must be at least 2 for NMF/LDA")
        if self.model_type == "bertopic" and isinstance(self.nr_topics, int) and self.nr_topics < 2:
            raise ValueError("nr_topics must be at least 2 when specified")
        if self.min_df <= 0 or self.max_df <= 0:
            raise ValueError("min_df and max_df must be positive")
        if self.max_features is not None and self.n_topics is not None and self.max_features < self.n_topics:
            raise ValueError("max_features must be at least n_topics")
        if self.umap_n_neighbors < 2 or self.umap_n_components < 2:
            raise ValueError("UMAP neighbors and components must be at least 2")
        if self.hdbscan_min_cluster_size < 2:
            raise ValueError("hdbscan_min_cluster_size must be at least 2")
        if self.hdbscan_min_samples is not None and self.hdbscan_min_samples < 1:
            raise ValueError("hdbscan_min_samples must be at least 1 or None")
        if not isinstance(self.calculate_probabilities, bool):
            raise ValueError("calculate_probabilities must be a boolean")
        if len(self.ngram_range) != 2 or self.ngram_range[0] < 1:
            raise ValueError("ngram_range must be a valid (min_n, max_n) pair")


def _topic_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in frame.columns:
        match = TOPIC_COLUMN_RE.match(str(column))
        if match:
            columns.append((int(match.group(1)), str(column)))
    columns.sort()
    result = [column for _, column in columns]
    if len(result) < 2:
        raise ValueError("At least two columns named topic_000, topic_001, ... are required")
    expected = [f"topic_{index:03d}" for index in range(len(result))]
    if result != expected:
        raise ValueError(f"Topic columns must be contiguous and zero-padded: {expected}")
    return result


def _normalise_distribution(values: np.ndarray, *, axis: int = -1) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
        squeeze = True
    elif array.ndim == 2:
        squeeze = False
    else:
        raise ValueError("Topic distributions must be one- or two-dimensional")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Topic distributions must be finite and non-negative")
    totals = array.sum(axis=axis, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Topic distributions must have positive row sums")
    result = array / totals
    return result[0] if squeeze else result


def _distribution_metric_inputs(first: Sequence[float], second: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    left = _normalise_distribution(np.asarray(first, dtype=float))
    right = _normalise_distribution(np.asarray(second, dtype=float))
    if left.shape != right.shape:
        raise ValueError("Topic distributions must have identical shapes")
    return left, right


def prepare_documents(
    articles: pd.DataFrame,
    comments: pd.DataFrame,
    *,
    article_text_columns: Sequence[str] = ("title", "subtitle", "body"),
    comment_text_column: str = "effective_text",
) -> pd.DataFrame:
    """Build a common document table for article and comment text."""

    for frame, label in ((articles, "articles"), (comments, "comments")):
        for column in ("story_id",):
            if column not in frame.columns:
                raise ValueError(f"{label} is missing required column {column!r}")
    missing_article = [column for column in article_text_columns if column not in articles]
    if missing_article:
        raise ValueError(f"articles is missing text columns: {missing_article}")
    comment_text_column = comment_text_column if comment_text_column in comments else "text"
    if comment_text_column not in comments:
        raise ValueError("comments must contain effective_text or text")

    article_text = articles.loc[:, ["story_id", *article_text_columns]].copy()
    article_values = article_text.loc[:, article_text_columns].fillna("").to_numpy(dtype=object)
    article_text["text"] = [
        "\n".join(str(value) for value in row).strip() for row in article_values
    ]
    article_story_ids = article_text["story_id"].astype(str).to_numpy(dtype=object)
    article_docs = pd.DataFrame({
        "doc_id": [f"article:{story_id}" for story_id in article_story_ids],
        "story_id": article_story_ids,
        "doc_type": "article",
        "text": article_text["text"],
    })

    comment_docs = comments.loc[:, ["story_id", comment_text_column]].copy()
    if "comment_id" not in comments:
        raise ValueError("comments is missing required column 'comment_id'")
    comment_docs["comment_id"] = comments["comment_id"].astype(str).to_numpy()
    comment_docs["text"] = comment_docs[comment_text_column].fillna("").astype(str).str.strip()
    comment_docs = comment_docs.loc[comment_docs["text"].ne("")].copy()
    comment_docs["story_id"] = comment_docs["story_id"].astype(str)
    comment_story_ids = comment_docs["story_id"].to_numpy(dtype=object)
    comment_ids = comment_docs["comment_id"].to_numpy(dtype=object)
    comment_docs["doc_id"] = [
        f"comment:{story_id}:{comment_id}"
        for story_id, comment_id in zip(comment_story_ids, comment_ids)
    ]
    comment_docs["doc_type"] = "comment"
    comment_docs = comment_docs.loc[:, ["doc_id", "story_id", "comment_id", "doc_type", "text"]]

    result = pd.concat([article_docs, comment_docs], ignore_index=True)
    result = result.loc[result["text"].ne("")].reset_index(drop=True)
    if result["doc_id"].duplicated().any():
        raise ValueError("Document identifiers must be unique")
    return result


def prepare_article_passages(articles: pd.DataFrame) -> pd.DataFrame:
    """Build one topic-model document per stored article passage.

    The passage text and identifiers are generated by the same routine used
    when the BGE-M3 article-passage embeddings were created.
    """

    required = {"story_id", "title", "subtitle", "body"}
    missing = sorted(required - set(articles.columns))
    if missing:
        raise ValueError(f"articles is missing required columns: {missing}")
    from .embeddings import _passage_records

    rows = []
    for _, article in articles.iterrows():
        for passage in _passage_records(article):
            rows.append(
                {
                    "doc_id": f"article_passage:{passage['passage_id']}",
                    "story_id": passage["story_id"],
                    "doc_type": "article",
                    "passage_id": passage["passage_id"],
                    "passage_kind": passage["passage_kind"],
                    "passage_index": passage["passage_index"],
                    "text": passage["text"],
                }
            )
    return pd.DataFrame(
        rows,
        columns=["doc_id", "story_id", "doc_type", "passage_id", "passage_kind", "passage_index", "text"],
    )


def load_precomputed_embeddings(
    documents: pd.DataFrame,
    embedding_store: Path | str,
) -> np.ndarray:
    """Load BGE-M3 vectors aligned to the prepared document order."""

    required = {"story_id", "doc_type", "doc_id"}
    missing = sorted(required - set(documents.columns))
    if missing:
        raise ValueError(f"documents is missing required columns: {missing}")
    if documents.empty:
        raise ValueError("At least one document is required")
    embedding_store = Path(embedding_store)
    manifest_path = embedding_store / "embedding_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing embedding manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    dimension = int(manifest.get("embedding_dimension", 0))
    if dimension < 1:
        raise ValueError(f"Invalid embedding dimension in {manifest_path}")
    comment_paths = {path.stem: path for path in (embedding_store / "comments").glob("**/*.parquet")}
    passage_paths = {path.stem: path for path in (embedding_store / "article_passages").glob("**/*.parquet")}
    if not comment_paths and not passage_paths:
        raise FileNotFoundError(f"No embedding shards found below {embedding_store}")

    result = np.empty((len(documents), dimension), dtype=np.float32)
    comment_cache: dict[str, dict[str, np.ndarray]] = {}
    passage_cache: dict[str, np.ndarray] = {}

    def normalize(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if vector.shape != (dimension,) or not np.isfinite(vector).all() or norm <= np.finfo(np.float32).eps:
            raise ValueError("Embedding vectors must be finite, non-zero, and match the store dimension")
        return vector / norm

    for row_index, document in documents.reset_index(drop=True).iterrows():
        story_id = str(document["story_id"])
        if document["doc_type"] == "comment":
            if "comment_id" not in documents:
                raise ValueError("Comment documents must contain comment_id")
            if story_id not in comment_cache:
                path = comment_paths.get(story_id)
                if path is None:
                    raise ValueError(f"No comment embedding shard found for story {story_id}")
                table = pq.read_table(path, columns=["comment_id", "embedding"])
                comment_cache[story_id] = {
                    str(comment_id): np.asarray(vector, dtype=np.float32)
                    for comment_id, vector in zip(table["comment_id"].to_pylist(), table["embedding"].to_pylist())
                }
            comment_id = str(document["comment_id"])
            if comment_id not in comment_cache[story_id]:
                raise ValueError(f"No embedding found for comment {story_id}/{comment_id}")
            result[row_index] = normalize(comment_cache[story_id][comment_id])
        elif document["doc_type"] == "article":
            passage_id = document.get("passage_id")
            if pd.notna(passage_id):
                if story_id not in passage_cache:
                    path = passage_paths.get(story_id)
                    if path is None:
                        raise ValueError(f"No article-passage embedding shard found for story {story_id}")
                    table = pq.read_table(path, columns=["passage_id", "embedding"])
                    passage_cache[story_id] = {
                        str(key): np.asarray(vector, dtype=np.float32)
                        for key, vector in zip(table["passage_id"].to_pylist(), table["embedding"].to_pylist())
                    }
                vector = passage_cache[story_id].get(str(passage_id))
                if vector is None:
                    raise ValueError(f"No embedding found for article passage {passage_id}")
                result[row_index] = normalize(vector)
            else:
                if story_id not in passage_cache:
                    path = passage_paths.get(story_id)
                    if path is None:
                        raise ValueError(f"No article-passage embedding shard found for story {story_id}")
                    table = pq.read_table(path, columns=["embedding"])
                    vectors = np.asarray(table["embedding"].to_pylist(), dtype=np.float32)
                    if vectors.ndim != 2 or not len(vectors):
                        raise ValueError(f"Article embedding shard is empty or malformed for story {story_id}")
                    passage_cache[story_id] = normalize(vectors.mean(axis=0))
                result[row_index] = passage_cache[story_id]
        else:
            raise ValueError(f"Unsupported document type: {document['doc_type']!r}")
    return result


def topic_stopwords():
    """German/English stopwords normalised like the representation vectorizer."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, strip_accents_unicode
    german = set('aber alle allem allen aller alles als also am an andere anderem anderen anderer anderes auch auf aus bei beim bis bist da dabei dadurch dafür dagegen daher dahin damit dann darauf daraus darin darüber darum davon dazu dein deine dem den denn der des das die dass diese diesem diesen dieser dieses doch dort du durch ein eine einem einen einer eines er es etwas für gegen hat habe haben hier hinter ich ihm ihn ihre ihr im in ist ja jede jedem jeden jeder jedes jetzt kann können man mehr mein meine mit muss müssen nach nein nicht noch nur ob oder ohne schon sehr sein seine selbst sich sie sind so über um und uns unter vom von vor war waren warum was weil weiter welche welchem welchen welcher welches wenn wer wen wem werde werden wessen wie wieder wieso weshalb will wir wo woran worauf woraus worin womit wovon wozu zu zum zur'.split())
    return sorted({strip_accents_unicode(word.lower()) for word in german | set(ENGLISH_STOP_WORDS)})


@dataclass
class TopicModelBundle:
    """Fitted vectorizer/model pair with a stable topic-column contract."""

    config: TopicModelConfig
    vectorizer: object
    model: object
    topic_terms: tuple[tuple[str, ...], ...]
    topic_ids: tuple[int, ...] = ()

    @property
    def topic_columns(self) -> list[str]:
        return [f"topic_{index:03d}" for index in range(len(self.topic_terms))]

    def reduce_outlier_labels(self, texts, topics, embeddings):
        """Assign only outliers to fixed topic embeddings above a cosine threshold."""
        labels = np.asarray(topics, dtype=int)
        if self.config.outlier_threshold is None or not np.any(labels == -1) or not self.topic_ids:
            return labels.copy()
        if embeddings is None:
            raise ValueError("Embedding outlier reduction requires precomputed embeddings")
        if not self.model._outliers:
            # BERTopic rejects reduce_outliers when the fit had no noise topic,
            # even if prediction rejects new documents. Use the identical cosine rule.
            from sklearn.metrics.pairwise import cosine_similarity
            mask = labels == -1
            similarity = cosine_similarity(np.asarray(embeddings)[mask], self.model.topic_embeddings_)
            best = similarity.argmax(axis=1)
            result = labels.copy()
            result[mask] = np.where(similarity.max(axis=1) >= self.config.outlier_threshold,
                                    np.asarray(self.topic_ids)[best], -1)
            return result
        return np.asarray(self.model.reduce_outliers(
            texts, labels.tolist(), strategy="embeddings", embeddings=embeddings,
            threshold=self.config.outlier_threshold,
        ), dtype=int)

    def transform(self, documents: pd.DataFrame, embeddings: np.ndarray | None = None) -> pd.DataFrame:
        if not {"doc_id", "story_id", "doc_type", "text"}.issubset(documents.columns):
            raise ValueError("documents must contain doc_id, story_id, doc_type, and text")
        texts = documents["text"].fillna("").astype(str).tolist()
        if self.config.model_type == "bertopic":
            embeddings = None if embeddings is None else np.asarray(embeddings, dtype=np.float32)
            if embeddings is not None and (embeddings.ndim != 2 or embeddings.shape[0] != len(texts)):
                raise ValueError("Precomputed embeddings must have one row per document")
            topics, memberships = self.model.transform(texts, embeddings=embeddings)
        else:
            matrix = self.vectorizer.transform(texts)
            memberships = self.model.transform(matrix)
            topics = np.asarray(memberships).argmax(axis=1)
        raw_topics = np.asarray(topics, dtype=int).copy()
        if self.config.model_type == "bertopic":
            topics = self.reduce_outlier_labels(texts, raw_topics, embeddings)
        topics = np.asarray(topics, dtype=int)
        memberships = np.asarray(memberships, dtype=float) if memberships is not None else np.zeros(
            (len(texts), len(self.topic_columns)), dtype=float
        )
        if memberships.ndim == 1:
            # BERTopic returns a 1-D confidence vector when probability
            # calculation is disabled; retain the predicted labels and use
            # one-hot memberships for valid topics.
            if self.config.calculate_probabilities:
                raise ValueError(
                    "BERTopic did not return a topic-by-document probability matrix "
                    "despite calculate_probabilities=True"
                )
            memberships = np.zeros((len(texts), len(self.topic_columns)), dtype=float)
        if memberships.ndim != 2 or memberships.shape[1] != len(self.topic_columns):
            raise ValueError("Topic transform returned an unexpected membership matrix shape")
        memberships[topics != raw_topics, :] = 0.0
        lookup = {topic_id: index for index, topic_id in enumerate(self.topic_ids)}
        zero_rows = memberships.sum(axis=1) <= 0
        for row_index in np.flatnonzero(zero_rows):
            topic_index = lookup.get(int(topics[row_index]))
            if topic_index is not None:
                memberships[row_index, topic_index] = 1.0
        valid_topic = topics != -1
        # Outliers remain explicit -1 assignments and carry zero mass in the
        # valid-topic distribution. They are excluded by downstream aggregators.
        memberships[~valid_topic, :] = 0.0
        valid_rows = memberships.sum(axis=1) > 0
        if valid_rows.any():
            memberships[valid_rows] = _normalise_distribution(memberships[valid_rows])
        metadata_columns = (
            "doc_id",
            "story_id",
            "comment_id",
            "doc_type",
            "passage_id",
            "passage_kind",
            "passage_index",
        )
        result = documents.loc[:, [column for column in metadata_columns if column in documents]].reset_index(drop=True).copy()
        topic_frame = pd.DataFrame(
            memberships.astype(np.float32, copy=False),
            columns=self.topic_columns,
            index=result.index,
        )
        label_frame = pd.DataFrame(
            {"topic_assignment": topics, "valid_topic": valid_topic,
             "raw_topic_assignment": raw_topics, "outlier_reassigned": topics != raw_topics},
            index=result.index,
        )
        return pd.concat([result, topic_frame, label_frame], axis=1, copy=False)


def fit_topic_model(
    documents: pd.DataFrame,
    config: TopicModelConfig = TopicModelConfig(),
    embeddings: np.ndarray | None = None,
) -> TopicModelBundle:
    """Fit automated BERTopic by default, or explicit NMF/LDA alternatives."""

    if documents.empty or documents["text"].fillna("").astype(str).str.strip().eq("").all():
        raise ValueError("At least one non-empty document is required")
    if config.model_type == "bertopic":
        try:
            from bertopic import BERTopic
            from bertopic.vectorizers import ClassTfidfTransformer
            from hdbscan import HDBSCAN
            from sklearn.feature_extraction.text import CountVectorizer
            from umap import UMAP
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "BERTopic, hdbscan, umap-learn, and scikit-learn are required; "
                "install requirements-analysis.txt"
            ) from exc
        vectorizer_model = CountVectorizer(
            stop_words=topic_stopwords() if config.representation_stopwords else None,
            min_df=config.min_df,
            max_df=config.max_df,
            max_features=config.max_features,
            ngram_range=config.ngram_range,
            strip_accents="unicode",
        )
        umap_model = UMAP(
            n_neighbors=config.umap_n_neighbors,
            n_components=config.umap_n_components,
            metric=config.umap_metric,
            random_state=config.random_state,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=config.hdbscan_min_cluster_size,
            min_samples=config.hdbscan_min_samples,
            prediction_data=True,
        )
        model = BERTopic(
            nr_topics=config.nr_topics,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=config.reduce_frequent_words),
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            calculate_probabilities=config.calculate_probabilities,
            verbose=False,
        )
        texts = documents["text"].fillna("").astype(str).tolist()
        embeddings = None if embeddings is None else np.asarray(embeddings, dtype=np.float32)
        if embeddings is not None and (embeddings.ndim != 2 or embeddings.shape[0] != len(texts)):
            raise ValueError("Precomputed embeddings must have one row per document")
        topics, probabilities = model.fit_transform(texts, embeddings=embeddings)
        topic_ids = tuple(sorted(topic_id for topic_id in model.get_topics() if topic_id != -1))
        probabilities = None if probabilities is None else np.asarray(probabilities, dtype=float)
        if probabilities is None or probabilities.ndim != 2 or len(topic_ids) != probabilities.shape[1]:
            if config.calculate_probabilities:
                raise ValueError(
                    "BERTopic did not return a topic-by-document probability matrix "
                    "despite calculate_probabilities=True"
                )
            probabilities = np.zeros((len(topics), len(topic_ids)), dtype=float)
            lookup = {topic_id: index for index, topic_id in enumerate(topic_ids)}
            for row_index, topic_id in enumerate(topics):
                if topic_id in lookup:
                    probabilities[row_index, lookup[topic_id]] = 1.0
        terms_by_topic = []
        for topic_id in topic_ids:
            terms_by_topic.append(tuple(term for term, _ in (model.get_topic(topic_id) or [])[:15]))
        return TopicModelBundle(
            config=config,
            vectorizer=None,
            model=model,
            topic_terms=tuple(terms_by_topic),
            topic_ids=topic_ids,
        )

    try:
        from sklearn.decomposition import LatentDirichletAllocation, NMF
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    except ImportError as exc:  # pragma: no cover - exercised only without analysis extras
        raise ImportError("Topic modeling requires scikit-learn; install requirements-analysis.txt") from exc

    if config.model_type == "nmf":
        vectorizer = TfidfVectorizer(
            min_df=config.min_df,
            max_df=config.max_df,
            max_features=config.max_features,
            ngram_range=config.ngram_range,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(documents["text"].fillna("").astype(str))
        model = NMF(n_components=config.n_topics, init="nndsvda", random_state=config.random_state, max_iter=config.max_iter)
    else:
        vectorizer = CountVectorizer(
            min_df=config.min_df,
            max_df=config.max_df,
            max_features=config.max_features,
            ngram_range=config.ngram_range,
            strip_accents="unicode",
        )
        matrix = vectorizer.fit_transform(documents["text"].fillna("").astype(str))
        model = LatentDirichletAllocation(n_components=config.n_topics, random_state=config.random_state, max_iter=config.max_iter, learning_method="batch")
    if matrix.shape[1] < config.n_topics:
        raise ValueError("The vocabulary is smaller than n_topics; lower min_df or n_topics")
    model.fit(matrix)
    terms = np.asarray(vectorizer.get_feature_names_out())
    components = np.asarray(model.components_)
    topic_terms = tuple(tuple(terms[np.argsort(row)[::-1][:15]].tolist()) for row in components)
    return TopicModelBundle(config=config, vectorizer=vectorizer, model=model, topic_terms=topic_terms)


def save_topic_model(bundle: TopicModelBundle, output_root: Path | str) -> dict[str, Path]:
    """Persist a fitted model and a human-readable manifest."""

    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Saving a topic model requires joblib from scikit-learn") from exc
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = output_root / "topic_model.joblib"
    manifest_path = output_root / "topic_model_manifest.json"
    joblib.dump(bundle, model_path)
    model_components = {}
    for name in ("umap_model", "hdbscan_model", "vectorizer_model"):
        component = getattr(bundle.model, name, None)
        if component is not None:
            model_components[name] = _model_component_manifest(component)
    manifest = {
        "version": TOPIC_MODEL_VERSION,
        "code_revision": _current_code_revision(),
        "config": asdict(bundle.config),
        "topic_columns": bundle.topic_columns,
        "topic_ids": list(bundle.topic_ids),
        "topic_terms": [list(terms) for terms in bundle.topic_terms],
        "model_components": model_components,
        "model_path": str(model_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"model": model_path, "manifest": manifest_path}


def load_topic_model(path: Path | str) -> TopicModelBundle:
    """Load a model saved by :func:`save_topic_model`."""

    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Loading a topic model requires joblib from scikit-learn") from exc
    bundle = joblib.load(path)
    if not isinstance(bundle, TopicModelBundle):
        raise ValueError(f"Unsupported topic model artifact: {path}")
    return bundle


def topic_model_quality_table(bundle: TopicModelBundle) -> pd.DataFrame:
    """Return compact term and vocabulary diagnostics for a fitted model."""

    if bundle.config.model_type == "bertopic":
        vocabulary = np.asarray(bundle.model.vectorizer_model.get_feature_names_out())
        component_matrix = bundle.model.c_tf_idf_
        if hasattr(component_matrix, "toarray"):
            component_matrix = component_matrix.toarray()
        component_matrix = np.asarray(component_matrix, dtype=float)
        component_matrix = component_matrix[-len(bundle.topic_terms):]
    else:
        vocabulary = np.asarray(bundle.vectorizer.get_feature_names_out())
        component_matrix = np.asarray(bundle.model.components_, dtype=float)
    top_terms = [terms[:10] for terms in bundle.topic_terms]
    all_top_terms = [term for terms in top_terms for term in terms]
    return pd.DataFrame({
        "model_type": bundle.config.model_type,
        "n_topics": len(bundle.topic_terms),
        "vocabulary_size": len(vocabulary),
        "topic": range(len(bundle.topic_terms)),
        "top_terms": [" | ".join(terms) for terms in top_terms],
        "topic_term_mass": component_matrix.sum(axis=1),
        "top_term_uniqueness": [len(set(terms)) / len(all_top_terms) for terms in top_terms],
    })


def aggregate_topic_distributions(
    memberships: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("story_id", "doc_type"),
    weight_column: str | None = None,
    include_invalid: bool = False,
) -> pd.DataFrame:
    """Aggregate valid document-topic memberships into group distributions.

    BERTopic outliers (assignment ``-1``) are retained in the membership
    artifact but excluded from distributions by default. Groups containing no
    valid documents are omitted.
    """

    topic_columns = _topic_columns(memberships)
    if not include_invalid and "valid_topic" in memberships.columns:
        memberships = memberships.loc[memberships["valid_topic"].astype(bool)].copy()
    missing = [column for column in group_columns if column not in memberships]
    if missing:
        raise ValueError(f"memberships is missing group columns: {missing}")
    if weight_column is not None and weight_column not in memberships:
        raise ValueError(f"memberships is missing weight column {weight_column!r}")
    rows = []
    for keys, group in memberships.groupby(list(group_columns), sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group[topic_columns].to_numpy(dtype=float)
        if weight_column is None:
            distribution = values.mean(axis=0)
        else:
            weights = group[weight_column].to_numpy(dtype=float)
            if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
                raise ValueError("Aggregation weights must be finite, non-negative, and non-zero")
            distribution = np.average(values, axis=0, weights=weights)
        row = dict(zip(group_columns, keys))
        row.update(dict(zip(topic_columns, _normalise_distribution(distribution))))
        row["n_documents"] = len(group)
        rows.append(row)
    return pd.DataFrame(rows, columns=[*group_columns, *topic_columns, "n_documents"])


def article_topic_coverage(memberships: pd.DataFrame) -> pd.DataFrame:
    """Return per-story article-section coverage and valid-topic counts.

    Article passage rows are the sections. Comment rows are retained only for
    the separate document-level audit counts. Outliers retain
    topic_assignment == -1 and are excluded from article distributions.
    """

    required = {"story_id", "topic_assignment", "valid_topic"}
    missing = sorted(required - set(memberships.columns))
    if missing:
        raise ValueError(f"memberships is missing required coverage columns: {missing}")
    rows = []
    for story_id, group in memberships.groupby("story_id", sort=True, dropna=False):
        article_rows = group.loc[group["doc_type"].eq("article")] if "doc_type" in group else group.iloc[0:0]
        article_valid = article_rows["valid_topic"].astype(bool)
        document_valid = group["valid_topic"].astype(bool)
        rows.append(
            {
                "story_id": story_id,
                "n_sections": int(len(article_rows)),
                "n_valid_sections": int(article_valid.sum()),
                "n_outlier_sections": int((~article_valid).sum()),
                "fraction_sections_valid_topic": float(article_valid.mean()) if len(article_rows) else np.nan,
                "n_unique_topics": int(article_rows.loc[article_valid, "topic_assignment"].nunique()),
                "has_valid_topic": bool(article_valid.any()),
                "n_documents": int(len(group)),
                "n_valid_documents": int(document_valid.sum()),
                "n_comment_documents": int(group["doc_type"].eq("comment").sum()) if "doc_type" in group else 0,
                "n_valid_comment_documents": int(
                    (document_valid & group["doc_type"].eq("comment")).sum()
                ) if "doc_type" in group else 0,
                "article_topic_assignment": -1,
                "article_has_valid_topic": bool(article_valid.any()),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "story_id",
            "n_sections",
            "n_valid_sections",
            "n_outlier_sections",
            "fraction_sections_valid_topic",
            "n_unique_topics",
            "has_valid_topic",
            "n_documents",
            "n_valid_documents",
            "n_comment_documents",
            "n_valid_comment_documents",
            "article_topic_assignment",
            "article_has_valid_topic",
        ],
    )

__all__ = [
    "TOPIC_COLUMN_RE",
    "TOPIC_MODEL_VERSION",
    "TopicModelConfig",
    "prepare_documents",
    "prepare_article_passages",
    "load_precomputed_embeddings",
    "topic_stopwords",
    "TopicModelBundle",
    "fit_topic_model",
    "save_topic_model",
    "load_topic_model",
    "topic_model_quality_table",
    "aggregate_topic_distributions",
    "article_topic_coverage",
]
