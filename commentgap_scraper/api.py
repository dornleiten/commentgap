from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .http import HttpClient, HttpFailure


GRAPHQL_ENDPOINT = "https://api-gateway.prod.cloud.ds.at/forum-serve-graphql/v1/"


def _posting_selection(depth: int) -> str:
    fields = """
      id
      lifecycleStatus
      flags
      rootPostingId
      text
      title
      author { id name followerCount }
      reactions { aggregated { name value statistic } }
      history { created }
      legacy { communityName communityIdentityId postingId }
    """
    selection = fields
    for _ in range(depth):
        selection = f"{fields} replies {{ {selection} }}"
    return selection


def forum_info_query(reply_depth: int) -> str:
    posting = _posting_selection(reply_depth)
    return f"""
      query GetForumInfo($contextUri: String!) {{
        getForumByContextUri(contextUri: $contextUri) {{
          id
          flags
          totalPostingCount
          metadata {{ key value }}
          stickyPostings {{ {posting} }}
        }}
      }}
    """


def threads_query(reply_depth: int) -> str:
    posting = _posting_selection(reply_depth)
    return f"""
      query ThreadsByForumQuery(
        $id: String!,
        $first: RootPostingsToRequest,
        $nextCursor: String,
        $sortOrder: PostingSortOrder
      ) {{
        getForumRootPostingsV2(getForumRootPostingsParamsV2: {{
          forumId: $id,
          after: $nextCursor,
          first: $first,
          sortOrder: $sortOrder
        }}) {{
          pageInfo {{ nextCursor previousCursor hasNextPage hasPreviousPage }}
          edges {{ cursor node {{ {posting} }} }}
        }}
      }}
    """


@dataclass(slots=True)
class ForumApi:
    http: HttpClient
    reply_depth: int = 32
    endpoint: str = GRAPHQL_ENDPOINT
    _forum_info_cache: dict[str, dict[str, Any] | None] = field(
        default_factory=dict, init=False, repr=False
    )

    def get_forum_info(
        self, context_uri: str, *, refresh: bool = False
    ) -> dict[str, Any] | None:
        if not refresh and context_uri in self._forum_info_cache:
            return self._forum_info_cache[context_uri]
        payload = {
            "operationName": "GetForumInfo",
            "variables": {"contextUri": context_uri},
            "query": forum_info_query(self.reply_depth),
        }
        response = self.http.post_json(self.endpoint, payload)
        result = (response.get("data") or {}).get("getForumByContextUri")
        self._forum_info_cache[context_uri] = result
        return result

    def get_threads_page(self, forum_id: str, cursor: str | None = None) -> dict[str, Any]:
        payload = {
            "operationName": "ThreadsByForumQuery",
            "variables": {
                "id": forum_id,
                "first": "Max",
                "nextCursor": cursor,
                "sortOrder": "ByTime",
            },
            "query": threads_query(self.reply_depth),
        }
        response = self.http.post_json(self.endpoint, payload)
        page = (response.get("data") or {}).get("getForumRootPostingsV2")
        if not isinstance(page, dict):
            raise HttpFailure(f"missing getForumRootPostingsV2 result for forum {forum_id}")
        return page


def context_uri(story_id: str) -> str:
    return f"https://www.derstandard.at/story/{story_id}"
