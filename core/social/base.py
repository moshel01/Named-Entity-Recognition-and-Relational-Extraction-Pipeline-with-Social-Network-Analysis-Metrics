# Common social-post model + its conversion to pipeline objects. A SocialPost is one
# post/comment/tweet. Two outputs: a Document (the text, so NER/relations still run) and
# the EXPLICIT structure (who replied to / mentioned whom, who posted in which community)
# as asserted edges. Users -> PERSON, communities -> ORG, so the existing tie-class +
# projection machinery applies unchanged (reply = interaction, posted_in = affiliation);
# both are tagged (social_user / social_community) so they stay distinguishable.

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.schema import Document, EntityMention, Relationship, stable_id

# @handle / u/name / r/name mention forms across platforms.
_MENTION_RE = re.compile(r"(?<![\w@])@([A-Za-z0-9_]{2,30})\b|(?<![\w/])/?u/([A-Za-z0-9_\-]{2,30})\b")


def extract_mentions(text: str) -> list[str]:
    """Pull @handles / u/names out of post text (deduped, order-preserving)."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _MENTION_RE.finditer(text or ""):
        name = m.group(1) or m.group(2)
        low = name.lower()
        if name and low not in seen:
            seen.add(low)
            out.append(name)
    return out


@dataclass
class SocialPost:
    """One post / comment / tweet, platform-agnostic."""
    platform: str
    post_id: str
    author: str
    text: str = ""
    title: str = ""
    created_utc: float = 0.0
    parent_id: str = ""            # platform id of the post this replies to
    parent_author: str = ""        # resolved author of the parent (the reply edge target)
    forwarded_from: str = ""       # source account a forward/repost originated in
    community: str = ""            # subreddit / instance / hashtag / list
    mentions: list[str] = field(default_factory=list)
    url: str = ""
    score: int = 0
    extra: dict = field(default_factory=dict)

    def doc_id(self) -> str:
        return stable_id(self.platform, self.post_id, prefix="soc_", length=10)

    def to_document(self) -> Document:
        body = (f"{self.title}\n{self.text}" if self.title else self.text).strip()
        # Prefix the author so a bare NER pass still has the speaker in-text.
        text = f"[{self.author}]: {body}" if self.author else body
        return Document(
            doc_id=self.doc_id(),
            source_path=self.url or f"{self.platform}:{self.post_id}",
            text=text,
            meta={
                "filename": self.url or f"{self.platform}:{self.post_id}",
                "source_type": "social",
                "platform": self.platform,
                "author": self.author,
                "parent_author": self.parent_author,
                "forwarded_from": self.forwarded_from,
                "community": self.community,
                "mentions": list(self.mentions),
                "created_utc": self.created_utc,
                "score": self.score,
                "n_chars": len(text),
            },
        )


def posts_to_documents(posts: list[SocialPost]) -> list[Document]:
    """Resolve parent_author within the batch, then convert to Documents."""
    by_id = {p.post_id: p for p in posts}
    for p in posts:
        if p.parent_id and not p.parent_author and p.parent_id in by_id:
            p.parent_author = by_id[p.parent_id].author
    return [p.to_document() for p in posts]


def _user_mention(name: str, doc_id: str) -> EntityMention:
    return EntityMention(
        text=name, label="PERSON", start_char=0, end_char=0, chunk_id=doc_id,
        doc_id=doc_id, confidence=1.0, sources=["social"],
        attributes={"social_user": True})


def _community_mention(name: str, doc_id: str) -> EntityMention:
    return EntityMention(
        text=name, label="ORG", start_char=0, end_char=0, chunk_id=doc_id,
        doc_id=doc_id, confidence=1.0, sources=["social"],
        attributes={"social_community": True})


def _edge(src: str, tgt: str, rel: str, doc_id: str, directed: bool, note: str) -> Relationship:
    return Relationship(
        source=src, target=tgt, rel_type=rel, doc_id=doc_id, evidence=note,
        confidence=1.0, directed=directed, origin="extracted",
        attributes={"edge_source": "social_graph"})


def social_structure(doc: Document) -> tuple[list[EntityMention], list[Relationship]]:
    """Read a social Document's meta -> (USER/COMMUNITY mentions, explicit edges).

    Edges (all asserted by the platform, edge_source=social_graph):
      replied_to    author -> parent_author   (directed interaction)
      forwarded_from author -> source account  (directed propagation; Telegram forwards)
      mentions      author -> @handle          (directed interaction)
      posted_in     author -> community        (affiliation; feeds projection)
    The structure hook in run_extract appends these to the doc's extraction."""
    meta = doc.meta or {}
    if meta.get("source_type") != "social":
        return [], []
    did = doc.doc_id
    author = (meta.get("author") or "").strip()
    if not author:
        return [], []
    mentions: list[EntityMention] = [_user_mention(author, did)]
    seen_users = {author.lower()}
    edges: list[Relationship] = []

    community = (meta.get("community") or "").strip()
    if community:
        mentions.append(_community_mention(community, did))
        edges.append(_edge(author, community, "posted_in", did, True,
                           f"posted in {community}"))

    parent = (meta.get("parent_author") or "").strip()
    if parent and parent.lower() != author.lower():
        if parent.lower() not in seen_users:
            mentions.append(_user_mention(parent, did)); seen_users.add(parent.lower())
        edges.append(_edge(author, parent, "replied_to", did, True,
                           f"replied to {parent}"))

    fwd = (meta.get("forwarded_from") or "").strip().lstrip("@")
    if fwd and fwd.lower() != author.lower():
        if fwd.lower() not in seen_users:
            mentions.append(_user_mention(fwd, did)); seen_users.add(fwd.lower())
        edges.append(_edge(author, fwd, "forwarded_from", did, True,
                           f"forwarded from {fwd}"))

    for h in (meta.get("mentions") or []):
        h = (h or "").strip().lstrip("@")
        if not h or h.lower() == author.lower():
            continue
        if h.lower() not in seen_users:
            mentions.append(_user_mention(h, did)); seen_users.add(h.lower())
        edges.append(_edge(author, h, "mentions", did, True, f"mentioned @{h}"))
    return mentions, edges


def structure_from_document(doc: Document):
    """Alias kept for the run_extract hook import site."""
    return social_structure(doc)
