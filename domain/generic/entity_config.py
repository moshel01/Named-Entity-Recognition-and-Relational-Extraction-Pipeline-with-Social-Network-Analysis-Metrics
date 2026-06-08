# Entity-type configuration for the generic domain.

from __future__ import annotations

# Forced label overrides keyed by lowercased entity name.
LABEL_OVERRIDES: dict[str, str] = {}

# Default canonical entity types this domain expects (informational).
DEFAULT_TYPES: list[str] = ["PERSON", "ORG", "LOCATION", "EVENT"]

# Common English nouns the NER/LLM mislabels as entities. Exact normalized-name
# match, so specific names ("King Street") survive. Keeps generic-text output
# (books, articles, scraped pages) free of "the man / the road / morning" nodes.
STOPWORDS: set[str] = {
    "man", "woman", "men", "women", "boy", "girl", "child", "children", "baby",
    "people", "person", "everyone", "someone", "anybody", "nobody", "guy",
    "family", "father", "mother", "dad", "mom", "son", "daughter", "brother",
    "sister", "parents", "wife", "husband", "friend", "friends", "neighbor",
    "king", "queen", "lord", "lady", "master", "sir", "madam", "mister",
    "day", "days", "night", "nights", "morning", "afternoon", "evening", "today",
    "tomorrow", "yesterday", "time", "times", "year", "years", "week", "month",
    "hour", "minute", "moment", "summer", "winter", "spring", "autumn", "season",
    "world", "earth", "home", "house", "room", "door", "place", "places", "way",
    "road", "path", "town", "land", "ground", "sky", "sea", "mountain", "river",
    "thing", "things", "something", "nothing", "anything", "everything", "stuff",
    "life", "death", "love", "war", "peace", "work", "money", "food", "water",
    "fire", "light", "hand", "hands", "head", "eye", "eyes", "face", "feet",
    "heart", "voice", "name", "word", "words", "story", "book", "letter",
    "group", "side", "part", "end", "kind", "sort", "number", "people",
}

# General-purpose subtype vocabulary handed to the LLM enricher as a controlled
# list. Domains with sharper categories (nazi_era) override this.
ENTITY_SUBTYPES: dict[str, list[str]] = {
    "PERSON": ["leader", "official", "military", "religious", "professional",
               "artist", "family_member", "public_figure", "other"],
    "ORG": ["government", "company", "political", "military", "religious",
            "educational", "media", "social", "criminal", "other"],
    "LOCATION": ["city", "region", "country", "settlement", "building",
                 "geographic_feature", "venue", "other"],
    "EVENT": ["battle", "ceremony", "political_event", "disaster", "gathering",
              "journey", "other"],
    "INSTITUTION": ["government", "educational", "religious", "judicial",
                    "media", "financial"],
}
