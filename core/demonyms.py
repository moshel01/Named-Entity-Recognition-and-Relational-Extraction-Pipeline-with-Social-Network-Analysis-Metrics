# Demonym -> place mapping. NER tags nationality adjectives as NORP
# ("American", "Russian"); benchmarks and SNA practice treat them as references
# to the place/state actor. The table relabels them LOCATION and lets dedup
# fold them into the country node as aliases ("American" -> "United States").
# Lowercased keys; plural "s" is stripped by the lookup helper.

from __future__ import annotations

DEMONYM_TO_PLACE: dict[str, str] = {
    # Major states (English adjectivals)
    "american": "United States", "british": "United Kingdom",
    "english": "England", "scottish": "Scotland", "welsh": "Wales",
    "irish": "Ireland", "french": "France", "german": "Germany",
    "italian": "Italy", "spanish": "Spain", "portuguese": "Portugal",
    "dutch": "Netherlands", "belgian": "Belgium", "swiss": "Switzerland",
    "austrian": "Austria", "danish": "Denmark", "swedish": "Sweden",
    "norwegian": "Norway", "finnish": "Finland", "icelandic": "Iceland",
    "polish": "Poland", "czech": "Czech Republic", "slovak": "Slovakia",
    "hungarian": "Hungary", "romanian": "Romania", "bulgarian": "Bulgaria",
    "greek": "Greece", "turkish": "Turkey", "russian": "Russia",
    "ukrainian": "Ukraine", "belarusian": "Belarus", "lithuanian": "Lithuania",
    "latvian": "Latvia", "estonian": "Estonia", "serbian": "Serbia",
    "croatian": "Croatia", "bosnian": "Bosnia and Herzegovina",
    "slovenian": "Slovenia", "albanian": "Albania", "macedonian": "North Macedonia",
    "chinese": "China", "japanese": "Japan", "korean": "Korea",
    "indian": "India", "pakistani": "Pakistan", "bangladeshi": "Bangladesh",
    "afghan": "Afghanistan", "iranian": "Iran", "persian": "Persia",
    "iraqi": "Iraq", "syrian": "Syria", "lebanese": "Lebanon",
    "israeli": "Israel", "palestinian": "Palestine", "saudi": "Saudi Arabia",
    "egyptian": "Egypt", "libyan": "Libya", "tunisian": "Tunisia",
    "algerian": "Algeria", "moroccan": "Morocco", "ethiopian": "Ethiopia",
    "kenyan": "Kenya", "nigerian": "Nigeria", "ghanaian": "Ghana",
    "south african": "South Africa", "congolese": "Congo",
    "mexican": "Mexico", "canadian": "Canada", "cuban": "Cuba",
    "brazilian": "Brazil", "argentine": "Argentina", "argentinian": "Argentina",
    "chilean": "Chile", "peruvian": "Peru", "colombian": "Colombia",
    "venezuelan": "Venezuela", "bolivian": "Bolivia", "paraguayan": "Paraguay",
    "uruguayan": "Uruguay", "ecuadorian": "Ecuador",
    "australian": "Australia", "indonesian": "Indonesia",
    "malaysian": "Malaysia", "filipino": "Philippines", "thai": "Thailand",
    "vietnamese": "Vietnam", "cambodian": "Cambodia", "burmese": "Myanmar",
    "mongolian": "Mongolia", "kazakh": "Kazakhstan", "uzbek": "Uzbekistan",
    "armenian": "Armenia", "georgian": "Georgia", "azerbaijani": "Azerbaijan",
    "barbadian": "Barbados", "jamaican": "Jamaica", "haitian": "Haiti",
    "dominican": "Dominican Republic",
    # Historical / supranational (Abel-era relevant)
    "soviet": "Soviet Union", "prussian": "Prussia", "bavarian": "Bavaria",
    "saxon": "Saxony", "rhinelander": "Rhineland", "silesian": "Silesia",
    "pomeranian": "Pomerania", "westphalian": "Westphalia",
    "yugoslav": "Yugoslavia", "czechoslovak": "Czechoslovakia",
    "ottoman": "Ottoman Empire", "bohemian": "Bohemia",
    # German-language demonyms (Abel corpus)
    "deutsche": "Deutschland", "deutscher": "Deutschland",
    "preuße": "Preußen", "preusse": "Preußen",
    "bayer": "Bayern", "sachse": "Sachsen", "schlesier": "Schlesien",
    "österreicher": "Österreich", "franzose": "Frankreich",
    "engländer": "England", "russe": "Russland", "amerikaner": "USA",
}


def demonym_place(surface: str) -> str | None:
    """Return the place for a demonym surface form, or None."""
    s = surface.strip().lower()
    hit = DEMONYM_TO_PLACE.get(s)
    if hit is None and len(s) > 4 and s.endswith("s"):
        hit = DEMONYM_TO_PLACE.get(s[:-1])      # plural: "Americans"
    return hit
