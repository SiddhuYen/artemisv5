"""Name variants: nicknames, initials, transliteration, particles, suffixes.

Two directions matter and both are handled here:
  many people -> one name string (homonyms), and
  one person  -> many name strings (maiden names, Bob/Robert, initials).

Nothing here decides whether two people are the same — that is resolve.py's job
on the evidence ladder. These functions only say whether two strings *could*
denote the same person, which is a precondition, never a conclusion.
"""

from __future__ import annotations

import re
import unicodedata

_TITLES = {
    "mr", "mrs", "ms", "miss", "dr", "prof", "professor", "sir", "dame", "lord",
    "lady", "rev", "reverend", "hon", "capt", "captain", "gen", "col", "lt",
    "sgt", "sen", "senator", "rep", "judge", "justice",
}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "phd", "md", "dds", "esq", "mba", "ma", "ba"}
_PARTICLES = {"de", "del", "della", "di", "da", "van", "von", "der", "den", "ter",
              "la", "le", "bin", "ibn", "al", "st", "mac", "mc", "o"}

#: Common English short forms. Bidirectional at lookup time.
_NICKNAMES: dict[str, set[str]] = {
    "robert": {"bob", "rob", "bobby", "robbie"},
    "william": {"will", "bill", "billy", "willie"},
    "richard": {"rick", "dick", "richie", "rich"},
    "james": {"jim", "jimmy", "jamie"},
    "john": {"jack", "johnny", "jon"},
    "michael": {"mike", "mick", "micky"},
    "christopher": {"chris", "topher"},
    "katherine": {"kate", "katie", "kathy", "kat", "catherine"},
    "elizabeth": {"liz", "beth", "betsy", "eliza", "lizzie", "libby"},
    "margaret": {"maggie", "meg", "peggy", "greta"},
    "patricia": {"pat", "patty", "trish"},
    "jennifer": {"jen", "jenny"},
    "thomas": {"tom", "tommy"},
    "anthony": {"tony"},
    "daniel": {"dan", "danny"},
    "matthew": {"matt"},
    "nicholas": {"nick", "nico"},
    "andrew": {"andy", "drew"},
    "joseph": {"joe", "joey"},
    "edward": {"ed", "eddie", "ted", "teddy", "ned"},
    "charles": {"charlie", "chuck", "chas"},
    "stephen": {"steve", "steven"},
    "alexander": {"alex", "sasha", "xander"},
    "benjamin": {"ben", "benji"},
    "samuel": {"sam", "sammy"},
    "susan": {"sue", "susie", "suzy"},
    "deborah": {"deb", "debbie"},
    "rebecca": {"becca", "becky"},
    "victoria": {"vicky", "tori"},
    "priyanka": {"priya"},
    "abhishek": {"abhi"},
    "abhimanyu": {"abhi"},
    # Added after "Larry Ellison" and "Lawrence Joseph Ellison" were held as two
    # people — the target endpoint fragmenting on a missing nickname is the same
    # failure that lost Drew Glover's network, arriving by a different route.
    "lawrence": {"larry", "laurie", "lars"},
    "laurence": {"larry", "laurie"},
    "theodore": {"ted", "teddy", "theo"},
    "ronald": {"ron", "ronnie"},
    "donald": {"don", "donnie"},
    "kenneth": {"ken", "kenny"},
    "gerald": {"gerry", "jerry"},
    "jeffrey": {"jeff"},
    "gregory": {"greg"},
    "timothy": {"tim", "timmy"},
    "peter": {"pete"},
    "philip": {"phil"},
    "phillip": {"phil"},
    "frederick": {"fred", "freddie"},
    "albert": {"al", "bert"},
    "alfred": {"al", "fred"},
    "arthur": {"art", "artie"},
    "eugene": {"gene"},
    "francis": {"frank", "fran"},
    "harold": {"harry", "hal"},
    "henry": {"hank", "harry"},
    "howard": {"howie"},
    "leonard": {"len", "lenny"},
    "martin": {"marty"},
    "raymond": {"ray"},
    "russell": {"russ"},
    "vincent": {"vince"},
    "walter": {"walt"},
    "zachary": {"zach", "zack"},
    "jonathan": {"jon", "johnny", "jonny"},
    "joshua": {"josh"},
    "nathaniel": {"nate", "nathan"},
    "gabriel": {"gabe"},
    "barbara": {"barb", "babs"},
    "cynthia": {"cindy"},
    "dorothy": {"dot", "dottie"},
    "kimberly": {"kim"},
    "pamela": {"pam"},
    "sandra": {"sandy"},
    "stephanie": {"steph", "steffi"},
    "virginia": {"ginny"},
    "charlotte": {"lottie"},
    "alexandra": {"alex", "sasha", "lexi"},
    "veronica": {"ronnie"},
    "theresa": {"terry", "tess"},
    "teresa": {"terry", "tess"},
}
_NICK_TO_FORMAL: dict[str, set[str]] = {}
for _formal, _nicks in _NICKNAMES.items():
    for _n in _nicks:
        _NICK_TO_FORMAL.setdefault(_n, set()).add(_formal)

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s'-]", re.UNICODE)

# A plausible full personal name: two or three capitalised tokens.
_PERSON_NAME_RE = re.compile(
    r"\b([A-Z][a-z'’\-]{1,20}|[A-Z]\.)"
    r"(?:\s+(?:van|von|de|del|della|di|da|der|den|la|le|bin|ibn|al|Mc|Mac))?"
    r"(?:\s+([A-Z][a-z'’\-]{1,20}|[A-Z]\.)){1,2}\b"
)
# Tokens that make a capitalised bigram an organisation, not a person.
_ORG_MARKERS = {
    "inc", "llc", "ltd", "corp", "corporation", "company", "university",
    "institute", "college", "school", "foundation", "hospital", "center",
    "centre", "group", "partners", "capital", "ventures", "labs", "laboratory",
    "association", "society", "council", "committee", "department", "ministry",
    "agency", "bank", "trust", "press", "journal", "times", "post", "news",
    "street", "avenue", "road", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "january", "february", "march", "april",
    "may", "june", "july", "august", "september", "october", "november",
    "december", "the", "this", "that", "these", "those", "his", "her", "their",
}


def fold(text: str) -> str:
    """Accent-fold and casefold: 'Müller' -> 'muller', 'Ravi Menón' -> 'ravi menon'."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.casefold()


def tokens(name: str) -> list[str]:
    cleaned = _PUNCT_RE.sub(" ", fold(name).replace("’", "'"))
    parts = [p for p in _WS_RE.split(cleaned) if p]
    out: list[str] = []
    for p in parts:
        bare = p.strip(".").strip("'")
        if not bare or bare in _TITLES or bare in _SUFFIXES:
            continue
        out.append(bare)
    return out


def normalize_name(name: str) -> str:
    """Canonical comparable form: titles and suffixes gone, accents folded."""
    return " ".join(tokens(name))


def surname(name: str) -> str:
    t = [x for x in tokens(name) if x not in _PARTICLES]
    return t[-1] if t else ""


def given(name: str) -> str:
    t = tokens(name)
    return t[0] if t else ""


def name_key(name: str) -> str:
    """Grouping key for candidate homonyms: given-initial + surname.

    Deliberately coarse — it decides who gets *considered* for merging, never
    who gets merged.
    """
    g, s = given(name), surname(name)
    return f"{g[:1]}|{s}" if g and s else normalize_name(name)


def _given_forms(g: str) -> set[str]:
    forms = {g}
    forms |= _NICKNAMES.get(g, set())
    forms |= _NICK_TO_FORMAL.get(g, set())
    return forms


def could_be_same_name(a: str, b: str) -> bool:
    """True when two strings could denote one person. Not evidence that they do."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True

    sa, sb = surname(a), surname(b)
    if not sa or not sb or sa != sb:
        return False  # different family names: never a variant in this build

    ga, gb = given(a), given(b)
    if ga == gb:
        return True
    if len(ga) == 1 or len(gb) == 1:  # initial vs full given name
        return ga[:1] == gb[:1]
    return bool(_given_forms(ga) & _given_forms(gb))


def variants(name: str) -> set[str]:
    """Name strings this person may also appear under."""
    g, s = given(name), surname(name)
    out = {normalize_name(name)}
    if g and s:
        for form in _given_forms(g):
            out.add(f"{form} {s}")
        out.add(f"{g[:1]} {s}")
    return out


def name_slug_variants(name: str) -> set[str]:
    """URL-ish spellings of a name: 'priya-raman', 'priyaraman', 'p-raman'."""
    t = tokens(name)
    if not t:
        return set()
    g, s = t[0], t[-1]
    return {"-".join(t), "".join(t), f"{g}-{s}", f"{g}{s}", f"{g[:1]}-{s}", f"{g[:1]}{s}"}


def looks_canonical_for(url: str, name: str) -> bool:
    """Does this URL look like this person's own profile/bio page?

    A personal domain or a /people/<slug> path is the strongest cheap identity
    signal available without a knowledge base to link against.
    """
    slugs = name_slug_variants(name)
    if not slugs:
        return False
    lowered = fold(url)
    host_and_path = lowered.split("://", 1)[-1]
    host = host_and_path.split("/", 1)[0]
    path = host_and_path[len(host) :]
    host_compact = host.replace(".", "").replace("-", "")

    for slug in slugs:
        if slug in path:
            return True
        if slug.replace("-", "") and slug.replace("-", "") in host_compact:
            return True
    return False


#: Tokens that mark a name as an organisation rather than a human. Kept separate
#: from _ORG_MARKERS (which filters the surface name-matching regex) because
#: this list also has to catch company names that look like personal names.
_ORG_NAME_MARKERS = frozenset(
    _ORG_MARKERS
    | {
        "biosciences", "bioscience", "biosystems", "therapeutics", "pharmaceuticals",
        "pharma", "biopharma", "biotech", "diagnostics", "genomics", "sciences",
        "holdings", "technologies", "technology", "systems", "solutions", "networks",
        "industries", "enterprises", "international", "worldwide", "global",
        "ventures", "capital", "equity", "advisors", "consulting", "services",
        "media", "studios", "records", "publishing", "communications",
        "resources", "energy", "motors", "airlines", "railway", "insurance",
        "plc", "gmbh", "ag", "sa", "nv", "bv", "spa", "srl", "pty", "co",
        # Brand-name suffixes common in tech/startup naming.
        "reality", "labs", "lab", "works", "dynamics", "robotics", "ai",
        "analytics", "software", "computing", "cloud", "data", "digital",
        "interactive", "games", "gaming", "studio", "design", "brands",
        "collective", "co-op", "cooperative", "council", "alliance", "network",
        "fund", "funds", "management", "advisers", "securities", "asset",
        # Religious and civic bodies — ProPublica's 990 corpus is full of them.
        "church", "chapel", "parish", "diocese", "ministries", "temple",
        "synagogue", "mosque", "congregation", "fellowship", "mission",
        "charities", "charity", "orphanage", "shelter", "academy", "seminary",
    }
)


def _is_full_personal_name(name: str) -> bool:
    """A human's full name: at least a given name and a family name.

    The extraction prompt already demands full names; enforcing it here is what
    catches single-token brand names like "Niantic" or "Stripe", which carry no
    corporate suffix and would otherwise pass every other test.
    """
    return len(tokens(name)) >= 2


def looks_like_org(name: str) -> bool:
    """True when a 'name' denotes an organisation, not a human.

    The extraction prompt forbids person-to-organisation edges, but a model will
    occasionally emit one anyway ("Brian Murphy -> NEMUS Bioscience"). An org
    admitted as a node becomes a super-connector wiring together everyone who
    works there, which is the same failure mode as name-as-identity.
    """
    raw = name.strip()
    if not raw:
        return True
    if not _is_full_personal_name(raw):
        return True
    lowered = {t.strip(".,").casefold() for t in raw.split()}
    if lowered & _ORG_NAME_MARKERS:
        return True
    letters = [c for c in raw if c.isalpha()]
    # ALL-CAPS multiword strings are org styling, not how people are written.
    if len(letters) > 3 and all(c.isupper() for c in letters) and " " in raw:
        return True
    return False


def find_person_names(text: str) -> list[str]:
    """Candidate full personal names in a block of text.

    Used by degraded (no-Claude) extraction and by seed disambiguation. It is a
    surface pattern, not an NER model: it over-generates on title case and is
    filtered by _ORG_MARKERS.
    """
    found: list[str] = []
    seen: set[str] = set()
    for m in _PERSON_NAME_RE.finditer(text):
        candidate = _WS_RE.sub(" ", m.group(0)).strip()
        parts = candidate.split()
        if len(parts) < 2:
            continue
        if any(p.strip(".").casefold() in _ORG_MARKERS for p in parts):
            continue
        if all(len(p.strip(".")) <= 1 for p in parts):
            continue
        key = normalize_name(candidate)
        if key and key not in seen:
            seen.add(key)
            found.append(candidate)
    return found
