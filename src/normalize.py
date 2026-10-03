"""Record normalization. Country-agnostic: nothing here filters or branches on the country label.

Learned from train pairs only (no external data):
  * INDIC_TOK : Indic-script name token -> Latin token (names are word-by-word translations)
  * INDIC_COMP: Indic-script address component -> Latin component (mostly state names)
  * ADDR_MAP  : address token -> canonical token (abbreviations, state full<->abbrev, frequent typos)
"""
import re
from unidecode import unidecode

INDIC = re.compile(r"[ऀ-෿]")
NONASCII = re.compile(r"[^\x00-\x7f]")
DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:co\.in|com|in|net|org|co|biz|info|fr|us|io)$")
FKA = re.compile(r"\b(?:[dfa]\s*/\s*[bk]\s*/\s*a|dba|fka|aka|formerly|previously|trading as)\b")
MS = re.compile(r"\bm\s*/\s*s\b")
ORD = re.compile(r"(\d+)\s*(?:st|nd|rd|th)\b")
HYPH = re.compile(r"(\d+)\s*-\s*(\d+)")
TOKS = re.compile(r"[a-z]+|\d+")

LEGAL = {
    "inc": "inc", "incorporated": "inc", "llc": "llc", "ltd": "ltd", "limited": "ltd", "pvt": "pvt",
    "private": "pvt", "corp": "corp", "corporation": "corp", "co": "co", "company": "co", "pc": "pc",
    "lp": "lp", "llp": "llp", "pllc": "pllc", "plc": "plc", "lllp": "lllp", "pa": "pa", "public": "public",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci", "ei": "ei",
    "snc": "snc", "gmbh": "gmbh", "cie": "cie",
}
DOMAIN_PEEL = ["private", "limited", "pvt", "ltd", "llc", "llp", "inc", "corp", "sarl", "sas", "eurl", "pllc"]
NAME_STOP = {"the", "of", "and", "a", "an", "de", "du", "des", "la", "le", "les", "et", "d", "l", "ms"}

# generic abbreviations (US/India/French); applied identically to both sides
ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard", "boul": "boulevard", "dr": "drive",
    "ln": "lane", "ct": "court", "cir": "circle", "pl": "place", "pkwy": "parkway", "pky": "parkway",
    "hwy": "highway", "sq": "square", "ter": "terrace", "trl": "trail", "cres": "crescent", "mt": "mount",
    "ft": "fort", "n": "north", "s": "south", "e": "east", "w": "west", "r": "rue", "rte": "route",
    "ch": "chemin", "imp": "impasse", "all": "allee", "fbg": "faubourg", "crs": "cours", "qu": "quai",
    "sec": "sector", "sect": "sector", "ngr": "nagar", "cly": "colony", "mkt": "market", "cplx": "complex",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6", "seventh": "7",
    "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12",
}
ADDR_NOISE = {
    "null", "none", "nan", "na", "no", "nos", "number", "h", "hno", "house", "door", "dno", "flat", "plot",
    "pno", "unit", "apt", "apartment", "ste", "suite", "fl", "floor", "flr", "pmb", "po", "box", "near",
    "nr", "opp", "opposite", "behind", "bldg", "building", "the", "of", "and", "de", "du", "des", "la",
    "le", "les", "d", "l", "et", "bis", "b", "c", "a",
}

# learned maps are installed into these globals (worker initializer)
INDIC_TOK, INDIC_COMP, ADDR_MAP = {}, {}, {}


KEY_SKIP = set(ABBR.values()) | {
    "street", "road", "avenue", "rue", "nagar", "colony", "sector", "block", "main", "cross", "layout",
    "marg", "lane", "phase", "stage", "village", "vill", "post", "dist", "district", "tq", "taluk",
    "industrial", "area", "estate", "complex", "chs", "society", "residency", "tower", "towers",
}


def set_maps(indic_tok=None, indic_comp=None, addr_map=None):
    global INDIC_TOK, INDIC_COMP, ADDR_MAP
    INDIC_TOK = indic_tok or {}
    INDIC_COMP = indic_comp or {}
    ADDR_MAP = addr_map or {}


def _ascii(s):
    return unidecode(s) if NONASCII.search(s) else s


def norm_name(s):
    """-> (core, legal, is_domain, is_indic)"""
    s = s or ""
    is_indic = 1 if INDIC.search(s) else 0
    if is_indic:
        s = " ".join(INDIC_TOK.get(t, t) for t in s.split())
    s = _ascii(s).lower().strip()
    s = FKA.sub(" ", s)
    s = MS.sub(" ", s)
    s = s.replace("&", " and ")
    is_domain = 0
    m = DOMAIN.match(s.strip(" \t\"'"))
    if m:
        is_domain = 1
        s = m.group(1).replace("-", "")
    s = s.replace(".", "").replace("'", "")
    toks = re.findall(r"[a-z0-9]+", s)
    legal, core = set(), []
    for tk in toks:
        c = LEGAL.get(tk)
        if c:
            legal.add(c)
        elif tk not in NAME_STOP:
            core.append(tk)
    if is_domain and len(core) == 1:
        w = core[0]
        changed = True
        while changed:
            changed = False
            for suf in DOMAIN_PEEL:
                if w.endswith(suf) and len(w) - len(suf) >= 3:
                    legal.add(LEGAL[suf]); w = w[: -len(suf)]; changed = True
        core = [w]
    if not core:
        core = [t for t in toks if t not in NAME_STOP] or toks
    return " ".join(core), " ".join(sorted(legal)), is_domain, is_indic


def _addr_tokens(s, use_map=True):
    s = s or ""
    if INDIC.search(s):
        s = ",".join(INDIC_COMP.get(c.strip(), c) for c in s.split(","))
    s = _ascii(s).lower()
    s = re.sub(r"\b1/2\b", " ", s)
    s = ORD.sub(r"\1", s)
    extra = [a.lstrip("0") + b for a, b in HYPH.findall(s)]
    raw = TOKS.findall(s)
    out = []
    for tk in raw:
        if tk.isdigit():
            out.append(tk.lstrip("0") or "0")
            continue
        tk = ABBR.get(tk, tk)
        if use_map:
            tk = ADDR_MAP.get(tk, tk)
        if tk in ADDR_NOISE:
            continue
        out.append(tk)
    return out, extra


def norm_addr(s, use_map=True):
    """-> (addr_n, nums, num1, alpha_toks, key_addr, key_n2, empty)"""
    if s is None or not str(s).strip() or str(s).strip().lower() in ("null", "none", "nan"):
        return "", [], "", [], "", "", 1
    toks, extra = _addr_tokens(s, use_map)
    nums, seen = [], set()
    for tk in toks + extra:
        if tk.isdigit() and tk not in seen:
            seen.add(tk); nums.append(tk)
    num1 = next((tk for tk in toks if tk.isdigit()), "")
    alpha = list(dict.fromkeys(tk for tk in toks if not tk.isdigit()))
    key = ""
    if num1:
        i = toks.index(num1)
        nxt = next((tk for tk in toks[i + 1:] if not tk.isdigit() and tk not in KEY_SKIP and len(tk) > 2), "")
        if nxt:
            key = num1 + "|" + nxt[:4]
    key2 = nums[0] + "|" + nums[1] if len(nums) >= 2 else ""
    return " ".join(toks), nums, num1, alpha, key, key2, 0


def norm_record(name, addr):
    core, legal, dom, ind = norm_name(name)
    an, nums, n1, alpha, key, key2, empty = norm_addr(addr)
    return core, legal, dom, ind, an, nums, n1, alpha, key, key2, empty
