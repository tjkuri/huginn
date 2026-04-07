import re
import unicodedata

_NAME_SUFFIX_RE = re.compile(r'\s+(jr|sr|ii|iii|iv)$')


def normalize_name(name: str) -> str:
    """Canonical player name normalizer shared across all MLB data modules.

    Steps applied in order:
    1. Strip Unicode diacritical marks via NFD decomposition (Agustín → Agustin)
    2. Remove periods, e.g. initials (J.C. → JC)
    3. Lowercase and collapse internal/leading/trailing whitespace
    4. Drop trailing name suffixes: Jr, Sr, II, III, IV
    """
    nfkd = unicodedata.normalize('NFD', str(name))
    normalized = ''.join(c for c in nfkd if not unicodedata.category(c).startswith('M'))
    normalized = normalized.replace('.', '')
    normalized = ' '.join(normalized.strip().lower().split())
    normalized = _NAME_SUFFIX_RE.sub('', normalized)
    return normalized
