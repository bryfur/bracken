#!/bin/sh
# logdiff.sh — message-body preservation check for the single-log-channel work.
#
# Converting a log call changes its macro, its severity and its tag, so
# "diff the smoke output" cannot tell an intended severity change from a
# message that was accidentally dropped or mangled. This checks the one thing
# that must NOT change: the set of message bodies the engine can emit.
#
# It extracts every string literal in the non-test engine sources (runs of
# adjacent literals are concatenated the way the compiler would), normalizes
# away everything this migration is allowed to change — printf specifiers vs
# std::format placeholders, a leading "[Tag]", an "ERROR:"/"Warning:" word now
# carried by the level — and diffs the resulting multiset against a git rev.
# Unrelated literals appear identically on both sides and cancel out.
#
# Usage: tools/logdiff.sh [<rev>]      (default: HEAD)
set -eu

REV="${1:-HEAD}"
cd "$(git rev-parse --show-toplevel)"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

extract() {
    perl -0777 -ne '
        # Fold ostream interpolation into format-string shape, so a
        # `cerr << "a " << x << " b"` and a `LOG(tag, "a {} b", x)` normalize
        # to the same body. Applied to both sides, so ordinary ostream code
        # elsewhere folds identically and cancels.
        s/\s*<<\s*\x27[^\x27]*\x27//g;   # drop `<< \x27\\n\x27` style char emitters
        1 while s/"\s*<<\s*[^;"]*?\s*<<\s*"/{}/s;
        s/"\s*<<\s*[^;"]*?\s*;/{}";/gs;

        while (/((?:"(?:[^"\\]|\\.)*"\s*)+)/gs) {
            my $run = $1; my $m = "";
            $m .= $1 while $run =~ /"((?:[^"\\]|\\.)*)"/gs;
            $m =~ s/\s*\b(?:ERROR|Error|WARNING|Warning)\s*[:-]\s*/ /g;
            $m =~ s/^\s*\[[^\]]*\]\s*//;
            $m =~ s/%[-+ #0]*[0-9]*(?:\.[0-9]+)?(?:zu|zd|llu|lld|lu|ld|hu|hd|[udisfgpxXc])/{}/g;
            $m =~ s/\{[^{}]*\}/{}/g;
            $m =~ s/\\n/ /g; $m =~ s/\s+/ /g; $m =~ s/^ | $//g;
            next unless $m =~ / /;   # message bodies contain spaces; tags and paths do not
            print "$m\n";
        }
    '
}

git ls-tree -r --name-only "$REV" -- src/bracken |
    grep -E '\.(cpp|hpp)$' | grep -v '\.test\.' |
    while read -r f; do git show "$REV:$f" | extract; done | sort > "$TMP/before"

git ls-files -- src/bracken |
    grep -E '\.(cpp|hpp)$' | grep -v '\.test\.' |
    while read -r f; do [ -f "$f" ] && extract < "$f"; done | sort > "$TMP/after"

if diff -u "$TMP/before" "$TMP/after" > "$TMP/delta"; then
    echo "logdiff: message bodies unchanged vs $REV ($(wc -l < "$TMP/after" | tr -d ' ') messages)"
    exit 0
fi

echo "logdiff: message bodies CHANGED vs $REV ('-' lost, '+' gained):"
grep -E '^[-+][^-+]' "$TMP/delta" || true
exit 1
