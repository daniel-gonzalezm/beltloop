#!/usr/bin/env bash
# Build a flat submission folder for the journal's online system: one LaTeX file
# with every \input and the bibliography (.bbl) expanded, and the figure files next
# to it, so the system can compile it without the repository layout.
#
# Run from paper/ after a full build of main.pdf and supplementary.pdf (see the
# header of supplementary.tex) and of highlights.pdf:
#   bash make_submission.sh
# Output: submission/ (git-ignored): manuscript.tex, the six figures (PDF),
# references.bib (for reference only; manuscript.tex carries the bibliography),
# a test compilation manuscript.pdf, which must match main.pdf, the
# supplementary material as supplementary_material.pdf (the journal publishes it
# as received), and the highlights as editable files, highlights.docx (needs
# pandoc) and highlights.txt.
set -euo pipefail
cd "$(dirname "$0")"

[ -f main.bbl ] || { echo "main.bbl missing: build main.pdf first" >&2; exit 1; }
[ -f supplementary.pdf ] || { echo "supplementary.pdf missing: build it after main" >&2; exit 1; }
out=submission
rm -rf "$out"; mkdir -p "$out"

# Expand \input files and the bibliography; drop the comments.
latexpand --empty-comments --expand-bbl main.bbl main.tex > "$out/manuscript.tex"

# latexpand leaves \InputIfFileExists alone: paste paper_numbers.tex in its place.
python3 - "$out/manuscript.tex" <<'EOF'
import re, sys
path = sys.argv[1]
s = open(path, encoding="utf-8").read()
pat = re.compile(r"\\InputIfFileExists\{paper_numbers\}\{\}\{%\n[^\n]*\}\n")
numbers = open("paper_numbers.tex", encoding="utf-8").read()
s, n = pat.subn(lambda m: numbers.rstrip("\n") + "\n", s)
assert n == 1, "paper_numbers input not found"
open(path, "w", encoding="utf-8").write(s)
EOF

# Figures sit next to the .tex file in the submission.
sed -i -e 's|^\\graphicspath{.*}$|% figures are in the same folder|' \
       -e 's|\\IfFileExists{figures/fig_loop.pdf}|\\IfFileExists{fig_loop.pdf}|' "$out/manuscript.tex"
grep -q 'IfFileExists{figures/' "$out/manuscript.tex" && { echo "unhandled figures/ path" >&2; exit 1; }
for f in fig_loop lumped_validation fig_modal fig_beta fig_startup fig_validity; do
  src=""
  for d in ../maps/figures/paper ../validation/figures figures; do
    [ -f "$d/$f.pdf" ] && { src="$d/$f.pdf"; break; }
  done
  [ -n "$src" ] || { echo "figure $f not found" >&2; exit 1; }
  cp "$src" "$out/"
done
cp references.bib "$out/"
cp supplementary.pdf "$out/supplementary_material.pdf"

# Highlights as a separate editable file (the guide asks for "highlights" in the
# file name; 3 to 5 bullets of at most 85 characters, spaces included). The text
# is taken from the compiled highlights.pdf, so the macros are already expanded.
[ -f highlights.pdf ] || { echo "highlights.pdf missing: pdflatex highlights" >&2; exit 1; }
pdftotext -enc UTF-8 highlights.pdf "$out/highlights_raw.txt"
python3 - "$out" <<'EOF'
import pathlib, re, sys
out = pathlib.Path(sys.argv[1])
raw = (out / "highlights_raw.txt").read_text(encoding="utf-8")
items, cur = [], None
for line in raw.splitlines():
    line = line.strip()
    if not line or line == "Highlights":
        continue
    if line.startswith("•"):
        if cur is not None:
            items.append(cur)
        cur = line[1:].strip()
    elif cur is not None:
        cur += " " + line
if cur is not None:
    items.append(cur)
items = [re.sub(r"\s+", " ", s) for s in items]
assert 3 <= len(items) <= 5, f"{len(items)} highlights"
for s in items:
    assert len(s) <= 85, f"{len(s)} characters: {s}"
(out / "highlights_raw.txt").unlink()
(out / "highlights.txt").write_text("".join(f"• {s}\n" for s in items), encoding="utf-8")
(out / "highlights.md").write_text("".join(f"- {s}\n" for s in items), encoding="utf-8")
print("highlights:", ", ".join(str(len(s)) for s in items), "characters")
EOF
if command -v pandoc >/dev/null; then
  ( cd "$out" && pandoc highlights.md -o highlights.docx )
else
  echo "pandoc not found: only highlights.txt written" >&2
fi
rm -f "$out/highlights.md"

# Every graphic named in the file must be present.
grep -o '\\includegraphics\[[^]]*\]{[^}]*}' "$out/manuscript.tex" | sed 's/.*{//;s/}//' |
  while read -r g; do [ -f "$out/$g.pdf" ] || { echo "missing $g.pdf" >&2; exit 1; }; done

# Test compilation of the flat copy.
( cd "$out" && pdflatex -interaction=nonstopmode manuscript >/dev/null &&
  pdflatex -interaction=nonstopmode manuscript >/dev/null &&
  pdflatex -interaction=nonstopmode manuscript >/dev/null )
grep -q "Rerun to get" "$out/manuscript.log" && { echo "cross-references not settled" >&2; exit 1; }
echo "pages: main $(pdfinfo main.pdf | awk '/^Pages/{print $2}'), submission $(pdfinfo $out/manuscript.pdf | awk '/^Pages/{print $2}')"
if cmp -s <(pdftotext main.pdf - ) <(pdftotext "$out/manuscript.pdf" - ); then
  echo "text identical to main.pdf"
else
  echo "WARNING: text differs from main.pdf" >&2; exit 1
fi
