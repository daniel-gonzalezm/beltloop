#!/usr/bin/env bash
# Build a flat submission folder for the journal's online system: one LaTeX file
# with every \input and the bibliography (.bbl) expanded, and the figure files next
# to it, so the system can compile it without the repository layout.
#
# Run from paper/ after a full build of main.pdf (pdflatex, bibtex, pdflatex x2):
#   bash make_submission.sh
# Output: submission/ (git-ignored): manuscript.tex, the seven figures (PDF),
# references.bib (for reference only; manuscript.tex carries the bibliography),
# and a test compilation manuscript.pdf, which must match main.pdf.
set -euo pipefail
cd "$(dirname "$0")"

[ -f main.bbl ] || { echo "main.bbl missing: build main.pdf first" >&2; exit 1; }
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
for f in fig_loop lumped_validation fig_modal fig_beta fig_startup fig_validity fig_applications; do
  src=""
  for d in ../maps/figures/paper ../validation/figures figures; do
    [ -f "$d/$f.pdf" ] && { src="$d/$f.pdf"; break; }
  done
  [ -n "$src" ] || { echo "figure $f not found" >&2; exit 1; }
  cp "$src" "$out/"
done
cp references.bib "$out/"

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
