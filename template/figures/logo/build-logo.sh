#!/usr/bin/env bash
# Regenerates every logo asset from the single vector master (kil-logo.tex).
set -e
cd "$(dirname "$0")"
for v in lockup lockup-glow lockup-mono lockup-white stacked mark mark-mono mark-white; do
  pdflatex -interaction=batchmode -halt-on-error \
    -jobname="kil-$v" "\def\kilvariant{$v}\input{kil-logo.tex}" > /dev/null
  echo "  PDF  kil-$v.pdf"
done
# High-res PNGs (transparent) for slides, posters, GitHub, favicons
for v in lockup lockup-glow lockup-white mark mark-white; do
  pdftocairo -png -r 600 -transp -singlefile "kil-$v.pdf" "kil-$v"
  echo "  PNG  kil-$v.png"
done
rm -f *.aux *.log
