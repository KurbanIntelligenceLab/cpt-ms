# latexmk configuration for the lab template.
#
# The README documents this file as shipping with every tarball and doing one job:
# skipping BibTeX when no .bib is present, so a tarball containing only the .bbl
# still builds with citations resolved. It was referenced by the Makefile's arxiv
# target but absent from the repository, which made `make arxiv` fail outright.

# pdfLaTeX, matching the \pdfoutput=1 in main.tex that tells arXiv the same thing.
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 %O %S';

# Trust a shipped .bbl when the .bib is not beside main.tex. arXiv's AutoTeX does
# not run Biber and older tarballs shipped only the .bbl; without this, latexmk
# reruns BibTeX, fails to open the database, and the citations come out as [?].
$bibtex_use = 1.5;
$bibtex = 'bibtex %O %B';

# Keep the .bbl -- it is a submission artifact, not a build intermediate.
$cleanup_includes_generated = 0;
push @generated_exts, 'synctex.gz';
$clean_ext = 'aux bbl blg log out synctex.gz fls fdb_latexmk';

# Enough passes for cref/hyperref/natbib to converge on this document.
$max_repeat = 5;
