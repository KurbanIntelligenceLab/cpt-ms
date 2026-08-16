#===============================================================================
# Kurban Intelligence Lab -- paper build
#
#   make          build main.pdf (pdflatex -> bibtex -> pdflatex x2)
#   make arxiv    arxiv-submission.tar.gz, ready to upload
#   make blind    main-blind.pdf   (double-blind: no authors, no logo, no acks)
#   make logo     regenerate every logo asset from the vector master
#   make figures  regenerate the example figure
#   make clean    remove build artefacts (keeps the PDFs)
#   make purge    remove everything generated
#===============================================================================
MAIN  := main
LATEX := pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all arxiv blind logo figures clean purge
all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex kurbanlab.sty affiliations.tex references.bib
	$(LATEX) $(MAIN)
	-bibtex $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)
	@echo "==> $(MAIN).pdf"

# Build the upload. arXiv processes .bib into .bbl itself since November 2025, so the .bib
# is the file that matters; we ship the .bbl too because it is small and harmless.
arxiv: $(MAIN).pdf
	@rm -rf arxiv && mkdir -p arxiv/figures/logo arxiv/sections
	@cp $(MAIN).tex kurbanlab.sty affiliations.tex references.bib .latexmkrc $(MAIN).bbl arxiv/
# The manuscript is split across sections/*.tex via \input, so the directory must travel
# with it or the build has a preamble and nothing else.
	@cp sections/*.tex arxiv/sections/
# Ship only the figures the manuscript actually \includegraphics, so the template's
# example-*.pdf placeholders do not travel to arXiv as unreferenced files. Scan main.tex
# AND sections/ -- every float now lives in a section file, so scanning main.tex alone
# matches nothing and ships an empty figures/ directory.
	@for f in $$(grep -ohE 'kilgraphics(\[[^]]*\])?\{figures/[^}]+\}' $(MAIN).tex sections/*.tex \
	            | sed -E 's/.*\{figures\/([^}]+)\}/\1/' | sort -u); do \
	    cp "figures/$$f" arxiv/figures/ 2>/dev/null || true; \
	done
# The lockup lives under figures/logo/ -- kurbanlab.sty says an earlier logo/ path
# failed \IfFileExists silently, so copying from logo/ shipped a tarball with the
# wordmark missing and no error anywhere.
	@cp figures/logo/kil-lockup.png figures/logo/kil-lockup.pdf arxiv/figures/logo/ 2>/dev/null || true
	@cd arxiv && tar czf ../arxiv-submission.tar.gz .
	@rm -rf arxiv
	@echo "==> arxiv-submission.tar.gz  (upload this)"
	@echo "    contains: $(MAIN).tex, sections/, kurbanlab.sty, affiliations.tex,"
	@echo "              references.bib, $(MAIN).bbl, figures/, figures/logo/"
	@echo "    NOT included: *.aux, *.log, *.blg, .DS_Store."
	@echo "    references.bib is the file arXiv now uses (it runs BibTeX since Nov 2025);"
	@echo "    $(MAIN).bbl ships too, so a no-BibTeX build still resolves citations."

blind:
	@sed -E 's/\\usepackage\[([^]]*)\]\{kurbanlab\}/\\usepackage[\1, blind]{kurbanlab}/' \
	     $(MAIN).tex > $(MAIN)-blind.tex
	$(LATEX) $(MAIN)-blind
	-bibtex $(MAIN)-blind
	$(LATEX) $(MAIN)-blind
	$(LATEX) $(MAIN)-blind
	@echo "==> $(MAIN)-blind.pdf  (check it: your name should appear nowhere)"

logo:
	@cd logo && sh build-logo.sh

figures:
	@cd figures && python3 make-figure.py

# Remove build intermediates.
#
# BIBLIOGRAPHY: as of November 2025 arXiv processes .bib into .bbl itself, so the .bib is
# what the submission needs and a precompiled .bbl is no longer required. (arXiv's own
# LEGACY help pages still say the opposite -- they are stale. See blog.arxiv.org.) We keep
# main.bbl out of `clean` anyway: it costs a few kB in the tarball and covers a build that
# resolves citations without a BibTeX pass. `make purge` removes it.
#
# The globs below only reach the project root, so sweep the subdirectories too: stray
# per-run logs from manual pdflatex calls, aux files under sections/, and .DS_Store all
# survive `rm -f *.aux *.log` and would otherwise travel into the tarball.
clean:
	@rm -f *.aux *.log *.out *.blg *.toc *.lof *.lot *.fls *.fdb_latexmk \
	       *.synctex.gz *.nav *.snm *.vrb $(MAIN)-blind.tex
	@rm -f bibtex.log run*.log texput.log
	@rm -f sections/*.aux sections/*.log
	@rm -f .DS_Store figures/.DS_Store sections/.DS_Store
	@echo "==> cleaned (main.bbl kept: arXiv needs it)"

purge: clean
	@rm -f $(MAIN).pdf $(MAIN)-blind.pdf arxiv-submission.tar.gz $(MAIN).bbl
	@rm -rf arxiv
