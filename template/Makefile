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

# arXiv wants the .bbl, not the .bib, and no build cruft.
arxiv: $(MAIN).pdf
	@rm -rf arxiv && mkdir -p arxiv/figures/logo
	@cp $(MAIN).tex kurbanlab.sty affiliations.tex references.bib .latexmkrc $(MAIN).bbl arxiv/
# Ship only the figures this manuscript actually \includegraphics, so the template's
# example-*.pdf placeholders do not travel to arXiv as unreferenced files.
	@for f in $$(grep -oE 'kilgraphics(\[[^]]*\])?\{figures/[^}]+\}' $(MAIN).tex \
	            | sed -E 's/.*\{figures\/([^}]+)\}/\1/'); do \
	    cp "figures/$$f" arxiv/figures/ 2>/dev/null || true; \
	done
# The lockup lives under figures/logo/ -- kurbanlab.sty says an earlier logo/ path
# failed \IfFileExists silently, so copying from logo/ shipped a tarball with the
# wordmark missing and no error anywhere.
	@cp figures/logo/kil-lockup.png figures/logo/kil-lockup.pdf arxiv/figures/logo/ 2>/dev/null || true
	@cd arxiv && tar czf ../arxiv-submission.tar.gz .
	@rm -rf arxiv
	@echo "==> arxiv-submission.tar.gz  (upload this)"
	@echo "    contains: $(MAIN).tex, kurbanlab.sty, affiliations.tex, $(MAIN).bbl, figures/, logo/"
	@echo "    NOT included: *.aux, *.log. references.bib IS included: arXiv ignores it,"
	@echo "    but anyone opening this tarball in Overleaf gets a working BibTeX setup."

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

clean:
	@rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot *.fls *.fdb_latexmk \
	       *.synctex.gz *.nav *.snm *.vrb $(MAIN)-blind.tex
	@echo "==> cleaned"

purge: clean
	@rm -f $(MAIN).pdf $(MAIN)-blind.pdf arxiv-submission.tar.gz
	@rm -rf arxiv
