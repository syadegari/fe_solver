LATEXMK ?= latexmk

TEX_SOURCES := $(wildcard *.tex)
PDF_TARGETS := $(TEX_SOURCES:.tex=.pdf)

.PHONY: all rebuild clean

all: $(PDF_TARGETS)

%.pdf: %.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error $<

rebuild:
	$(LATEXMK) -gg -pdf -interaction=nonstopmode -halt-on-error -file-line-error $(TEX_SOURCES)

clean:
	$(LATEXMK) -c $(TEX_SOURCES)
