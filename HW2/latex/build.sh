#!/bin/bash
pdflatex ConvNet
bibtex ConvNett
pdflatex ConvNet
pdflatex ConvNet
rm -rf *.log
rm -rf *.out
rm -rf *.aux
rm -rf *.bbl
rm -rf *.blg
rm -rf *.gz
#rm body.tex
echo "DONE"

