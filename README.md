# Jewel1

This is a repository containing the data and source code used to generate the figures in the main text of "Jewel masks I: non-redundant Fizeau beam combination without the guilt". 

## Quickstart

The code requires python >=3.10. The `requirements.txt` file lists all Python libraries that are required. They can be installed using:

```bash
pip install -r requirements.txt
```

You will also need to install `ehtplot` from source:
    
```bash
git clone https://github.com/liamedeiros/ehtplot.git
cd ehtplot
pip install .
```

The figure to script mapping is the following:
- Figures 1,5: `Fig_1_5.py` (also uses the image data from `data/`)
- Figure 2: `Fig_2.py`
- Figure 6: `m_achromat_design.py`

## Citing

If you find this work helpful, please cite:

```bibtex
@article{10.1117/1.JATIS.11.1.015004,
    author = {Adam K. Taras and Grace Piroscia and Peter Tuthill},
    title = {{Jewel Optics I: non-redundant Fizeau beam combination without the guilt}},
    volume = {11},
    journal = {Journal of Astronomical Telescopes, Instruments, and Systems},
    number = {1},
    publisher = {SPIE},
    pages = {015004},
    year = {2025},
    doi = {10.1117/1.JATIS.11.1.015004},
    URL = {https://doi.org/10.1117/1.JATIS.11.1.015004}
}
```
