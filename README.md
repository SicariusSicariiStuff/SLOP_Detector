# SLOP_Detector

(SLOP is now the the acronym for Superfluous Language Overuse Pattern)
SLOP Detector is an analyzer based on dictionary for any JSON or text

# Installation (assuming you use linux):

```shell
git clone https://github.com/SicariusSicariiStuff/SLOP_Detector.git
cd SLOP_Detector
python -m venv env
source env/bin/activate
pip install -r requirements.txt 
```

# Usage: (any file - json, txt etc...)

```shell
python SLOP_Detector.py somefile.json
```

The result will be exported into a text file with the GPTisms found, and a slop score.

  The way the voodoo is being calculated can be adjusted using the YAML files.

  Annoying phrases get penalties because they are annoying.

  Let's nuke GPTisms together.

  Contributions and forks are welcomed.
  
  2 Example files included, a Claude creative writing dataset and a text file story made with Dusk_Rainbow.
