# SLOP_Detector
SLOP Detector and analyzer based on dictionary for shareGPT JSON and text

# Installation (assuming you use linux):

```shell
git clone https://github.com/SicariusSicariiStuff/SLOP_Detector.git
cd SLOP_Detector
python -m venv env
source env/bin/activate
pip install -r requirements.txt 
```

# Usage:

```shell
python SLOP_Detector.py somefile.json
```

The result will be exported into a text file with the GPTisms found, and a slop score.

  The way the voodoo is being calculated can be adjusted using the YAML files.

  Annoying phrases get penalties because they are annoying.

  Let's nuke GPTisms together.

  Contributions and forks are welcomed.
