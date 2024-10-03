Use these commands to set up environment:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Change the `DIR_NAME` to the folder with scans you want to start creating spreadsheets.

Run the main Python file to start creating the spreadsheets:

```
python ./src/main.py
```

This code is not perfect and contains inconveniences. However, they are manageable and don't waste *too* much time. If you make a mistake, you may need to redo some minute's worth of work or correct it after the CSV is written.
