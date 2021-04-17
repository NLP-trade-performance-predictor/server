#!/bin/bash
pip3 install -r requirements.txt
FLASK_ENV=development FLASK_APP=app.py python3 -m flask run