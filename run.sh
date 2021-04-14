#!/bin/bash

pip install -r requirements.txt
FLASK_ENV=development
FLASK_APP=app.py flask run