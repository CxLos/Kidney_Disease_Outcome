# =========================== Imports ========================== #

import os
import sys

# Ensure project root is on sys.path when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash

from app.layouts.layout import create_layout

# =========================== DASH APP ========================== #

assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')
app = dash.Dash(__name__, assets_folder=assets_folder)
server = app.server

app.layout = create_layout()

# =========================== RUN APP ========================== #

if __name__ == '__main__':  # pragma: no cover
    current_file = os.path.basename(__file__)
    print(f"Serving Flask app '{current_file}'! 🚀")
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=True)

# ----------------------- KILL PORT -------------------------- #

# netstat -ano | findstr :8050
# taskkill /PID 24772 /F
# npx kill-port 8050

# -------------------- Host Application ------------------------ #

# 1. pip freeze > requirements.txt
# 2. add this to procfile: 'web: gunicorn app.kidney_disease:server'

# python -m venv venv
# source venv/Scripts/activate
# pip install -r requirements.txt

# ------------ Heroku ---------------- #

# heroku login
# heroku create kidney-disease-outcome
# heroku git:remote -a kidney-disease-outcome
# git push heroku main
# heroku buildpacks:set heroku/python
