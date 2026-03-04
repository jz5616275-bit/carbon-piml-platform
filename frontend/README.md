# Carbon PIML Platform

This project demonstrates a small platform for carbon emission forecasting using a combination of machine learning and simple physics-based constraints.

Users can upload a dataset, generate a baseline prediction, apply physics-informed corrections, simulate disturbance scenarios, and view the results through charts.


## Project Structure

backend  
- API endpoints and backend logic  
- prediction pipeline and physics correction  
- data validation utilities  

frontend  
- index.html – upload dataset and configure prediction  
- results.html – display results and charts  
- app.js – frontend interaction logic  


## Technology Stack

Backend  
- Python  
- Flask  
- MongoDB  

Frontend  
- HTML  
- CSS  
- JavaScript  

Authentication  
- Auth0


## Running the Project

### Option 1 (Recommended)

Run the provided script:

```powershell
run_local.ps1
```

This will start the backend server.


### Option 2 (Manual Run)

Start the backend:

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Open the frontend page in your browser:

```
frontend/index.html
```

If the browser blocks requests due to CORS, run a simple static server:

```bash
python -m http.server
```

Then open:

```
http://localhost:8000/frontend/index.html
```


## Notes

- Disturbance sliders use percentages (for example 10% → 0.10).
- Evaluation uses a history holdout split.
- Disturbance is used for scenario analysis rather than ground truth evaluation.
- Login is a demo stub that only affects the UI state.


## Main Features

- Dataset upload  
- Baseline emission prediction  
- Physics-informed correction  
- Disturbance scenario simulation  
- Result visualization