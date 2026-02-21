# Carbon PIML Platform (Frontend)

This folder is a simple 2-page frontend:
- Setup: upload CSV + choose scenario + run
- Results: key numbers + chart + export

## How to run
1) Start backend

2) Open frontend
- Just open `frontend/index.html` in a browser

If CORS blocks requests:
- Ensure backend CORS is enabled in `backend/app.py`
- Use a simple static server (recommended):
  - VSCode Live Server, or
  - `python -m http.server` in project root, then open `/frontend/index.html`

## Notes

- Disturbance sliders are percent-based. The payload sends decimals (e.g. 10% -> 0.10).
- Evaluation is "history holdout". Disturbance is what-if only (no ground truth).
- Login is a demo stub (Level 1). It only changes UI state.