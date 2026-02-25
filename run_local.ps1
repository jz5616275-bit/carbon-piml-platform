# Start backend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "py -m backend.app"

# Start frontend 
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; py -m http.server 5500"