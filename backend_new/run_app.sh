# cd backend
nohup python3 app.py > app.log 2>&1 &
echo $! > app.pid
