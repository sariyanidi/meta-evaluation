python3 -m venv env
source env/bin/activate
pip install --upgrade pip

pip install -r requirements.txt
python table_results.py --num_processes=8

