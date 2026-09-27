# How to Use
1. Create venv
2. pip install -r requirements.txt
3. run setup_laya.py

The setup script will pull Laya weights, convert it to ONNX and run the benchmark.py. 
On subsequent runs it will automatically detect and skip pulling weight and converting and directly run the benchmarks.