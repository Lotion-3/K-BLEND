# SKALE-356

## Setup

Use requires linux terminal or ubuntu WSL.
WSL setup directions
1. Open command prompt run
```bash
wsl --install
```
3. Download ubuntu for WSL https://ubuntu.com/desktop/wsl
4. Open WSL

Virtual Environment setup  
```bash
git clone https://github.com/Lotion-3/SKALE-356
```
```bash
cd SKALE-356
```
```bash
sudo apt update
```
```bash
sudo apt install python3-venv
```
```bash
python3 -m venv SKALE356env
```
```bash
source SKALE356env/bin/activate
```
```bash
pip install -r req.txt
```

## SKALE-356 Usage  
Pre usage setup  
1. .sh Permissions  
```bash
chmod +x SKALE356.sh
```
Run the SKALE356.sh file, this automated script trains 25 ensemble models and evaluated them.  
Evaluation information is saved in the PIPELINE_HPC_RESULTS.csv file.  
Models are saved in saved_models/ directory  
