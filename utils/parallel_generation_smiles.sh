echo $PWD
rm /home/mila/g/golemofl/data/smiles-project/my_smiles.txt
echo "Removed previous submission smiles '/home/mila/g/golemofl/data/smiles-project/my_smiles.txt'"
python utils/generate_smiles.py
python utils/generate_smiles.py
