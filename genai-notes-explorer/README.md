


****Testing Commands *******
python3 src/loaders.py data
python3 src/chunker.py data --out index/chunks.jsonl --size 1000 --overlap 200
python3 src/embedder.py build --chunks index/chunks.jsonl --out index/ --model text-embedding-3-small --batch 128
python3 src/embedder.py stats --out index/
python3 src/retriever.py --index index/ \
  --query "What metrics did we optimize in the retail purchase project and why?" --k 5

python3 src/cli.py rebuild \
  --data data --chunks index/chunks.jsonl --out index \
  --size 1000 --overlap 200 --embed_model text-embedding-3-small --batch 128

  python3 src/cli.py ask \
  --index index --k 5 \
  --query "What is the plan for week 4-5?"

 python3 src/cli.py ask \
  --index index --k 5 \
  --query "What is the Account number?"

   python3 src/cli.py ask \
  --index index --k 5 \
  --query "how much has been spent for City of Austin?"

Testing :
python3 src/cli.py ask --index index --k 3 --query "What is PageValues?"
python3 src/cli.py ask --index index --k 3 --query "Next steps in Week 4–5 plan?"
python3 src/cli.py ask --index index --k 3 --query "How does Random Forest reduce overfitting?"
python3 src/cli.py ask --index index --k 3 --query "What metrics did we optimize in the retail purchase project and why?"
python3 src/cli.py ask --index index --k 3 --query "how much has been spent for City of Austin?"
python3 src/cli.py ask --index index --k 3 --query "What is the Account number?"