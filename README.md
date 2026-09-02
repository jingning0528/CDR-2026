conda activate "RecBole-CDR-main 2"
python run_recbole_cdr.py

- CMF
Evaluate   : 100%|██████████████████████████████████████████████| 315/315 [00:00<00:00, 3554.29it/s]
28 Aug 16:32    INFO  best valid : OrderedDict([('recall@10', 0.2434), ('mrr@10', 0.3643), ('ndcg@10', 0.2299), ('hit@10', 0.7158), ('precision@10', 0.1419)])
28 Aug 16:32    INFO  test result: OrderedDict([('recall@10', 0.254), ('mrr@10', 0.4305), ('ndcg@10', 0.2656), ('hit@10', 0.7253), ('precision@10', 0.1721)])