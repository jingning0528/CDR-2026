conda activate "RecBole-CDR-main 2"
python run_recbole_cdr.py \
  --model TransformerDTCDR \
  --dataset_preset movielens


- ML
- CMF
Evaluate   : 100%|██████████████████████████████████████████████| 315/315 [00:00<00:00, 3554.29it/s]
28 Aug 16:32    INFO  best valid : OrderedDict([('recall@10', 0.2434), ('mrr@10', 0.3643), ('ndcg@10', 0.2299), ('hit@10', 0.7158), ('precision@10', 0.1419)])
28 Aug 16:32    INFO  test result: OrderedDict([('recall@10', 0.254), ('mrr@10', 0.4305), ('ndcg@10', 0.2656), ('hit@10', 0.7253), ('precision@10', 0.1721)])
- DTCDR
Evaluate   : 100%|███████████████████████████████████| 315/315 [00:00<00:00, 598.52it/s]
04 Sep 15:31    INFO  best valid : OrderedDict([('recall@10', 0.2242), ('recall@20', 0.3389), ('recall@50', 0.5406), ('mrr@10', 0.3323), ('mrr@20', 0.3414), ('mrr@50', 0.3456), ('ndcg@10', 0.2134), ('ndcg@20', 0.2436), ('ndcg@50', 0.3106), ('hit@10', 0.6691), ('hit@20', 0.7996), ('hit@50', 0.9215), ('precision@10', 0.1302), ('precision@20', 0.1035), ('precision@50', 0.072)])
04 Sep 15:31    INFO  test result: OrderedDict([('recall@10', 0.2248), ('recall@20', 0.3521), ('recall@50', 0.5556), ('mrr@10', 0.3808), ('mrr@20', 0.3897), ('mrr@50', 0.3927), ('ndcg@10', 0.2328), ('ndcg@20', 0.2618), ('ndcg@50', 0.3287), ('hit@10', 0.6935), ('hit@20', 0.8197), ('hit@50', 0.9088), ('precision@10', 0.1521), ('precision@20', 0.1202), ('precision@50', 0.0801)])
- Attention DTCDR
Evaluate   : 100%|███████████████████████████████████| 315/315 [00:01<00:00, 172.70it/s]
04 Sep 15:47    INFO  best valid : OrderedDict([('recall@10', 0.2217), ('recall@20', 0.3357), ('recall@50', 0.54), ('mrr@10', 0.3358), ('mrr@20', 0.346), ('mrr@50', 0.35), ('ndcg@10', 0.2073), ('ndcg@20', 0.2386), ('ndcg@50', 0.3058), ('hit@10', 0.6628), ('hit@20', 0.8059), ('hit@50', 0.9215), ('precision@10', 0.1276), ('precision@20', 0.104), ('precision@50', 0.0716)])
04 Sep 15:47    INFO  test result: OrderedDict([('recall@10', 0.2269), ('recall@20', 0.3399), ('recall@50', 0.5412), ('mrr@10', 0.3823), ('mrr@20', 0.3902), ('mrr@50', 0.3938), ('ndcg@10', 0.2315), ('ndcg@20', 0.2561), ('ndcg@50', 0.3211), ('hit@10', 0.6903), ('hit@20', 0.8038), ('hit@50', 0.913), ('precision@10', 0.1495), ('precision@20', 0.1168), ('precision@50', 0.0778)])

- Transformer DTCDR
Evaluate   : 100%|████████████████████████████████████| 315/315 [00:06<00:00, 50.43it/s]
04 Sep 16:15    INFO  best valid : OrderedDict([('recall@10', 0.2333), ('recall@20', 0.3493), ('recall@50', 0.5367), ('mrr@10', 0.3347), ('mrr@20', 0.3435), ('mrr@50', 0.3472), ('ndcg@10', 0.213), ('ndcg@20', 0.2442), ('ndcg@50', 0.3062), ('hit@10', 0.6893), ('hit@20', 0.8144), ('hit@50', 0.9247), ('precision@10', 0.1335), ('precision@20', 0.1074), ('precision@50', 0.0719)])
04 Sep 16:15    INFO  test result: OrderedDict([('recall@10', 0.2309), ('recall@20', 0.3525), ('recall@50', 0.5474), ('mrr@10', 0.3816), ('mrr@20', 0.3883), ('mrr@50', 0.3922), ('ndcg@10', 0.2372), ('ndcg@20', 0.2635), ('ndcg@50', 0.3265), ('hit@10', 0.6967), ('hit@20', 0.7964), ('hit@50', 0.913), ('precision@10', 0.1558), ('precision@20', 0.1213), ('precision@50', 0.0789)])
(RecBole-CDR-main 2) jing@Jings-MacBook-Pro RecBole-CDR % 
- TransformerGPU
Evaluate   : 100%|███████████████████████████████████████████████████████| 315/315 [00:03<00:00, 79.74it/s]
06 Sep 14:47    INFO  best valid : OrderedDict([('recall@10', 0.2266), ('recall@20', 0.3352), ('recall@50', 0.5273), ('mrr@10', 0.3368), ('mrr@20', 0.3454), ('mrr@50', 0.3493), ('ndcg@10', 0.209), ('ndcg@20', 0.237), ('ndcg@50', 0.2998), ('hit@10', 0.684), ('hit@20', 0.8081), ('hit@50', 0.9226), ('precision@10', 0.1295), ('precision@20', 0.103), ('precision@50', 0.0698)])
06 Sep 14:47    INFO  test result: OrderedDict([('recall@10', 0.2229), ('recall@20', 0.3385), ('recall@50', 0.5356), ('mrr@10', 0.3682), ('mrr@20', 0.3767), ('mrr@50', 0.3806), ('ndcg@10', 0.2284), ('ndcg@20', 0.2534), ('ndcg@50', 0.3178), ('hit@10', 0.6723), ('hit@20', 0.7953), ('hit@50', 0.9088), ('precision@10', 0.1499), ('precision@20', 0.1165), ('precision@50', 0.0773)])



- Bookcrossing
- DTCDR
```jsx
02 Sep 04:56    INFO  valid result: 
recall@10 : 0.0541    recall@20 : 0.09    recall@50 : 0.1676    mrr@10 : 0.077    mrr@20 : 0.0837    mrr@50 : 0.0888    ndcg@10 : 0.0433    ndcg@20 : 0.0548    ndcg@50 : 0.0771    hit@10 : 0.1981    hit@20 : 0.2962    hit@50 : 0.4539    precision@10 : 0.0241    precision@20 : 0.0203    precision@50 : 0.0152
02 Sep 04:56    INFO  Finished training, best eval result in epoch 35
02 Sep 04:56    INFO  Loading model structure and parameters from saved/DTCDR-Sep-02-2026_04-54-56.pth

02 Sep 04:56    INFO  best valid : OrderedDict([('recall@10', 0.056), ('recall@20', 0.0892), ('recall@50', 0.161), ('mrr@10', 0.0793), ('mrr@20', 0.0856), ('mrr@50', 0.0905), ('ndcg@10', 0.0447), ('ndcg@20', 0.0552), ('ndcg@50', 0.0763), ('hit@10', 0.2048), ('hit@20', 0.296), ('hit@50', 0.4469), ('precision@10', 0.0246), ('precision@20', 0.0201), ('precision@50', 0.015)])
02 Sep 04:56    INFO  test result: OrderedDict([('recall@10', 0.0525), ('recall@20', 0.0877), ('recall@50', 0.162), ('mrr@10', 0.0751), ('mrr@20', 0.0818), ('mrr@50', 0.0866), ('ndcg@10', 0.0422), ('ndcg@20', 0.0533), ('ndcg@50', 0.0747), ('hit@10', 0.2037), ('hit@20', 0.2993), ('hit@50', 0.4547), ('precision@10', 0.0252), ('precision@20', 0.0207), ('precision@50', 0.0153)])
```

- AttentionDTCDR
```jsx
04 Sep 00:04    INFO  valid result: 
recall@10 : 0.0555    recall@20 : 0.0913    recall@50 : 0.1699    mrr@10 : 0.0766    mrr@20 : 0.083    mrr@50 : 0.0881    ndcg@10 : 0.0444    ndcg@20 : 0.0558    ndcg@50 : 0.0785    hit@10 : 0.2054    hit@20 : 0.3003    hit@50 : 0.4614    precision@10 : 0.0252    precision@20 : 0.0208    precision@50 : 0.0156
04 Sep 00:04    INFO  Finished training, best eval result in epoch 47
04 Sep 00:04    INFO  Loading model structure and parameters from saved/AttentionDTCDR-Sep-03-2026_23-23-03.pth

04 Sep 00:04    INFO  best valid : OrderedDict([('recall@10', 0.0574), ('recall@20', 0.0931), ('recall@50', 0.1653), ('mrr@10', 0.08), ('mrr@20', 0.0864), ('mrr@50', 0.0911), ('ndcg@10', 0.0452), ('ndcg@20', 0.0564), ('ndcg@50', 0.0775), ('hit@10', 0.2114), ('hit@20', 0.3041), ('hit@50', 0.4527), ('precision@10', 0.0255), ('precision@20', 0.0207), ('precision@50', 0.0153)])
04 Sep 00:04    INFO  test result: OrderedDict([('recall@10', 0.055), ('recall@20', 0.0898), ('recall@50', 0.167), ('mrr@10', 0.0796), ('mrr@20', 0.0862), ('mrr@50', 0.091), ('ndcg@10', 0.0446), ('ndcg@20', 0.0553), ('ndcg@50', 0.0775), ('hit@10', 0.2099), ('hit@20', 0.3061), ('hit@50', 0.4601), ('precision@10', 0.0259), ('precision@20', 0.021), ('precision@50', 0.0157)])

```

- TransformerDTCDR

```jsx
03 Sep 23:11    INFO  valid result: 
recall@10 : 0.0579    recall@20 : 0.0939    recall@50 : 0.1758    mrr@10 : 0.0781    mrr@20 : 0.0847    mrr@50 : 0.0899    ndcg@10 : 0.0446    ndcg@20 : 0.0562    ndcg@50 : 0.0801    hit@10 : 0.208    hit@20 : 0.3062    hit@50 : 0.4665    precision@10 : 0.0253    precision@20 : 0.0212    precision@50 : 0.0163
03 Sep 23:11    INFO  Finished training, best eval result in epoch 71
03 Sep 23:11    INFO  Loading model structure and parameters from saved/TransformerDTCDR-Sep-03-2026_22-06-04.pth

03 Sep 23:12    INFO  best valid : OrderedDict([('recall@10', 0.0598), ('recall@20', 0.0928), ('recall@50', 0.1711), ('mrr@10', 0.081), ('mrr@20', 0.0873), ('mrr@50', 0.0922), ('ndcg@10', 0.0468), ('ndcg@20', 0.0574), ('ndcg@50', 0.0801), ('hit@10', 0.2141), ('hit@20', 0.3052), ('hit@50', 0.4629), ('precision@10', 0.0263), ('precision@20', 0.0213), ('precision@50', 0.0159)])
03 Sep 23:12    INFO  test result: OrderedDict([('recall@10', 0.055), ('recall@20', 0.0911), ('recall@50', 0.1747), ('mrr@10', 0.0804), ('mrr@20', 0.0874), ('mrr@50', 0.0926), ('ndcg@10', 0.045), ('ndcg@20', 0.0567), ('ndcg@50', 0.0805), ('hit@10', 0.2088), ('hit@20', 0.3103), ('hit@50', 0.4722), ('precision@10', 0.0257), ('precision@20', 0.0216), ('precision@50', 0.0163)])

```