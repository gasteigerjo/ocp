# Distill Training

- Baseline

```
python main.py --mode train --config-yml configs/s2ef/200k/painn/painn_h512.yml logger=wandb
```


- node2node Distill

```
python main.py --mode train --config-yml configs/s2ef/200k/painn/painn_h512_n2ndistill.yml logger=wandb
```


- edge2node Distill

```
python main.py --mode train --config-yml configs/s2ef/200k/painn/painn_h512_e2ndistill.yml logger=wandb
```



