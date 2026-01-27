import wandb
import random

wandb.login()

project = "dickinson-comp560-JohnLee-tutorial"

config = {
    'epochs' : 10,
    'lr' : 0.01
}

with wandb.init(project=project, config=config) as run:
    offset = random.random() / 5
    
    for epoch in range(1, config['epochs'] + 1):
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset
        print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
        run.log({"accuracy": acc, "loss": loss})