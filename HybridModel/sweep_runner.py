import wandb

sweep_config = {
    "method": "bayes",
    "metric": {"name": "test_accuracy",
                "goal": "maximize"},
    "parameters": {
        "train_aug": {
            "values": ["No_Augmentation", "Gaussian_Blur"]
        },
        "test_aug" : {
            "values": ["No_Augmentation"]
        },
        "batch_size" : {
            "values": [50]
        },
        "percentage_data": {
            "values": [10]
        },
        "num_transformer": {
            "values": [5, 10]
        },
        "mlp_size": {
            "values": [1024, 2048]
        },
        "attn_dropout": {
            "values": [0, 0.1]
        },	
        "mlp_dropout": {
            "values": [0, 0.1]
        },	
        "embedding_dropout": {
            "values": [0, 0.1]
        },
        "lr": {
            "values": [0.01, 0.001, 0.0001]
        },
        "epochs": {
            "values": [10]
        }
    }
}

sweep_id = wandb.sweep(
          sweep_config,
          project="AI_gen",
          entity="yuvrajbalagoni-indian-institute-of-technology-dhanbad"
)

def train():
    from HybridModel import sweep_train
    sweep_train.main()

wandb.agent(sweep_id, function=train)