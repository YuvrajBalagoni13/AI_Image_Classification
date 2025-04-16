import wandb

sweep_config = {
    "method": "random",
    "metrics": {"name": "train_acc",
                "goal": "maximize"},
    "parameters": {
        "train_aug": {
            "values": ["No_Augmentation", "Gaussian_Blur"]
        },
        "test_aug" : {
            "values": ["No_Augmentation"]
        },
        "batch_size" : {
            "values": [25, 50, 100]
        },
        "percentage_data": {
            "values": [10]
        },
        "learning_rate_classifier": {
            "values": [0.01, 0.001, 0.0001]
        },
        "learning_rate_unfreeze": {
            "values": [0.01, 0.001, 0.0001]
        },
        "unfreeze_layers": {
            "values": [3, 5, 7]
        },
        "epochs": {
            "values": [5, 10, 15]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project= "AI_Image_Classification")

def train():
    import sweep_train
    sweep_train.main()

wandb.agent(sweep_id = sweep_id, function = train)