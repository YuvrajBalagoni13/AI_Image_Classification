import torch
import torchvision
import datapreprocess, download_data, engine, model, utils
import wandb
import onnx
from tqdm.auto import tqdm
from pathlib import Path

train_aug = "No_Augmentation"
test_aug = "No_Augmentation"
batch_size = 50
percentage_data = 10
learning_rate_classifier = 0.01
learning_rate_unfreeze = 0.001
model_weights = "EfficientNet_B0_Weights"
model_name = "efficientnet_b0"
unfreeze_layers = 5
num_classes = 2
layer_name = "classifier"
epochs = 5

data_URL = "https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images"
train_dir, test_dir = download_data.download_data(data_URL)

device = "cuda" if torch.cuda.is_available() else "cpu"

with wandb.init(project= "AI_Image_Classification", settings=wandb.Settings(symlink=False)) as run:

    run.config.learning_rate = learning_rate_classifier
    run.config.learning_rate_unfrozenlayer = learning_rate_unfreeze
    run.config.batch_size = batch_size
    run.config.epochs = epochs
    run.config.percentage_data = percentage_data
    run.config.ARCHITECHTURE = model_name

    preprocess = datapreprocess.DataPreprocessor(train_dir= train_dir,
                                                test_dir= test_dir)

    train_dataloader, test_dataloader = preprocess.Build_Dataloaders(train_augmentation= train_aug,
                                                                    test_augmentation= test_aug,
                                                                    num_subsets= 40,
                                                                    batch_size= batch_size,
                                                                    percentage_data= percentage_data)
    
    CNN_model = model.model_builder(model_weights= model_weights,
                                    model_name= model_name,
                                    unfreeze_layers= unfreeze_layers,
                                    num_classes= num_classes,
                                    layer_name= layer_name).to(device)
    
    loss_fn, optimizer = utils.create_loss_and_optim(CNN_model,
                                                     unfreeze_layers,
                                                     learning_rate_classifier,
                                                     learning_rate_unfreeze)
    
    results = { 
            "train loss": [],
            "train acc": [],
            "test loss": [],
            "test acc": []
        }
    
    for epoch in tqdm(range(epochs)):

        train_loss, train_acc, y_train_actual, y_train_predicted = engine.train_loop(CNN_model,
                                                                                     train_dataloader,
                                                                                     loss_fn,
                                                                                     optimizer,
                                                                                     device)
        test_loss, test_acc, y_test_actual, y_test_predicted = engine.test_loop(CNN_model,
                                                                                test_dataloader,
                                                                                loss_fn,
                                                                                device)
        
        results["train loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

        run.log({
            "epoch" : epoch + 1,
            "train_loss" : train_loss,
            "train_accuracy" : train_acc,
            "test_loss" : test_loss,
            "test_accuracy" : test_acc,
        })

        print(f"Epoch {epoch + 1}/{epochs}: train loss: {train_loss:.4f} |\ntrain accuracy: {train_acc:.4f} |\ntest loss: {test_loss:.4f} |\ntest accuracy: {test_acc:.4f}")

        model_path = Path("Models/")
        if model_path.is_dir() == False :
            model_path.mkdir()

        torch.onnx.export(
            CNN_model,
            torch.randn(1,3,224,224, device = device),
            "Models/model.onnx",
            input_names = ["input"],
            output_names = ["output"],
        )

    run.log_artifact("Models/model.onnx", type= "model")
    print("Model training completed.")