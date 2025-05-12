import torch
import onnx
import wandb
import tqdm
from pathlib import Path
from HybridModel import datapreprocess, model, download_data

data_URL = "https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images"
train_dir, test_dir = download_data.download_data(data_URL)

device = "cuda" if torch.cuda.is_available() else "cpu"

with wandb.init(project="AI_gen", 
                settings=wandb.Settings(symlink=False, start_method="thread",_disable_meta=True)) as run:
    
    run.config.attn_dropout = 0.1
    run.config.batch_size = 50
    run.config.embedding_dropout = 0
    run.config.epochs = 50
    run.config.lr = 0.0001
    run.config.mlp_dropout = 0
    run.config.mlp_size = 2048
    run.config.num_transformer = 5
    run.config.percentage_data = 100
    run.config.test_aug = "No_Augmentation"
    run.config.train_aug = "Gaussian_Blur"

    config = run.config
    preprocess = datapreprocess.DataPreprocessor(train_dir= train_dir,
                                                test_dir= test_dir)
    
    train_dataloader, test_dataloader = preprocess.Build_Dataloaders(train_augmentation= config.train_aug,
                                                                    test_augmentation= config.test_aug,
                                                                    num_subsets= 40,
                                                                    batch_size= config.batch_size,
                                                                    percentage_data= config.percentage_data)
    
    hybrid_model = model.HybridModel(image_size= 32,
                                    in_channels= 64,
                                    hidden_units= 32,
                                    output_shape= 64,
                                    patch_size= 5,
                                    num_transformer_layers= config.num_transformer,
                                    embedding_dim= 256,
                                    mlp_size= config.mlp_size,
                                    num_heads= 128,
                                    attn_dropout= config.attn_dropout,
                                    mlp_dropout= config.mlp_dropout,
                                    embedding_dropout= config.embedding_dropout,
                                    units= 128,
                                    num_classes= 2
                                    ).to(device)

    optimizer = torch.optim.Adam(hybrid_model.parameters(), lr = config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer= optimizer,
        mode= "min",
        factor= 0.5,
        patience= 5,
        verbose= True
    )
    lrs = []

    loss_func = torch.nn.CrossEntropyLoss()
    
    results = { 
              "train loss": [],
              "train acc": [],
              "test loss": [],
              "test acc": []
          }
    
    for epoch in tqdm(range(config.epochs)):
      hybrid_model.train()
      train_loss, train_acc = 0, 0
      y_train_actual = []
      y_train_predicted = []

      for batch_idx, (x,y) in enumerate(train_dataloader):
        x, y = x.to(device), y.to(device)
        y_pred = hybrid_model(x)
        loss = loss_func(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_class_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_class_pred == y).sum().item() / len(y_pred)

        y_train_predicted.extend(y_class_pred.cpu().numpy())
        y_train_actual.extend(y.cpu().numpy())

      train_loss /= len(train_dataloader)
      train_acc /= len(train_dataloader)

      hybrid_model.eval()
      test_loss, test_acc = 0, 0
      y_test_actual = []
      y_test_predicted = []

      with torch.inference_mode():
        for batch_idx, (x,y) in enumerate(test_dataloader):
          x, y = x.to(device), y.to(device)
          y_pred = hybrid_model(x)
          loss = loss_func(y_pred, y)
          test_loss += loss.item()

          y_class_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
          test_acc += (y_class_pred == y).sum().item() / len(y_pred)

          y_test_predicted.extend(y_class_pred.cpu().numpy())
          y_test_actual.extend(y.cpu().numpy())
        
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

        curr_lr = optimizer.param_groups[0]["lr"]
        lrs.append(curr_lr)
        scheduler.step(test_loss)
      
      results["train loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
      results["train acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
      results["test loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
      results["test acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

      run.log({
              "epoch" : epoch + 1,
              "lr" : config.lr,
              "train_loss" : train_loss,
              "train_accuracy" : train_acc,
              "test_loss" : test_loss,
              "test_accuracy" : test_acc,
          })
      
      print(f"Epoch {epoch + 1}/{config.epochs} & LR {curr_lr}: train loss: {train_loss:.4f} |\ntrain accuracy: {train_acc:.4f} |\ntest loss: {test_loss:.4f} |\ntest accuracy: {test_acc:.4f}")

      model_path = Path("Models/")
      if model_path.is_dir() == False :
          model_path.mkdir()

      torch.onnx.export(
              hybrid_model,
              torch.randn(1,3,32,32, device = device),
              "Models/model.onnx",
              input_names = ["input"],
              output_names = ["output"],
      )
    run.log_artifact("Models/model.onnx", type = "model")
    print("Model training completed.")