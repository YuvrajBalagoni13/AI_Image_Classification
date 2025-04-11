import torch
from tqdm.auto import tqdm

def train_loop(model : torch.nn.Module,
               train_dataloader : torch.utils.data.DataLoader,
               loss_fn : torch.nn.Module,
               optimizer : torch.optim.Optimizer,
               device : torch.device):
    model.train()
    train_loss, train_acc = 0, 0
    y_train_actual = []
    y_train_predicted = []

    for batch_idx, (x,y) in enumerate(train_dataloader):
        x,y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
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
    return train_loss, train_acc, y_train_actual, y_train_predicted

def test_loop(model : torch.nn.Module,
              test_dataloader : torch.utils.data.DataLoader,
              loss_fn : torch.nn.Module,
              device : torch.device):
    model.eval()
    test_loss, test_acc = 0, 0
    y_test_actual = []
    y_test_predicted = []

    with torch.inference_mode():
        for batch_idx, (x,y) in enumerate(test_dataloader):
            x,y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            y_class_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            test_acc += (y_class_pred == y).sum().item() / len(y_pred)

            y_test_predicted.extend(y_class_pred.cpu().numpy())
            y_test_actual.extend(y.cpu().numpy())

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    return test_loss, test_acc, y_test_actual, y_test_predicted

def exp_train(model : torch.nn.Module,
              train_dataloader : torch.utils.data.DataLoader,
              test_dataloader : torch.utils.data.DataLoader,
              loss_fn : torch.nn.Module,
              optimizer : torch.optim.Optimizer,
              epochs : int,
              device : torch.device):

    results = { 
        "train loss": [],
        "train acc": [],
        "test loss": [],
        "test acc": []
    }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, y_train_actual, y_train_predicted = train_loop(model,
                                                                              train_dataloader,
                                                                              loss_fn,
                                                                              optimizer,
                                                                              device)
        test_loss, test_acc, y_test_actual, y_test_predicted = test_loop(model,
                                                                         test_dataloader,
                                                                         loss_fn,
                                                                         device)

        results["train loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

        print(f"Epoch {epoch + 1}/{epochs}: train loss: {train_loss:.4f} |\ntrain accuracy: {train_acc:.4f} |\ntest loss: {test_loss:.4f} |\ntest accuracy: {test_acc:.4f}")

    return results, y_train_actual, y_train_predicted, y_test_actual, y_test_predicted
