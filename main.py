import numpy as np
from common_blocks import utils, models, loaders
import os
from torch import optim, nn
import copy
import torch

# global param
data_path = './data/input/'
output_path = './data/output/'
model_save_path = os.path.join(output_path, 'cache/model.pth')
BATCH_SIZE = 128
EPOCHS = 99999
DEVICE = torch.device('cuda:0')
logger = utils.get_logger('tiangong')


def train(model, dataloaders, update_layer, train_on_val=False, lr=1e-4):
    # train specific layer
    for param in update_layer.parameters():
        param.requires_grad = True

    # train
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(DEVICE)
    dataloader_train, dataloader_val = dataloaders
    # todo: change to detect params
    optimizer = optim.Adam(params=model.fc.parameters(), lr=lr)
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 0
    for epoch in range(EPOCHS):
        # train
        model.train()
        running_loss = 0.0
        running_correct = 0
        num = 0
        for images, labels in dataloader_train:
            batch_size = images.size(0)
            num += batch_size
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(images)
                _, labels_ = torch.max(outputs, 1)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_size
                running_correct += torch.sum(labels_ == labels.data)
        epoch_loss = running_loss / num
        epoch_acc = running_correct.double() / num

        # val
        if not train_on_val:
            model.eval()
            running_loss = 0.0
            running_correct = 0
            num = 0
            for images, labels in dataloader_val:
                batch_size = images.size(0)
                num += batch_size
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                with torch.set_grad_enabled(False):
                    outputs = model(images)
                    _, labels_ = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)
                    running_loss += loss.item() * batch_size
                    running_correct += torch.sum(labels_ == labels.data)
            epoch_loss_val = running_loss / num
            epoch_acc_val = running_correct.double() / num
            if epoch_acc_val > best_acc:
                patience = 0
                best_acc = epoch_acc_val
                best_model_wts = copy.deepcopy(model.state_dict())
            else:
                patience += 1
                if patience >= 50:
                    logger.info('early stop')
                    break
            logger.info('train loss: {:.4f} acc: {:.4f}; val loss: {:.4f} acc: {:.4f}'.format(
                epoch_loss, epoch_acc, epoch_loss_val, epoch_acc_val))
        else:
            raise NotImplementedError
    # save
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_save_path)
    logger.info('saved model. best acc:{}'.format(best_acc))
    return model


def predict(model, dataloader):
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            __import__('ipdb').set_trace()
            pass


# data
metadata = loaders.Metadata(data_path)
dataloader_train, dataloader_val, dataloader_test = loaders.get_dataloader(
    metadata, batch_size=BATCH_SIZE)

# get model
model = models.resnet101(
    num_classes=len(metadata.encoder.classes_),
    model_save_path=model_save_path,
).to(DEVICE)

# train & fine tune
update_layers = [model.fc, model.layer4,
                 model.layer3, model.layer2, model.layer1]
lrs = [1e-3, 1e-4, 1e-4, 1e-5, 1e-6]
for layer, lr in zip(update_layers, lrs):
    model = train(
        model=model,
        dataloaders=(dataloader_train, dataloader_val),
        update_layer=layer,
        train_on_val=False,
        lr=lr,
    )
    # logger.info('fine tune on {}'.format(layer.__str__()))
    print('=' * 100)

# predict
pass
